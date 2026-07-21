"""类似 maddpg4test.py 的符号推理运行入口，默认推理并实际执行。"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from log import Log

from .agent import (
    Conclusion,
    Decision,
    ReasoningFacts,
    RETURN_FUEL_THRESHOLD_PCT,
    SymbolicReasoningAgent,
    TargetEvaluation,
)
from .control import FrontendControl
from .entity import (
    EncodedEntity,
    EncodedSituation,
    EntityEncoder,
    TargetDomain,
    load_situation,
)
from .mission import load_project_mission_areas
from .live import RpcSituationSource
from .state import (
    FIRE_CONTROL_REJECTION_COOLDOWN_FRAMES,
    TARGET_CONTACT_LOSS_GRACE_FRAMES,
    EngagementState,
)
from .execute_actions import (
    DEFAULT_ATTACK_QUANTITY,
    DEFAULT_ATTACK_RPC_TARGET,
    WeaponFirePrecheck,
    WeaponFirePrecheckBackend,
    execute_actions,
    precheck_weapon_fire,
)


MISSILE_EVADE_DISTANCE_KM = 5.0
MISSILE_INTERCEPT_PRIORITY_DISTANCE_KM = 50.0
MISSILE_POINTING_TOLERANCE_DEG = 30.0
EVADE_ROUTE_LENGTH_KM = 5.0
AIM_TOLERANCE_DEG = 30.0


@dataclass(frozen=True)
class SymbolicStepResult:
    situation: EncodedSituation
    facts: Dict[str, ReasoningFacts]
    decisions: Dict[str, Decision]
    actions_dict: Dict[str, List[List[Any]]]
    execute_results: Any = None
    rewards: Any = None
    execution_status: Optional[Dict[str, str]] = None
    cooldown_frame: int = 0
    time_compression_multiplier: int = 1


@dataclass(frozen=True)
class PatrolContext:
    is_patrol_aircraft: bool
    has_patrol_mission: bool
    inside_patrol_area: bool
    route_lons: Tuple[float, ...] = ()
    route_lats: Tuple[float, ...] = ()
    altitude_level: int = 1


class SymbolicReasoningEnv:
    """完成“编码 → 校验 → 规则匹配 → 推理 → 默认实际执行”。"""

    def __init__(
        self,
        max_entities: int = 700,
        aim_tolerance_deg: float = AIM_TOLERANCE_DEG,
        missile_intercept_distance_km: float = (
            MISSILE_INTERCEPT_PRIORITY_DISTANCE_KM
        ),
        mission_areas: Optional[Mapping[str, Any]] = None,
        engagement_state: Optional[EngagementState] = None,
        weapon_fire_prechecker: Optional[WeaponFirePrecheckBackend] = None,
    ) -> None:
        if not 0.0 < aim_tolerance_deg <= 180.0:
            raise ValueError("aim_tolerance_deg 必须位于 (0, 180]")
        if missile_intercept_distance_km <= 0.0:
            raise ValueError("missile_intercept_distance_km 必须大于 0")
        if weapon_fire_prechecker is not None and not callable(
            weapon_fire_prechecker
        ):
            raise ValueError("weapon_fire_prechecker 必须可调用或为 None")
        self.encoder = EntityEncoder(max_entities=max_entities)
        self.agent = SymbolicReasoningAgent()
        self.aim_tolerance_deg = float(aim_tolerance_deg)
        self.missile_intercept_distance_km = float(
            missile_intercept_distance_km
        )
        self.mission_areas = dict(mission_areas or {})
        self.engagement_state = engagement_state or EngagementState()
        self.weapon_fire_prechecker = weapon_fire_prechecker
        self._geometry_cache: Dict[int, Tuple[float, float]] = {}
        self.current_step = 0

    def step(
        self,
        payload: Mapping[str, Any],
        execute_commands: bool = True,
        logger: Any = None,
        executor: Any = None,
        time_compression_multiplier: int = 1,
    ) -> SymbolicStepResult:
        if (
            not isinstance(time_compression_multiplier, int)
            or isinstance(time_compression_multiplier, bool)
            or time_compression_multiplier <= 0
        ):
            raise ValueError("time_compression_multiplier 必须是正整数")
        situation = self.encoder.encode(payload)
        # 冷却时钟按当前 UI 时间压缩倍率推进。例如 50 倍速时，每处理一份
        # 新态势，600 帧剩余量减少 50；暂停期间不调用 step，因此不会推进。
        current_frame = self.current_step + time_compression_multiplier
        self.current_step = current_frame
        self.engagement_state.update_from_situation(situation, current_frame)
        facts = self.build_facts(situation)

        if execute_commands:
            run_result = self.agent.run(
                facts.values(),
                enemy_ids=self._enemy_ids(situation.targets),
                logger=logger,
                executor=executor,
            )
            decisions = run_result.decisions
            actions_dict = run_result.actions_dict
            execute_results = run_result.execute_results
            rewards = run_result.rewards
            execution_status = run_result.execution_status
            self._record_successful_attacks(
                facts, decisions, execution_status, current_frame
            )
            self._record_successful_lifecycle_actions(
                decisions, execution_status, current_frame
            )
        else:
            decisions, actions_dict = self.agent.build_actions(facts.values())
            execute_results = None
            rewards = None
            execution_status = {
                entity_id: "DRY_RUN" for entity_id in decisions
            }

        return SymbolicStepResult(
            situation=situation,
            facts=facts,
            decisions=decisions,
            actions_dict=actions_dict,
            execute_results=execute_results,
            rewards=rewards,
            execution_status=execution_status,
            cooldown_frame=current_frame,
            time_compression_multiplier=time_compression_multiplier,
        )

    def build_facts(
        self, situation: EncodedSituation
    ) -> Dict[str, ReasoningFacts]:
        """构造事实，并在同一帧内原子预留并发及拦截弹名额。"""

        facts_by_entity: Dict[str, ReasoningFacts] = {}
        targets = tuple(situation.targets)
        planned_attackers: Dict[str, Set[str]] = {}
        planned_interceptors: Dict[str, int] = {}

        for own in sorted(situation.own_entities, key=lambda item: item.command_id):
            fuel_low = (
                own.is_aircraft
                and own.fuel_percentage >= 0.0
                and own.fuel_percentage <= RETURN_FUEL_THRESHOLD_PCT
            )
            ammunition_low = (
                own.is_aircraft
                and own.has_strike_weapon_system
                and own.strike_weapon_count <= 0
            )
            current_attack_target_id = (
                self.engagement_state.attack_target_for(own.command_id)
            )
            current_attack_target = next(
                (
                    target
                    for target in targets
                    if current_attack_target_id
                    and self.engagement_state.same_target(
                        current_attack_target_id,
                        target.command_id,
                        target.entity_id,
                    )
                ),
                None,
            )
            currently_attacking = current_attack_target_id is not None
            attack_target_missing_frames = (
                self.engagement_state.target_missing_frames(
                    own.command_id, current_attack_target_id
                )
                if current_attack_target_id is not None
                else 0
            )
            target_loss_within_grace = (
                attack_target_missing_frames
                < self.engagement_state.target_contact_loss_grace_frames
            )
            lifecycle_facts = {
                "is_aircraft": own.is_aircraft,
                "is_airborne": own.is_airborne,
                "is_parked": own.is_parked,
                "takeoff_pending": self.engagement_state.takeoff_pending(
                    own.command_id
                ),
                "return_pending": self.engagement_state.return_pending(
                    own.command_id
                ),
                "fuel_percentage": own.fuel_percentage,
                "fuel_low": fuel_low,
                "has_strike_weapon_system": (
                    own.has_strike_weapon_system
                ),
                "strike_weapon_count": own.strike_weapon_count,
                "ammunition_low": ammunition_low,
                "currently_attacking": currently_attacking,
                "current_attack_target_id": current_attack_target_id,
                "attack_conditions_valid": (
                    currently_attacking
                    and (
                        current_attack_target is not None
                        or target_loss_within_grace
                    )
                    and (
                        current_attack_target is None
                        or current_attack_target.is_contact
                        or current_attack_target.health_pct > 0.0
                    )
                    and own.commandable
                    and own.communication_ok
                    and not own.radar_jammed
                    and not fuel_low
                ),
                "attack_target_missing_frames": (
                    attack_target_missing_frames
                ),
                "attack_target_loss_grace_frames": (
                    self.engagement_state.target_contact_loss_grace_frames
                ),
            }
            incoming = self._find_incoming_missile(own, targets)
            patrol = self._patrol_context(own)
            if incoming is not None:
                evade_lon, evade_lat = self._destination_point(
                    own.longitude,
                    own.latitude,
                    (incoming.heading_deg + 90.0) % 360.0,
                    EVADE_ROUTE_LENGTH_KM,
                )
                facts_by_entity[own.command_id] = ReasoningFacts(
                    entity_id=own.command_id,
                    own_platform_type=own.domain,
                    detected_target_count=len(targets),
                    incoming_missile=True,
                    incoming_missile_id=incoming.command_id,
                    incoming_missile_distance_km=own.distance_to_km(incoming),
                    incoming_missile_heading_deg=incoming.heading_deg,
                    evade_lon=evade_lon,
                    evade_lat=evade_lat,
                    attack_authorized=own.commandable,
                    safety_clearance=(
                        own.communication_ok and not own.radar_jammed
                    ),
                    is_patrol_aircraft=patrol.is_patrol_aircraft,
                    has_patrol_mission=patrol.has_patrol_mission,
                    inside_patrol_area=patrol.inside_patrol_area,
                    mission_id=own.mission_id,
                    patrol_route_lons=patrol.route_lons,
                    patrol_route_lats=patrol.route_lats,
                    patrol_altitude_level=patrol.altitude_level,
                    waypoint_velocity_level=3,
                    radar_available=own.has_radar_sensor,
                    sonar_available=own.has_sonar_sensor,
                    altitude_above_sea_m=own.altitude_m,
                    sonobuoy_count=own.sonobuoy_count,
                    **lifecycle_facts
                )
                continue

            selected = self._select_target(
                own,
                targets,
                planned_attackers,
                planned_interceptors,
            )
            if selected is None:
                facts_by_entity[own.command_id] = ReasoningFacts(
                    entity_id=own.command_id,
                    own_platform_type=own.domain,
                    detected_target_count=len(targets),
                    attack_authorized=own.commandable,
                    safety_clearance=(
                        own.communication_ok and not own.radar_jammed
                    ),
                    is_patrol_aircraft=patrol.is_patrol_aircraft,
                    has_patrol_mission=patrol.has_patrol_mission,
                    inside_patrol_area=patrol.inside_patrol_area,
                    mission_id=own.mission_id,
                    patrol_route_lons=patrol.route_lons,
                    patrol_route_lats=patrol.route_lats,
                    patrol_altitude_level=patrol.altitude_level,
                    waypoint_velocity_level=3,
                    radar_available=own.has_radar_sensor,
                    sonar_available=own.has_sonar_sensor,
                    altitude_above_sea_m=own.altitude_m,
                    sonobuoy_count=own.sonobuoy_count,
                    **lifecycle_facts
                )
                continue

            target, evaluation = selected
            target_id = evaluation.target_id
            planned_for_target = planned_attackers.setdefault(target_id, set())
            attack_quantity = DEFAULT_ATTACK_QUANTITY
            if evaluation.target_is_missile:
                attack_quantity = min(
                    DEFAULT_ATTACK_QUANTITY,
                    max(1, 4 - evaluation.interceptors_launched),
                )
            preflight_eligible = (
                evaluation.attack_request_allowed
                and not currently_attacking
                and not (
                    own.is_aircraft
                    and own.is_airborne
                    and (fuel_low or ammunition_low)
                )
                and not (own.is_aircraft and own.is_parked)
            )
            fire_control_facts = self._fire_control_facts(
                own=own,
                target=target,
                attack_quantity=attack_quantity,
                eligible=preflight_eligible,
            )

            facts = ReasoningFacts(
                entity_id=own.command_id,
                own_platform_type=own.domain,
                target_id=target_id,
                target_entity_id=target.entity_id,
                target_domain=target.domain,
                target_is_missile=evaluation.target_is_missile,
                detected_target_count=len(targets),
                attack_authorized=evaluation.attack_authorized,
                target_type_allowed=evaluation.target_type_allowed,
                weapon_available=evaluation.weapon_available,
                compatible_weapon_count=evaluation.compatible_weapon_count,
                expected_weapon_type=self._expected_weapon_type(target.domain),
                within_attack_range=evaluation.within_attack_range,
                distance_km=evaluation.distance_km,
                max_attack_range_km=evaluation.max_attack_range_km,
                aimed_at_target=evaluation.aimed_at_target,
                heading_difference_deg=evaluation.heading_difference_deg,
                safety_clearance=evaluation.safety_clearance,
                chase_allowed=evaluation.chase_allowed,
                concurrency_slot_available=(
                    evaluation.concurrency_slot_available
                ),
                active_attackers_on_target=(
                    evaluation.active_attackers_on_target
                ),
                already_attacking_target=(
                    evaluation.already_attacking_target
                ),
                interceptors_launched=evaluation.interceptors_launched,
                attack_quantity=attack_quantity,
                target_evaluation=evaluation,
                **fire_control_facts,
                target_lon=target.longitude,
                target_lat=target.latitude,
                attack_altitude_level=self._attack_altitude_level(own, target),
                is_patrol_aircraft=patrol.is_patrol_aircraft,
                has_patrol_mission=patrol.has_patrol_mission,
                inside_patrol_area=patrol.inside_patrol_area,
                mission_id=own.mission_id,
                patrol_route_lons=patrol.route_lons,
                patrol_route_lats=patrol.route_lats,
                patrol_altitude_level=patrol.altitude_level,
                waypoint_velocity_level=3,
                radar_available=own.has_radar_sensor,
                sonar_available=own.has_sonar_sensor,
                altitude_above_sea_m=own.altitude_m,
                sonobuoy_count=own.sonobuoy_count,
                **lifecycle_facts
            )
            facts_by_entity[own.command_id] = facts

            # 统一使用 TargetEvaluation 预留本帧名额；执行失败不写入
            # EngagementState，等同于自动回滚。
            if evaluation.attack_request_allowed:
                planned_for_target.add(own.command_id)
                if evaluation.target_is_missile:
                    planned_interceptors[target_id] = (
                        planned_interceptors.get(target_id, 0)
                        + attack_quantity
                    )

        return facts_by_entity

    def _select_target(
        self,
        own: EncodedEntity,
        targets: Sequence[EncodedEntity],
        planned_attackers: Mapping[str, Set[str]],
        planned_interceptors: Mapping[str, int],
    ) -> Optional[Tuple[EncodedEntity, TargetEvaluation]]:
        """射程内合法目标优先；仅飞机可在没有射程内目标时选超距目标。"""

        priority_intercept: Optional[
            Tuple[EncodedEntity, TargetEvaluation]
        ] = None
        immediate: Optional[Tuple[EncodedEntity, TargetEvaluation]] = None
        pursuit: Optional[Tuple[EncodedEntity, TargetEvaluation]] = None
        diagnostic: Optional[Tuple[EncodedEntity, TargetEvaluation]] = None

        def nearer(
            current: Optional[Tuple[EncodedEntity, TargetEvaluation]],
            candidate: Tuple[EncodedEntity, TargetEvaluation],
        ) -> Tuple[EncodedEntity, TargetEvaluation]:
            if current is None:
                return candidate
            current_key = (
                current[1].distance_km,
                current[1].target_id,
            )
            candidate_key = (
                candidate[1].distance_km,
                candidate[1].target_id,
            )
            return candidate if candidate_key < current_key else current

        previous_geometry_cache = self._geometry_cache
        self._geometry_cache = self._batch_target_geometry(own, targets)
        try:
            for target in targets:
                if target.domain is TargetDomain.UNKNOWN:
                    continue
                evaluation = self._evaluate_target(
                    own,
                    target,
                    planned_attackers,
                    planned_interceptors,
                )
                evaluated = (target, evaluation)
                diagnostic = nearer(diagnostic, evaluated)
                selection_kind = evaluation.selection_kind
                if selection_kind == 2:
                    immediate = nearer(immediate, evaluated)
                    if (
                        evaluation.target_is_missile
                        and evaluation.distance_km
                        <= self.missile_intercept_distance_km
                    ):
                        priority_intercept = nearer(
                            priority_intercept, evaluated
                        )
                elif selection_kind == 1:
                    pursuit = nearer(pursuit, evaluated)
        finally:
            self._geometry_cache = previous_geometry_cache

        if priority_intercept is not None:
            # 5 km 只负责紧急规避；防空平台在更远距离就优先选择导弹，
            # 避免先攻击较近舰艇而把拦截拖到末段。
            return priority_intercept
        if immediate is not None:
            return immediate
        if pursuit is not None:
            return pursuit
        if diagnostic is not None:
            # 没有合法候选时保留最近目标，以便规则路径明确说明被射程、
            # 弹药、并发或拦截数量中的哪一项拒绝。
            return diagnostic
        return None

    def _evaluate_target(
        self,
        own: EncodedEntity,
        target: EncodedEntity,
        planned_attackers: Mapping[str, Set[str]],
        planned_interceptors: Mapping[str, int],
    ) -> TargetEvaluation:
        """一次性计算目标选择、预留和最终规则共同需要的约束。"""

        target_id = target.command_id
        geometry = self._geometry_cache.get(id(target))
        if geometry is None:
            geometry = self._distance_and_bearing(own, target)
        distance_km, target_bearing = geometry
        strike_range_km = own.strike_range_for(target)
        weapon_count = own.weapon_count_for(target)
        target_is_missile = (
            target.is_weapon and target.domain is TargetDomain.AIR
        )
        heading_difference = abs(
            (own.heading_deg - target_bearing + 180.0) % 360.0 - 180.0
        )
        aimed = (
            own.domain is TargetDomain.SURFACE
            or heading_difference < self.aim_tolerance_deg
        )
        planned_for_target = planned_attackers.get(target_id, ())
        (
            active_attackers,
            slot_available,
            already_attacking,
            committed_interceptors,
        ) = self.engagement_state.target_attack_status(
            own.command_id,
            target_id,
            planned_for_target,
        )
        interceptors_launched = (
            committed_interceptors
            + planned_interceptors.get(target_id, 0)
        )
        target_type_allowed = (
            target.domain is not TargetDomain.UNKNOWN
            and strike_range_km > 0.0
        )
        within_range = (
            target_type_allowed and distance_km <= strike_range_km
        )
        safety_clearance = own.communication_ok and not own.radar_jammed

        return TargetEvaluation(
            target_id=target_id,
            own_platform_type=own.domain,
            target_domain=target.domain,
            target_is_missile=target_is_missile,
            attack_authorized=own.commandable,
            safety_clearance=safety_clearance,
            target_type_allowed=target_type_allowed,
            weapon_available=weapon_count > 0,
            compatible_weapon_count=weapon_count,
            within_attack_range=within_range,
            distance_km=distance_km,
            max_attack_range_km=strike_range_km,
            aimed_at_target=aimed,
            heading_difference_deg=heading_difference,
            chase_allowed=(
                own.is_aircraft
                and own.commandable
                and safety_clearance
                and target_type_allowed
                and weapon_count > 0
            ),
            concurrency_slot_available=slot_available,
            active_attackers_on_target=(
                active_attackers + len(planned_for_target)
            ),
            already_attacking_target=already_attacking,
            interceptors_launched=interceptors_launched,
        )

    def _fire_control_facts(
        self,
        own: EncodedEntity,
        target: EncodedEntity,
        attack_quantity: int,
        eligible: bool,
    ) -> Dict[str, Any]:
        """把 RPC 可发射性与跨帧拒绝冷却转换为最终攻击事实。"""

        quality_signature = target.contact_quality_signature
        result: Dict[str, Any] = {
            "fire_control_checked": False,
            "fire_control_available": False,
            "fire_control_cooldown": False,
            "fire_control_reason": "",
            "target_quality_signature": quality_signature,
        }
        if self.weapon_fire_prechecker is None or not eligible:
            return result

        rejection = self.engagement_state.fire_control_rejection(
            attacker_id=own.command_id,
            target_entity_id=target.entity_id,
            contact_quality_signature=quality_signature,
            current_frame=self.current_step,
        )
        if rejection is not None:
            result.update(
                fire_control_checked=True,
                fire_control_available=False,
                fire_control_cooldown=True,
                fire_control_reason=rejection.reason,
            )
            return result

        try:
            precheck = self.weapon_fire_prechecker(
                attacker_id=own.command_id,
                target_id=target.command_id,
                quantity=attack_quantity,
                mode="manual",
            )
            if not isinstance(precheck, WeaponFirePrecheck):
                raise TypeError(
                    "weapon_fire_prechecker 必须返回 WeaponFirePrecheck"
                )
            if (
                precheck.attacker_id != own.command_id
                or precheck.target_id != target.command_id
            ):
                raise ValueError("武器预检结果与当前攻击方/目标不一致")
        except Exception as error:
            precheck = WeaponFirePrecheck(
                can_fire=False,
                attacker_id=own.command_id,
                target_id=target.command_id,
                reason="武器预检异常：{}".format(error),
                reason_key="PRECHECK_ERROR:{}".format(
                    type(error).__name__
                ),
            )

        result.update(
            fire_control_checked=True,
            fire_control_available=precheck.can_fire,
            fire_control_reason=precheck.reason,
        )
        if precheck.can_fire:
            self.engagement_state.clear_fire_control_rejection(
                own.command_id, target.entity_id
            )
        else:
            self.engagement_state.record_fire_control_rejection(
                attacker_id=own.command_id,
                target_entity_id=target.entity_id,
                reason_key=precheck.reason_key,
                reason=precheck.reason,
                contact_quality_signature=quality_signature,
                current_frame=self.current_step,
            )
        return result

    def _find_incoming_missile(
        self, own: EncodedEntity, targets: Sequence[EncodedEntity]
    ) -> Optional[EncodedEntity]:
        threats: List[Tuple[float, EncodedEntity]] = []
        for missile in targets:
            if not missile.is_weapon or missile.domain is not TargetDomain.AIR:
                continue
            distance = own.distance_to_km(missile)
            if distance > MISSILE_EVADE_DISTANCE_KM:
                continue

            if missile.weapon_target_id:
                points_at_own = missile.weapon_target_id in (
                    own.entity_id,
                    own.command_id,
                )
            else:
                # 没有唯一目标 ID（包括名称重复无法唯一解析）时，按已确认
                # 口径使用导弹航向与距离进行几何推断。
                bearing = self._bearing_deg(
                    missile.longitude,
                    missile.latitude,
                    own.longitude,
                    own.latitude,
                )
                difference = abs(
                    (missile.heading_deg - bearing + 180.0) % 360.0 - 180.0
                )
                points_at_own = difference < MISSILE_POINTING_TOLERANCE_DEG

            if points_at_own:
                threats.append((distance, missile))

        if not threats:
            return None
        return min(threats, key=lambda item: (item[0], item[1].command_id))[1]

    def _patrol_context(self, own: EncodedEntity) -> PatrolContext:
        info = self.mission_areas.get(own.mission_id)
        area_points: Sequence[Any] = ()
        mission_is_patrol = own.has_patrol_mission

        if isinstance(info, Mapping):
            area_points = info.get("area_points") or info.get("areaPoints") or ()
            text = " ".join(
                str(info.get(key) or "")
                for key in ("mission_type", "missionType", "name", "missionName")
            ).lower()
            mission_is_patrol = mission_is_patrol or bool(info.get("is_patrol"))
            mission_is_patrol = mission_is_patrol or "巡逻" in text or "patrol" in text
        elif isinstance(info, Sequence) and not isinstance(info, (str, bytes)):
            area_points = info

        is_patrol_aircraft = own.is_aircraft and mission_is_patrol
        inside = (
            is_patrol_aircraft
            and self._point_in_polygon_with_boundary(
                own.longitude, own.latitude, area_points
            )
        )
        route_lons: Tuple[float, ...] = ()
        route_lats: Tuple[float, ...] = ()
        if is_patrol_aircraft:
            route_lons, route_lats = self._build_patrol_route(
                own, area_points
            )
        if own.altitude_m <= 500.0:
            altitude_level = 1
        elif own.altitude_m <= 5000.0:
            altitude_level = 3
        else:
            altitude_level = 5
        return PatrolContext(
            is_patrol_aircraft=is_patrol_aircraft,
            has_patrol_mission=mission_is_patrol,
            inside_patrol_area=inside,
            route_lons=route_lons,
            route_lats=route_lats,
            altitude_level=altitude_level,
        )

    @staticmethod
    def _build_patrol_route(
        own: EncodedEntity, raw_points: Sequence[Any]
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """把任务边界向中心收缩，生成始终位于区域内部的确定性巡逻航路。"""

        points: List[Tuple[float, float]] = []
        for item in raw_points:
            if isinstance(item, Mapping):
                lon = item.get("lon", item.get("longitude"))
                lat = item.get("lat", item.get("latitude"))
            elif isinstance(item, Sequence) and not isinstance(
                item, (str, bytes)
            ):
                if len(item) < 2:
                    continue
                lon, lat = item[0], item[1]
            else:
                continue
            try:
                point = (float(lon), float(lat))
            except (TypeError, ValueError):
                continue
            if not points or point != points[-1]:
                points.append(point)
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(points) < 3:
            return (), ()

        center_lon = sum(point[0] for point in points) / len(points)
        center_lat = sum(point[1] for point in points) / len(points)
        inset = [
            (
                center_lon + (lon - center_lon) * 0.8,
                center_lat + (lat - center_lat) * 0.8,
            )
            for lon, lat in points
        ]
        start = min(
            range(len(inset)),
            key=lambda index: (
                (inset[index][0] - own.longitude) ** 2
                + (inset[index][1] - own.latitude) ** 2,
                index,
            ),
        )
        ordered = inset[start:] + inset[:start]
        return (
            tuple(point[0] for point in ordered),
            tuple(point[1] for point in ordered),
        )

    def _record_successful_attacks(
        self,
        facts: Mapping[str, ReasoningFacts],
        decisions: Mapping[str, Decision],
        execution_status: Mapping[str, str],
        current_frame: int,
    ) -> None:
        for entity_id, decision in decisions.items():
            if decision.conclusion is not Conclusion.REQUEST_ATTACK:
                continue
            if execution_status.get(entity_id) != "SUCCESS":
                continue
            entity_facts = facts[entity_id]
            if entity_facts.target_id is None:
                continue
            self.engagement_state.record_successful_attack(
                attacker_id=entity_id,
                target_id=entity_facts.target_id,
                started_frame=current_frame,
                target_is_missile=entity_facts.target_is_missile,
                target_aliases=(entity_facts.target_entity_id,),
                interceptor_count=entity_facts.attack_quantity,
            )

    def _record_successful_lifecycle_actions(
        self,
        decisions: Mapping[str, Decision],
        execution_status: Mapping[str, str],
        current_frame: int,
    ) -> None:
        for entity_id, decision in decisions.items():
            if execution_status.get(entity_id) != "SUCCESS":
                continue
            if decision.conclusion is Conclusion.TAKEOFF:
                self.engagement_state.record_takeoff_request(
                    entity_id, current_frame
                )
            elif decision.conclusion is Conclusion.RETURN_TO_BASE:
                self.engagement_state.record_return_request(
                    entity_id, current_frame
                )
            elif decision.conclusion is Conclusion.CANCEL_ATTACK:
                self.engagement_state.release_attacker(entity_id)

    @staticmethod
    def _enemy_ids(targets: Sequence[EncodedEntity]) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for target in targets:
            if target.command_id not in seen:
                seen.add(target.command_id)
                result.append(target.command_id)
        return result

    @staticmethod
    def _expected_weapon_type(domain: TargetDomain) -> Optional[str]:
        if domain is TargetDomain.AIR:
            return "AIR_DEFENCE_OR_AIR_TO_AIR_MISSILE"
        if domain is TargetDomain.SURFACE:
            return "ANTI_SHIP_MISSILE"
        if domain is TargetDomain.SUBMARINE:
            return "ANTI_SUBMARINE_WEAPON_OR_TORPEDO"
        if domain is TargetDomain.LAND:
            return "LAND_ATTACK_WEAPON"
        return None

    @staticmethod
    def _attack_altitude_level(
        own: EncodedEntity, target: EncodedEntity
    ) -> int:
        if own.domain is TargetDomain.SURFACE:
            return 0
        if own.domain is TargetDomain.SUBMARINE:
            if target.domain is TargetDomain.SUBMARINE:
                return 3
            return 1
        if own.domain is TargetDomain.AIR:
            if target.domain is TargetDomain.AIR:
                return 5 if target.altitude_m > 5000.0 else 3
            if target.domain in (TargetDomain.SURFACE, TargetDomain.SUBMARINE):
                return 1
        return 0

    def _heading_difference(
        self, own: EncodedEntity, target: EncodedEntity
    ) -> float:
        bearing = self._bearing_deg(
            own.longitude, own.latitude, target.longitude, target.latitude
        )
        return abs((own.heading_deg - bearing + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _distance_and_bearing(
        own: EncodedEntity, target: EncodedEntity
    ) -> Tuple[float, float]:
        """共享球面中间量，一次计算三维距离和目标方位。"""

        radius_km = 6371.0088
        lat1 = math.radians(own.latitude)
        lat2 = math.radians(target.latitude)
        delta_lat = lat2 - lat1
        delta_lon = math.radians(target.longitude - own.longitude)
        sin_half_lat = math.sin(delta_lat / 2.0)
        sin_half_lon = math.sin(delta_lon / 2.0)
        haversine = (
            sin_half_lat * sin_half_lat
            + math.cos(lat1)
            * math.cos(lat2)
            * sin_half_lon
            * sin_half_lon
        )
        ground_km = 2.0 * radius_km * math.asin(
            min(1.0, math.sqrt(max(0.0, haversine)))
        )
        altitude_km = (target.altitude_m - own.altitude_m) / 1000.0
        distance_km = math.hypot(ground_km, altitude_km)

        x = math.sin(delta_lon) * math.cos(lat2)
        y = (
            math.cos(lat1) * math.sin(lat2)
            - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        )
        bearing = math.degrees(math.atan2(x, y)) % 360.0
        return distance_km, bearing

    @staticmethod
    def _batch_target_geometry(
        own: EncodedEntity, targets: Sequence[EncodedEntity]
    ) -> Dict[int, Tuple[float, float]]:
        """用向量运算批量计算一个平台到全部目标的距离与方位。"""

        if not targets:
            return {}
        target_lats = np.fromiter(
            (target.latitude for target in targets),
            dtype=np.float64,
            count=len(targets),
        )
        target_lons = np.fromiter(
            (target.longitude for target in targets),
            dtype=np.float64,
            count=len(targets),
        )
        target_alts = np.fromiter(
            (target.altitude_m for target in targets),
            dtype=np.float64,
            count=len(targets),
        )
        lat1 = math.radians(own.latitude)
        lon1 = math.radians(own.longitude)
        lat2 = np.radians(target_lats)
        lon2 = np.radians(target_lons)
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        haversine = (
            np.sin(delta_lat / 2.0) ** 2
            + math.cos(lat1)
            * np.cos(lat2)
            * np.sin(delta_lon / 2.0) ** 2
        )
        ground_km = 2.0 * 6371.0088 * np.arcsin(
            np.sqrt(np.clip(haversine, 0.0, 1.0))
        )
        distances = np.hypot(
            ground_km, (target_alts - own.altitude_m) / 1000.0
        )
        x = np.sin(delta_lon) * np.cos(lat2)
        y = (
            math.cos(lat1) * np.sin(lat2)
            - math.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
        )
        bearings = np.degrees(np.arctan2(x, y)) % 360.0
        return {
            id(target): (float(distance), float(bearing))
            for target, distance, bearing in zip(
                targets, distances, bearings
            )
        }

    @staticmethod
    def _bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(
            math.radians, (lon1, lat1, lon2, lat2)
        )
        delta_lon = lon2_rad - lon1_rad
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = (
            math.cos(lat1_rad) * math.sin(lat2_rad)
            - math.sin(lat1_rad)
            * math.cos(lat2_rad)
            * math.cos(delta_lon)
        )
        return math.degrees(math.atan2(x, y)) % 360.0

    @staticmethod
    def _destination_point(
        lon: float, lat: float, heading_deg: float, distance_km: float
    ) -> Tuple[float, float]:
        radius_km = 6371.0088
        angular = distance_km / radius_km
        bearing = math.radians(heading_deg)
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular)
            + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(angular) * math.cos(lat1),
            math.cos(angular) - math.sin(lat1) * math.sin(lat2),
        )
        return (
            (math.degrees(lon2) + 540.0) % 360.0 - 180.0,
            math.degrees(lat2),
        )

    @staticmethod
    def _point_in_polygon_with_boundary(
        lon: float, lat: float, raw_points: Sequence[Any]
    ) -> bool:
        points: List[Tuple[float, float]] = []
        for item in raw_points:
            if isinstance(item, Mapping):
                point_lon = item.get("lon", item.get("longitude"))
                point_lat = item.get("lat", item.get("latitude"))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                if len(item) < 2:
                    continue
                point_lon, point_lat = item[0], item[1]
            else:
                continue
            try:
                points.append((float(point_lon), float(point_lat)))
            except (TypeError, ValueError):
                continue
        if len(points) < 3:
            return False

        epsilon = 1e-10
        inside = False
        previous = points[-1]
        for current in points:
            x1, y1 = previous
            x2, y2 = current
            cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
            on_segment = (
                abs(cross) <= epsilon
                and min(x1, x2) - epsilon <= lon <= max(x1, x2) + epsilon
                and min(y1, y2) - epsilon <= lat <= max(y1, y2) + epsilon
            )
            if on_segment:
                return True
            crosses = (y1 > lat) != (y2 > lat)
            if crosses:
                x_at_lat = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
                if lon < x_at_lat:
                    inside = not inside
            previous = current
        return inside

def log_step(result: SymbolicStepResult, step_index: int, logger: Any) -> None:
    """完整推理路径只写 DEBUG，避免正常未命中结果淹没操作日志。"""

    debug = getattr(logger, "debug", None)
    if debug is None:
        return
    debug(
        "step={} cooldown_frame={} time_compression={}x "
        "entities={} own={} targets={}".format(
            step_index,
            result.cooldown_frame,
            result.time_compression_multiplier,
            len(result.situation.entities),
            len(result.situation.own_entities),
            len(result.situation.targets),
        )
    )
    entities = {entity.command_id: entity for entity in result.situation.entities}
    for entity_id, decision in result.decisions.items():
        entity = entities.get(entity_id)
        name = entity.name if entity is not None else entity_id
        status = (result.execution_status or {}).get(entity_id, "UNKNOWN")
        debug(
            "  {} -> {} | execution={}\n{}".format(
                name,
                decision.conclusion.value,
                status,
                decision.explanation,
            )
        )


def _load_mission_areas(path: Optional[Path]) -> Mapping[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("任务区域文件顶层必须是 mission_id -> mission_info 对象")
    return value


def main_loop(
    env: SymbolicReasoningEnv,
    input_path: Path,
    steps: int,
    interval: float,
    execute_commands: bool,
    logger: Any,
    frontend_control: Optional[FrontendControl] = None,
    executor: Any = None,
    situation_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
) -> None:
    step_index = 0
    last_situation_error = ""
    while steps <= 0 or step_index < steps:
        if (
            frontend_control is not None
            and not frontend_control.wait_until_runnable()
        ):
            debug = getattr(logger, "debug", None)
            if debug is not None:
                debug("[UI控制] 前端停止推演，符号推理循环退出")
            break

        try:
            payload = (
                situation_provider()
                if situation_provider is not None
                else load_situation(input_path)
            )
            last_situation_error = ""
        except Exception as error:
            if frontend_control is None:
                raise
            # 实时服务重启/短暂断连时不让长期运行进程退出。UI 监听会保持暂停，
            # 服务恢复并再次返回 running 后从同一逻辑帧继续。
            error_key = "{}:{}".format(type(error).__name__, error)
            if error_key != last_situation_error:
                logger.warning(
                    "[实时态势] 读取失败，等待服务恢复: %s", error
                )
                last_situation_error = error_key
            else:
                debug = getattr(logger, "debug", None)
                if debug is not None:
                    debug("[实时态势] 重复读取失败: %s", error)
            time.sleep(max(0.1, interval))
            continue

        # 文件读取期间可能收到暂停；执行前再次门控，避免在暂停后下发新命令。
        if (
            frontend_control is not None
            and not frontend_control.wait_until_runnable()
        ):
            debug = getattr(logger, "debug", None)
            if debug is not None:
                debug("[UI控制] 前端停止推演，符号推理循环退出")
            break

        result = env.step(
            payload,
            execute_commands=execute_commands,
            logger=logger,
            executor=executor,
            time_compression_multiplier=(
                frontend_control.time_compression_multiplier
                if frontend_control is not None
                else 1
            ),
        )
        log_step(result, step_index, logger)
        step_index += 1
        if steps <= 0 or step_index < steps:
            time.sleep(max(0.0, interval))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="运行符号推理测试智能体")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("source/我方视角下态势完整响应.txt"),
        help="SendMsg 完整响应 JSON 文件",
    )
    parser.add_argument(
        "--mission-areas",
        type=Path,
        default=None,
        help="可选任务区域 JSON；键为 mission_id，值含 is_patrol 和 area_points",
    )
    parser.add_argument(
        "--ignore-rpc-missions",
        action="store_true",
        help="离线运行时不调用 getMissionList；默认从推演服务读取任务区域",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="循环次数；0 或负数表示持续运行",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="请求下一帧前的现实时间间隔（秒），不参与规则帧计时",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "从 --attack-rpc-target 的红方 GetThreeSituation 获取实时态势和 "
            "Contact ID，并用 GetUnitData 补充真实武器库存；默认读取 --input 文件"
        ),
    )
    parser.add_argument(
        "--rpc-timeout",
        type=float,
        default=20.0,
        help="实时态势、单位数据和 UI 状态 RPC 超时秒数；默认 20",
    )
    parser.add_argument(
        "--missile-intercept-distance-km",
        type=float,
        default=MISSILE_INTERCEPT_PRIORITY_DISTANCE_KM,
        help=(
            "导弹进入该距离后优先于其他合法目标拦截；默认 50 km。"
            "5 km 紧急规避阈值保持不变"
        ),
    )
    parser.add_argument(
        "--target-loss-grace-frames",
        type=int,
        default=TARGET_CONTACT_LOSS_GRACE_FRAMES,
        help="攻击目标连续丢失多少帧后才取消；默认 3 帧",
    )
    parser.add_argument(
        "--fire-rejection-cooldown-frames",
        type=int,
        default=FIRE_CONTROL_REJECTION_COOLDOWN_FRAMES,
        help="相同攻击方、目标、Contact质量和拒绝原因的重试冷却；默认 10 帧",
    )
    parser.add_argument(
        "--attack-rpc-target",
        default=DEFAULT_ATTACK_RPC_TARGET,
        help=(
            "GetWeaponFiringInfo/AttackTarget 服务地址；默认 {}"
        ).format(DEFAULT_ATTACK_RPC_TARGET),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只编码和推理，不下发命令；默认会实际执行",
    )
    parser.add_argument(
        "--ignore-ui-control",
        action="store_true",
        help="离线运行时忽略前端开始、暂停和停止信号；默认启用前端控制",
    )
    parser.add_argument(
        "--verbose-reasoning",
        action="store_true",
        help="显示逐实体完整推理路径和未命中依据；默认只显示成功操作和告警",
    )
    args = parser.parse_args(argv)

    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_level = logging.DEBUG if args.verbose_reasoning else logging.INFO
    logger = Log(
        name=f"symbolic_reasoning_{current_time}",
        log_dir="logs",
        level=log_level,
    )
    # Log 使用进程级单例；显式重设可以保证本入口的参数在复用时仍然生效。
    logger.logger.setLevel(log_level)
    live_source = None
    if args.live:
        live_source = RpcSituationSource(
            rpc_target=args.attack_rpc_target,
            timeout=args.rpc_timeout,
            logger=logger,
        )
        logger.info(
            "[实时态势] 已连接 rpc=%s，红方默认可控，武器库存来源=GetUnitData",
            args.attack_rpc_target,
        )
    mission_areas: Dict[str, Any] = {}
    if not args.ignore_rpc_missions:
        try:
            mission_areas.update(
                load_project_mission_areas(
                    fetcher=(
                        live_source.fetch_missions
                        if live_source is not None
                        else None
                    )
                )
            )
            logger.info(
                "[任务区域] getMissionList 加载任务数量={}".format(
                    len(mission_areas)
                )
            )
        except Exception as error:
            logger.warning(
                "[任务区域] getMissionList 读取失败，继续使用本地任务区域: %s",
                error,
            )
    # 本地文件可覆盖同 missionId 的 RPC 数据，便于离线复现和人工校正。
    mission_areas.update(_load_mission_areas(args.mission_areas))
    engagement_state = EngagementState(
        target_contact_loss_grace_frames=args.target_loss_grace_frames,
        fire_control_rejection_cooldown_frames=(
            args.fire_rejection_cooldown_frames
        ),
    )
    weapon_fire_prechecker = (
        functools.partial(
            precheck_weapon_fire,
            rpc_target=args.attack_rpc_target,
            timeout=args.rpc_timeout,
            logger=logger,
            stub=live_source.stub,
        )
        if live_source is not None
        else None
    )
    env = SymbolicReasoningEnv(
        mission_areas=mission_areas,
        missile_intercept_distance_km=args.missile_intercept_distance_km,
        engagement_state=engagement_state,
        weapon_fire_prechecker=weapon_fire_prechecker,
    )
    logger.info(
        "[规则参数] missile_intercept_priority_km=%s missile_evade_km=%s "
        "target_loss_grace_frames=%s fire_rejection_cooldown_frames=%s",
        args.missile_intercept_distance_km,
        MISSILE_EVADE_DISTANCE_KM,
        args.target_loss_grace_frames,
        args.fire_rejection_cooldown_frames,
    )
    action_executor = functools.partial(
        execute_actions,
        rpc_target=args.attack_rpc_target,
    )
    frontend_control = None
    if not args.ignore_ui_control:
        frontend_control = FrontendControl(
            signal_provider=(
                live_source.control_signal
                if live_source is not None
                else None
            ),
            logger=logger,
        )
        frontend_control.start()
        logger.info("[UI控制] 已启动监听，等待前端运行信号")
    try:
        main_loop(
            env=env,
            input_path=args.input,
            steps=args.steps,
            interval=args.interval,
            execute_commands=not args.dry_run,
            logger=logger,
            frontend_control=frontend_control,
            executor=action_executor,
            situation_provider=(
                live_source.fetch_payload
                if live_source is not None
                else None
            ),
        )
    except KeyboardInterrupt:
        logger.debug("符号推理循环已停止")
    finally:
        # 先关闭 gRPC channel，使可能阻塞在 GetEngineStatus 的后台监听调用
        # 立即取消，再等待监听线程退出，避免 Ctrl+C 后额外等待 RPC 超时。
        if live_source is not None:
            live_source.close()
        if frontend_control is not None:
            frontend_control.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
