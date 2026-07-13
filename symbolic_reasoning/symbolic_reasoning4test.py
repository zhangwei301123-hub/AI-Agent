"""类似 maddpg4test.py 的符号推理运行入口，默认推理并实际执行。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .agent import Conclusion, Decision, ReasoningFacts, SymbolicReasoningAgent
from .entity import (
    EncodedEntity,
    EncodedSituation,
    EntityEncoder,
    TargetDomain,
    load_situation,
)
from .state import EngagementState


MISSILE_EVADE_DISTANCE_KM = 5.0
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


class SymbolicReasoningEnv:
    """完成“编码 → 校验 → 规则匹配 → 推理 → 默认实际执行”。"""

    def __init__(
        self,
        max_entities: int = 700,
        aim_tolerance_deg: float = AIM_TOLERANCE_DEG,
        mission_areas: Optional[Mapping[str, Any]] = None,
        engagement_state: Optional[EngagementState] = None,
    ) -> None:
        if not 0.0 < aim_tolerance_deg <= 180.0:
            raise ValueError("aim_tolerance_deg 必须位于 (0, 180]")
        self.encoder = EntityEncoder(max_entities=max_entities)
        self.agent = SymbolicReasoningAgent()
        self.aim_tolerance_deg = float(aim_tolerance_deg)
        self.mission_areas = dict(mission_areas or {})
        self.engagement_state = engagement_state or EngagementState()
        self.current_step = 0

    def step(
        self,
        payload: Mapping[str, Any],
        execute_commands: bool = True,
        logger: Any = None,
        executor: Any = None,
    ) -> SymbolicStepResult:
        situation = self.encoder.encode(payload)
        current_time = self._scenario_timestamp(situation.current_time)
        self.engagement_state.update_from_situation(situation, current_time)
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
                facts, decisions, execution_status, current_time
            )
        else:
            decisions, actions_dict = self.agent.build_actions(facts.values())
            execute_results = None
            rewards = None
            execution_status = {
                entity_id: "DRY_RUN" for entity_id in decisions
            }

        self.current_step += 1
        return SymbolicStepResult(
            situation=situation,
            facts=facts,
            decisions=decisions,
            actions_dict=actions_dict,
            execute_results=execute_results,
            rewards=rewards,
            execution_status=execution_status,
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
            incoming = self._find_incoming_missile(own, targets)
            patrol_facts = self._patrol_facts(own)
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
                    is_patrol_aircraft=patrol_facts[0],
                    has_patrol_mission=patrol_facts[1],
                    inside_patrol_area=patrol_facts[2],
                    altitude_above_sea_m=own.altitude_m,
                    sonobuoy_count=own.sonobuoy_count,
                )
                continue

            target = self._select_target(
                own,
                targets,
                planned_attackers,
                planned_interceptors,
            )
            if target is None:
                facts_by_entity[own.command_id] = ReasoningFacts(
                    entity_id=own.command_id,
                    own_platform_type=own.domain,
                    detected_target_count=len(targets),
                    attack_authorized=own.commandable,
                    safety_clearance=(
                        own.communication_ok and not own.radar_jammed
                    ),
                    is_patrol_aircraft=patrol_facts[0],
                    has_patrol_mission=patrol_facts[1],
                    inside_patrol_area=patrol_facts[2],
                    altitude_above_sea_m=own.altitude_m,
                    sonobuoy_count=own.sonobuoy_count,
                )
                continue

            target_id = target.command_id
            distance_km = own.distance_to_km(target)
            strike_range_km = own.strike_range_for(target)
            weapon_count = own.weapon_count_for(target)
            target_is_missile = (
                target.is_weapon and target.domain is TargetDomain.AIR
            )
            heading_difference = self._heading_difference(own, target)
            aimed = (
                True
                if own.domain is TargetDomain.SURFACE
                else heading_difference < self.aim_tolerance_deg
            )
            active_attackers = self.engagement_state.active_attackers(target_id)
            planned_for_target = planned_attackers.setdefault(target_id, set())
            slot_available = self.engagement_state.slot_available(
                own.command_id, target_id, planned_for_target
            )
            already_attacking = self.engagement_state.is_attacking(
                own.command_id, target_id
            )
            interceptors_launched = self.engagement_state.interceptors_launched(
                target_id
            )
            target_type_allowed = (
                target.domain is not TargetDomain.UNKNOWN
                and strike_range_km > 0.0
            )
            within_range = (
                target_type_allowed and distance_km <= strike_range_km
            )
            safety_clearance = own.communication_ok and not own.radar_jammed

            facts = ReasoningFacts(
                entity_id=own.command_id,
                own_platform_type=own.domain,
                target_id=target_id,
                target_entity_id=target.entity_id,
                target_domain=target.domain,
                target_is_missile=target_is_missile,
                detected_target_count=len(targets),
                attack_authorized=own.commandable,
                target_type_allowed=target_type_allowed,
                weapon_available=weapon_count > 0,
                compatible_weapon_count=weapon_count,
                expected_weapon_type=self._expected_weapon_type(target.domain),
                within_attack_range=within_range,
                distance_km=distance_km,
                max_attack_range_km=strike_range_km,
                aimed_at_target=aimed,
                heading_difference_deg=heading_difference,
                safety_clearance=safety_clearance,
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
                interceptors_launched=interceptors_launched
                + planned_interceptors.get(target_id, 0),
                target_lon=target.longitude,
                target_lat=target.latitude,
                attack_altitude_level=self._attack_altitude_level(own, target),
                is_patrol_aircraft=patrol_facts[0],
                has_patrol_mission=patrol_facts[1],
                inside_patrol_area=patrol_facts[2],
                altitude_above_sea_m=own.altitude_m,
                sonobuoy_count=own.sonobuoy_count,
            )
            facts_by_entity[own.command_id] = facts

            # 与 agent 的 REQUEST_ATTACK 条件保持一致。预留只在本帧防并发；
            # 执行失败不写入 EngagementState，等同于自动回滚。
            will_request_attack = (
                facts.attack_authorized
                and facts.safety_clearance
                and facts.target_type_allowed
                and facts.weapon_available
                and facts.within_attack_range
                and facts.concurrency_slot_available
                and not facts.already_attacking_target
                and (
                    own.domain is TargetDomain.SURFACE
                    or facts.aimed_at_target
                )
                and (
                    not target_is_missile
                    or facts.interceptors_launched < 4
                )
            )
            if will_request_attack:
                planned_for_target.add(own.command_id)
                if target_is_missile:
                    planned_interceptors[target_id] = (
                        planned_interceptors.get(target_id, 0) + 1
                    )

        return facts_by_entity

    def _select_target(
        self,
        own: EncodedEntity,
        targets: Sequence[EncodedEntity],
        planned_attackers: Mapping[str, Set[str]],
        planned_interceptors: Mapping[str, int],
    ) -> Optional[EncodedEntity]:
        """射程内合法目标优先；仅飞机可在没有射程内目标时选超距目标。"""

        immediate: List[EncodedEntity] = []
        pursuit: List[EncodedEntity] = []
        diagnostic: List[EncodedEntity] = []
        for target in targets:
            if target.domain is TargetDomain.UNKNOWN:
                continue
            diagnostic.append(target)
            strike_range = own.strike_range_for(target)
            weapon_count = own.weapon_count_for(target)
            if strike_range <= 0.0 or weapon_count <= 0:
                continue
            target_id = target.command_id
            planned = planned_attackers.get(target_id, set())
            if not self.engagement_state.slot_available(
                own.command_id, target_id, planned
            ):
                continue
            if target.is_weapon and target.domain is TargetDomain.AIR:
                if not self.engagement_state.interceptor_available(
                    target_id, planned_interceptors.get(target_id, 0)
                ):
                    continue
            distance = own.distance_to_km(target)
            if distance <= strike_range:
                immediate.append(target)
            elif own.is_aircraft:
                pursuit.append(target)

        key = lambda item: (own.distance_to_km(item), item.command_id)
        if immediate:
            return min(immediate, key=key)
        if pursuit:
            return min(pursuit, key=key)
        if diagnostic:
            # 没有合法候选时保留最近目标，以便规则路径明确说明被射程、
            # 弹药、并发或拦截数量中的哪一项拒绝。
            return min(diagnostic, key=key)
        return None

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

    def _patrol_facts(self, own: EncodedEntity) -> Tuple[bool, bool, bool]:
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
        return is_patrol_aircraft, mission_is_patrol, inside

    def _record_successful_attacks(
        self,
        facts: Mapping[str, ReasoningFacts],
        decisions: Mapping[str, Decision],
        execution_status: Mapping[str, str],
        current_time: float,
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
                started_at=current_time,
                target_is_missile=entity_facts.target_is_missile,
                target_aliases=(entity_facts.target_entity_id,),
            )

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

    @staticmethod
    def _scenario_timestamp(value: str) -> float:
        if value:
            text = value.strip()
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                return parsed.timestamp()
            except (TypeError, ValueError):
                pass
        return time.time()


def print_step(result: SymbolicStepResult, step_index: int) -> None:
    print(
        "step={} entities={} own={} targets={}".format(
            step_index,
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
        print(
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
) -> None:
    step_index = 0
    while steps <= 0 or step_index < steps:
        payload = load_situation(input_path)
        result = env.step(
            payload,
            execute_commands=execute_commands,
            logger=logger,
        )
        print_step(result, step_index)
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
        "--steps",
        type=int,
        default=1,
        help="循环次数；0 或负数表示持续运行",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="循环间隔秒数")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只编码和推理，不下发命令；默认会实际执行",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("symbolic_reasoning")
    env = SymbolicReasoningEnv(
        mission_areas=_load_mission_areas(args.mission_areas)
    )
    try:
        main_loop(
            env=env,
            input_path=args.input,
            steps=args.steps,
            interval=args.interval,
            execute_commands=not args.dry_run,
            logger=logger,
        )
    except KeyboardInterrupt:
        logger.info("符号推理循环已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
