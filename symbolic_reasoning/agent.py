"""符号推理智能体：按 rule.md 推理并生成 execute 兼容动作。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .entity import TargetDomain
from .state import MAX_INTERCEPTORS_PER_MISSILE


# 与根目录 execute.execute_actions 一致：8 个 Actor，每个 Actor 5 个参数。
ACTOR_COUNT = 8
ACTION_THRESHOLD = 0.9
ACTION_DISABLED = 0.01

RETURN_TO_BASE_ACTOR = 1
WAYPOINT_ACTOR = 2
MOBILITY_ACTOR = 3
ATTACK_ACTOR = 4
SENSOR_ACTOR = 5
SONOBUOY_ACTOR = 6


class Conclusion(str, Enum):
    """符号推理可以输出的主结论。"""

    EVADE_MISSILE = "EVADE_MISSILE"
    REQUEST_ATTACK = "REQUEST_ATTACK"
    # 兼容旧调用；新代码和解释统一使用 REQUEST_ATTACK。
    ATTACK = "REQUEST_ATTACK"
    CHASE_TO_RANGE = "CHASE_TO_RANGE"
    # 兼容旧调用；新代码区分追击入射程和转向对准。
    CHASE = "CHASE_TO_RANGE"
    CHASE_AND_ALIGN = "CHASE_AND_ALIGN"
    DEPLOY_SONOBUOY = "DEPLOY_SONOBUOY"
    SEARCH = "SEARCH"
    HOLD = "HOLD"
    RETURN_TO_BASE = "RETURN_TO_BASE"


@dataclass(frozen=True)
class TargetEvaluation:
    """对单个候选目标的统一攻击约束评估。

    目标选择、同帧攻击名额预留和最终规则推理共同使用本对象，避免三处
    分别维护射程、弹药、并发、拦截数量和朝向判断。
    """

    target_id: str
    own_platform_type: TargetDomain
    target_domain: TargetDomain
    target_is_missile: bool
    attack_authorized: bool
    safety_clearance: bool
    target_type_allowed: bool
    weapon_available: bool
    compatible_weapon_count: int
    within_attack_range: bool
    distance_km: float
    max_attack_range_km: float
    aimed_at_target: bool
    heading_difference_deg: float
    chase_allowed: bool
    concurrency_slot_available: bool
    active_attackers_on_target: int
    already_attacking_target: bool
    interceptors_launched: int

    @classmethod
    def from_facts(cls, facts: "ReasoningFacts") -> "TargetEvaluation":
        if facts.target_id is None:
            raise ValueError("没有目标时不能构造 TargetEvaluation")
        return cls(
            target_id=facts.target_id,
            own_platform_type=facts.own_platform_type,
            target_domain=facts.target_domain,
            target_is_missile=facts.target_is_missile,
            attack_authorized=facts.attack_authorized,
            safety_clearance=facts.safety_clearance,
            target_type_allowed=facts.target_type_allowed,
            weapon_available=facts.weapon_available,
            compatible_weapon_count=facts.compatible_weapon_count,
            within_attack_range=facts.within_attack_range,
            distance_km=facts.distance_km,
            max_attack_range_km=facts.max_attack_range_km,
            aimed_at_target=facts.aimed_at_target,
            heading_difference_deg=facts.heading_difference_deg,
            chase_allowed=facts.chase_allowed,
            concurrency_slot_available=facts.concurrency_slot_available,
            active_attackers_on_target=facts.active_attackers_on_target,
            already_attacking_target=facts.already_attacking_target,
            interceptors_launched=facts.interceptors_launched,
        )

    @property
    def authorization_blocked(self) -> bool:
        return not self.attack_authorized or not self.safety_clearance

    @property
    def concurrency_blocked(self) -> bool:
        return (
            not self.concurrency_slot_available
            or self.already_attacking_target
        )

    @property
    def interceptor_blocked(self) -> bool:
        return (
            self.target_is_missile
            and self.interceptors_launched >= MAX_INTERCEPTORS_PER_MISSILE
        )

    @property
    def range_capability_blocked(self) -> bool:
        return (
            not self.target_type_allowed
            or self.max_attack_range_km <= 0.0
        )

    @property
    def weapon_blocked(self) -> bool:
        return (
            not self.weapon_available
            or self.compatible_weapon_count <= 0
        )

    @property
    def aim_required(self) -> bool:
        return self.own_platform_type is not TargetDomain.SURFACE

    @property
    def aim_ok(self) -> bool:
        return not self.aim_required or self.aimed_at_target

    @property
    def candidate_eligible(self) -> bool:
        """是否具备进入立即攻击/追击候选集的目标级条件。"""

        return not (
            self.range_capability_blocked
            or self.weapon_blocked
            or self.concurrency_blocked
            or self.interceptor_blocked
        )

    @property
    def immediate_candidate(self) -> bool:
        return self.candidate_eligible and self.within_attack_range

    @property
    def pursuit_candidate(self) -> bool:
        return (
            self.candidate_eligible
            and not self.within_attack_range
            and self.own_platform_type is TargetDomain.AIR
        )

    @property
    def can_chase(self) -> bool:
        return (
            self.own_platform_type is TargetDomain.AIR
            and self.chase_allowed
        )

    @property
    def attack_request_allowed(self) -> bool:
        """所有 REQUEST_ATTACK 条件是否统一通过。"""

        return (
            not self.authorization_blocked
            and not self.concurrency_blocked
            and not self.interceptor_blocked
            and not self.range_capability_blocked
            and not self.weapon_blocked
            and self.within_attack_range
            and self.aim_ok
        )


@dataclass(frozen=True)
class ReasoningFacts:
    """单个实体的标准化输入事实。

    危险动作相关字段默认全部关闭，调用方漏传字段时会安全拒绝。
    """

    entity_id: str
    own_platform_type: TargetDomain = TargetDomain.UNKNOWN
    target_id: Optional[str] = None
    target_entity_id: Optional[str] = None
    target_domain: TargetDomain = TargetDomain.UNKNOWN
    target_is_missile: bool = False
    detected_target_count: int = 0

    incoming_missile: bool = False
    incoming_missile_id: Optional[str] = None
    incoming_missile_distance_km: float = -1.0
    incoming_missile_heading_deg: float = 0.0
    evade_lon: float = 0.0
    evade_lat: float = 0.0

    attack_authorized: bool = False
    target_type_allowed: bool = False
    weapon_available: bool = False
    compatible_weapon_count: int = 0
    expected_weapon_type: Optional[str] = None
    within_attack_range: bool = False
    distance_km: float = -1.0
    max_attack_range_km: float = 0.0
    aimed_at_target: bool = False
    heading_difference_deg: float = 180.0
    safety_clearance: bool = False
    chase_allowed: bool = False
    concurrency_slot_available: bool = False
    active_attackers_on_target: int = 0
    already_attacking_target: bool = False
    interceptors_launched: int = 0
    target_evaluation: Optional[TargetEvaluation] = None

    target_lon: float = 0.0
    target_lat: float = 0.0
    attack_altitude_level: int = 0
    waypoint_velocity_level: int = 4

    is_patrol_aircraft: bool = False
    has_patrol_mission: bool = False
    inside_patrol_area: bool = False
    altitude_above_sea_m: float = -1.0
    sonobuoy_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("entity_id 必须是非空字符串")
        if self.target_id is not None and not isinstance(self.target_id, str):
            raise ValueError("target_id 必须是字符串或 None")
        if self.target_entity_id is not None and not isinstance(
            self.target_entity_id, str
        ):
            raise ValueError("target_entity_id 必须是字符串或 None")
        if not isinstance(self.own_platform_type, TargetDomain):
            raise ValueError("own_platform_type 必须是 TargetDomain")
        if not isinstance(self.target_domain, TargetDomain):
            raise ValueError("target_domain 必须是 TargetDomain")
        if self.target_evaluation is not None:
            if not isinstance(self.target_evaluation, TargetEvaluation):
                raise ValueError("target_evaluation 必须是 TargetEvaluation 或 None")
            if self.target_evaluation.target_id != self.target_id:
                raise ValueError("target_evaluation 与 target_id 不一致")
            if self.target_evaluation != TargetEvaluation.from_facts(self):
                raise ValueError("target_evaluation 与标准化事实不一致")

        bool_fields = (
            "target_is_missile",
            "incoming_missile",
            "attack_authorized",
            "target_type_allowed",
            "weapon_available",
            "within_attack_range",
            "aimed_at_target",
            "safety_clearance",
            "chase_allowed",
            "concurrency_slot_available",
            "already_attacking_target",
            "is_patrol_aircraft",
            "has_patrol_mission",
            "inside_patrol_area",
        )
        for field_name in bool_fields:
            if type(getattr(self, field_name)) is not bool:
                raise ValueError("{} 必须是 bool".format(field_name))

        count_fields = (
            "detected_target_count",
            "compatible_weapon_count",
            "active_attackers_on_target",
            "interceptors_launched",
            "sonobuoy_count",
        )
        for field_name in count_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("{} 必须是非负整数".format(field_name))

        numeric_fields = (
            "incoming_missile_distance_km",
            "incoming_missile_heading_deg",
            "evade_lon",
            "evade_lat",
            "distance_km",
            "max_attack_range_km",
            "heading_difference_deg",
            "target_lon",
            "target_lat",
            "altitude_above_sea_m",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("{} 必须是有限数值".format(field_name))
            if not math.isfinite(float(value)):
                raise ValueError("{} 必须是有限数值".format(field_name))

        if not 0 <= self.attack_altitude_level <= 5:
            raise ValueError("attack_altitude_level 必须位于 [0, 5]")
        if not 0 <= self.waypoint_velocity_level <= 4:
            raise ValueError("waypoint_velocity_level 必须位于 [0, 4]")


@dataclass(frozen=True)
class InferenceStep:
    """一条规则的匹配结果和使用的事实。"""

    rule_id: str
    rule: str
    matched: bool
    evidence: Tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    """一次推理的结论、依据和 execute 动作。"""

    conclusion: Conclusion
    rule_id: str
    reason: str
    matched_facts: Tuple[str, ...]
    inference_path: Tuple[InferenceStep, ...]
    actions: List[List[Any]]
    target_id: Optional[str] = None
    expected_weapon_type: Optional[str] = None

    @property
    def explanation(self) -> str:
        lines = ["推理路径："]
        for index, step in enumerate(self.inference_path, 1):
            status = "命中" if step.matched else "未命中"
            lines.append(
                "{}. {} [{}] {}；事实：{}".format(
                    index,
                    step.rule_id,
                    status,
                    step.rule,
                    "，".join(step.evidence),
                )
            )
        lines.append(
            "结论：{}；决定规则：{}；依据：{}".format(
                self.conclusion.value,
                self.rule_id,
                "，".join(self.matched_facts),
            )
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class RunResult:
    """推理与 execute_actions 的合并返回结果。"""

    decisions: Dict[str, Decision]
    actions_dict: Dict[str, List[List[Any]]]
    execute_results: Any
    rewards: Any
    execution_status: Dict[str, str]


Executor = Callable[
    [Dict[str, List[List[Any]]], Sequence[str], float, Any], Tuple[Any, Any]
]


class SymbolicReasoningAgent:
    """按 rule.md 的优先级匹配规则并生成确定性动作。"""

    def reason(self, facts: ReasoningFacts) -> Decision:
        path: List[InferenceStep] = []

        def record(
            rule_id: str,
            rule: str,
            matched: bool,
            evidence: Tuple[str, ...],
        ) -> bool:
            path.append(
                InferenceStep(
                    rule_id=rule_id,
                    rule=rule,
                    matched=matched,
                    evidence=evidence,
                )
            )
            return matched

        # P1：来袭导弹规避覆盖攻击、追击和浮标部署。
        if record(
            "R-MSL-001",
            "5 km 内确认或推断为直接指向本实体的敌方导弹属于来袭威胁",
            facts.incoming_missile,
            (
                "incoming_missile={}".format(facts.incoming_missile),
                "missile_id={}".format(facts.incoming_missile_id),
                "distance_km={:.3f}".format(facts.incoming_missile_distance_km),
                "threshold_km=5",
            ),
        ):
            record(
                "R-MSL-002",
                "固定向导弹航向右侧 90 度规避，并提高速度和高度",
                True,
                (
                    "missile_heading_deg={:.3f}".format(
                        facts.incoming_missile_heading_deg
                    ),
                    "evade_heading_deg={:.3f}".format(
                        (facts.incoming_missile_heading_deg + 90.0) % 360.0
                    ),
                    "speed_level=4",
                    "altitude_level=5",
                ),
            )
            return self._decision(
                Conclusion.EVADE_MISSILE,
                "R-MSL-002",
                "近距离来袭导弹威胁，紧急规避优先",
                (
                    "incoming_missile=True",
                    "distance_km<=5",
                    "right_turn=90deg",
                ),
                facts,
                tuple(path),
            )

        has_target = facts.target_id is not None
        record(
            "R-TGT-001",
            "从合法候选目标中选择最近目标",
            has_target,
            (
                "target_id={}".format(facts.target_id),
                "detected_target_count={}".format(facts.detected_target_count),
                "distance_km={:.3f}".format(facts.distance_km),
            ),
        )

        if has_target:
            evaluation = (
                facts.target_evaluation
                or TargetEvaluation.from_facts(facts)
            )
            if record(
                "R-VAL-001",
                "实体必须可操纵且通信安全状态允许攻击",
                evaluation.authorization_blocked,
                (
                    "attack_authorized={}".format(facts.attack_authorized),
                    "safety_clearance={}".format(facts.safety_clearance),
                ),
            ):
                return self._hold(
                    "R-VAL-001",
                    "实体不可操纵或通信安全状态不允许攻击",
                    facts,
                    tuple(path),
                )

            if record(
                "R-CON-001",
                "同一目标最多允许 3 个不同攻击者占用或预留并发槽位",
                evaluation.concurrency_blocked,
                (
                    "active_attackers={}".format(
                        facts.active_attackers_on_target
                    ),
                    "limit=3",
                    "slot_available={}".format(
                        facts.concurrency_slot_available
                    ),
                    "already_attacking={}".format(
                        facts.already_attacking_target
                    ),
                ),
            ):
                reason = (
                    "本实体已有针对该目标的在途攻击"
                    if facts.already_attacking_target
                    else "同目标并发攻击槽位已满"
                )
                return self._hold("R-CON-001", reason, facts, tuple(path))

            if record(
                "R-INT-001",
                "一个导弹目标生命周期内累计最多发射 4 发拦截弹",
                evaluation.interceptor_blocked,
                (
                    "target_is_missile={}".format(facts.target_is_missile),
                    "interceptors_launched={}".format(
                        facts.interceptors_launched
                    ),
                    "limit={}".format(MAX_INTERCEPTORS_PER_MISSILE),
                ),
            ):
                return self._hold(
                    "R-INT-001",
                    "该导弹目标的累计拦截弹数量已达 4 发",
                    facts,
                    tuple(path),
                )

            if record(
                "R-RNG-001",
                "平台必须具备目标域对应的有效最大射程",
                evaluation.range_capability_blocked,
                (
                    "target_domain={}".format(facts.target_domain.name),
                    "target_type_allowed={}".format(
                        facts.target_type_allowed
                    ),
                    "max_attack_range_km={:.3f}".format(
                        facts.max_attack_range_km
                    ),
                ),
            ):
                return self._hold(
                    "R-RNG-001",
                    "平台没有该目标域的有效攻击能力",
                    facts,
                    tuple(path),
                )

            if record(
                self._weapon_rule_id(facts.target_domain),
                "对应类型武器数量必须大于 0，具体武器由系统选择",
                evaluation.weapon_blocked,
                (
                    "expected_weapon_type={}".format(
                        facts.expected_weapon_type
                    ),
                    "compatible_weapon_count={}".format(
                        facts.compatible_weapon_count
                    ),
                    "weapon_selection=SYSTEM",
                ),
            ):
                return self._hold(
                    self._weapon_rule_id(facts.target_domain),
                    "对应目标域的武器数量不足",
                    facts,
                    tuple(path),
                )

            if not evaluation.within_attack_range:
                can_chase = evaluation.can_chase
                record(
                    "R-RNG-004",
                    "超出射程时只有飞机允许追击，且不得立即发射",
                    can_chase,
                    (
                        "distance_km={:.3f}".format(facts.distance_km),
                        "max_attack_range_km={:.3f}".format(
                            facts.max_attack_range_km
                        ),
                        "own_platform={}".format(
                            facts.own_platform_type.name
                        ),
                        "chase_allowed={}".format(facts.chase_allowed),
                    ),
                )
                if can_chase:
                    return self._decision(
                        Conclusion.CHASE_TO_RANGE,
                        "R-RNG-004",
                        "飞机位于射程外，先追击目标，进入射程后重新推理",
                        (
                            "distance_km>max_attack_range_km",
                            "own_platform=AIRCRAFT",
                        ),
                        facts,
                        tuple(path),
                    )
                record(
                    "R-RNG-003",
                    "非飞机平台超出射程时禁止攻击且不允许追击",
                    True,
                    (
                        "distance_km>max_attack_range_km",
                        "own_platform={}".format(
                            facts.own_platform_type.name
                        ),
                    ),
                )
                return self._hold(
                    "R-RNG-003",
                    "目标超出射程，非飞机平台不允许追击",
                    facts,
                    tuple(path),
                )

            aim_required = evaluation.aim_required
            aim_ok = evaluation.aim_ok
            record(
                "R-AIM-001" if not aim_required else "R-AIM-002",
                "水面舰艇免除朝向限制；飞机和潜艇必须严格小于 30 度",
                aim_ok,
                (
                    "own_platform={}".format(facts.own_platform_type.name),
                    "heading_difference_deg={:.3f}".format(
                        facts.heading_difference_deg
                    ),
                    "limit_deg=<30",
                ),
            )
            if not aim_ok:
                if (
                    facts.own_platform_type is TargetDomain.AIR
                    and facts.chase_allowed
                ):
                    return self._decision(
                        Conclusion.CHASE_AND_ALIGN,
                        "R-AIM-002",
                        "飞机在射程内但未对准，先转向目标再重新推理",
                        ("heading_difference_deg>=30", "own_platform=AIRCRAFT"),
                        facts,
                        tuple(path),
                    )
                return self._hold(
                    "R-AIM-002",
                    "平台未满足严格小于 30 度的攻击朝向条件",
                    facts,
                    tuple(path),
                )

            if not evaluation.attack_request_allowed:
                raise RuntimeError("TargetEvaluation 与规则推理路径不一致")

            weapon_rule = self._weapon_rule_id(facts.target_domain)
            record(
                weapon_rule,
                "射程、朝向、并发和弹药条件均满足，向系统提交发射请求",
                True,
                (
                    "distance_km={:.3f}".format(facts.distance_km),
                    "max_attack_range_km={:.3f}".format(
                        facts.max_attack_range_km
                    ),
                    "heading_difference_deg={:.3f}".format(
                        facts.heading_difference_deg
                    ),
                    "compatible_weapon_count={}".format(
                        facts.compatible_weapon_count
                    ),
                ),
            )
            return self._decision(
                Conclusion.REQUEST_ATTACK,
                weapon_rule,
                "所有攻击约束通过，具体武器由系统自动选择",
                (
                    "within_attack_range=True",
                    "aimed_at_target=True_or_surface_ship_exempt",
                    "concurrency_slot_available=True",
                    "weapon_available=True",
                ),
                facts,
                tuple(path),
            )

        buoy_allowed = (
            facts.is_patrol_aircraft
            and facts.has_patrol_mission
            and facts.inside_patrol_area
            and 0.0 <= facts.altitude_above_sea_m <= 500.0
            and facts.sonobuoy_count > 0
        )
        if record(
            "R-BUOY-001",
            "巡逻机在含边界的巡逻区内且距海面 0 至 500 m 时允许部署浮标",
            buoy_allowed,
            (
                "is_patrol_aircraft={}".format(facts.is_patrol_aircraft),
                "has_patrol_mission={}".format(facts.has_patrol_mission),
                "inside_patrol_area={}".format(facts.inside_patrol_area),
                "altitude_above_sea_m={:.3f}".format(
                    facts.altitude_above_sea_m
                ),
                "sonobuoy_count={}".format(facts.sonobuoy_count),
            ),
        ):
            return self._decision(
                Conclusion.DEPLOY_SONOBUOY,
                "R-BUOY-001",
                "巡逻任务、区域、高度和浮标数量条件均满足",
                (
                    "patrol_aircraft=True",
                    "inside_patrol_area=True",
                    "0<=altitude_above_sea_m<=500",
                    "sonobuoy_count>0",
                ),
                facts,
                tuple(path),
            )

        if facts.detected_target_count > 0:
            record(
                "R-TGT-001",
                "存在敌方目标但没有满足全部约束的合法候选目标",
                True,
                ("detected_target_count={}".format(facts.detected_target_count),),
            )
            return self._hold(
                "R-TGT-001",
                "没有可立即攻击或可追击的合法目标",
                facts,
                tuple(path),
            )

        record(
            "R-SEARCH-001",
            "没有敌方目标且不满足浮标部署条件时开启传感器搜索",
            True,
            ("detected_target_count=0",),
        )
        return self._decision(
            Conclusion.SEARCH,
            "R-SEARCH-001",
            "没有发现敌方目标，开启传感器搜索",
            ("detected_target_count=0",),
            facts,
            tuple(path),
        )

    def build_actions(
        self, facts_list: Iterable[ReasoningFacts]
    ) -> Tuple[Dict[str, Decision], Dict[str, List[List[Any]]]]:
        decisions: Dict[str, Decision] = {}
        actions_dict: Dict[str, List[List[Any]]] = {}
        for facts in facts_list:
            if facts.entity_id in decisions:
                raise ValueError("重复 entity_id：{}".format(facts.entity_id))
            decision = self.reason(facts)
            decisions[facts.entity_id] = decision
            actions_dict[facts.entity_id] = decision.actions
        return decisions, actions_dict

    def run(
        self,
        facts_list: Iterable[ReasoningFacts],
        enemy_ids: Sequence[str],
        logger: Any = None,
        probability: float = 0.7,
        executor: Optional[Executor] = None,
    ) -> RunResult:
        """完成“推理 → execute_actions”的完整过程，默认实际执行。"""

        decisions, actions_dict = self.build_actions(facts_list)
        selected_executor = executor or self._load_execute_actions()
        execute_results, rewards = selected_executor(
            actions_dict, enemy_ids, probability, logger
        )
        status = self._execution_status(decisions, execute_results)
        return RunResult(
            decisions=decisions,
            actions_dict=actions_dict,
            execute_results=execute_results,
            rewards=rewards,
            execution_status=status,
        )

    @staticmethod
    def _execution_status(
        decisions: Dict[str, Decision], execute_results: Any
    ) -> Dict[str, str]:
        status: Dict[str, str] = {}
        result_map = execute_results if isinstance(execute_results, dict) else {}
        for entity_id, decision in decisions.items():
            actor_index = SymbolicReasoningAgent._main_actor_index(
                decision.conclusion
            )
            if actor_index is None:
                status[entity_id] = "NOT_REQUESTED"
                continue
            entity_result = result_map.get(entity_id)
            if not isinstance(entity_result, Sequence) or isinstance(
                entity_result, (str, bytes)
            ):
                status[entity_id] = "UNKNOWN"
            elif actor_index >= len(entity_result):
                status[entity_id] = "UNKNOWN"
            else:
                status[entity_id] = (
                    "SUCCESS" if bool(entity_result[actor_index]) else "FAILED"
                )
        return status

    @staticmethod
    def _main_actor_index(conclusion: Conclusion) -> Optional[int]:
        if conclusion is Conclusion.EVADE_MISSILE:
            return WAYPOINT_ACTOR
        if conclusion is Conclusion.REQUEST_ATTACK:
            return ATTACK_ACTOR
        if conclusion in (Conclusion.CHASE_TO_RANGE, Conclusion.CHASE_AND_ALIGN):
            return WAYPOINT_ACTOR
        if conclusion is Conclusion.DEPLOY_SONOBUOY:
            return SONOBUOY_ACTOR
        if conclusion is Conclusion.SEARCH:
            return SENSOR_ACTOR
        if conclusion is Conclusion.RETURN_TO_BASE:
            return RETURN_TO_BASE_ACTOR
        return None

    @staticmethod
    def _load_execute_actions() -> Executor:
        from .execute_actions import execute_actions

        return execute_actions

    @staticmethod
    def _empty_actions() -> List[List[Any]]:
        return [
            [ACTION_DISABLED, None, None, None, None]
            for _ in range(ACTOR_COUNT)
        ]

    @staticmethod
    def _weapon_rule_id(target_domain: TargetDomain) -> str:
        if target_domain is TargetDomain.AIR:
            return "R-WPN-001"
        if target_domain is TargetDomain.SURFACE:
            return "R-WPN-002"
        if target_domain is TargetDomain.SUBMARINE:
            return "R-WPN-003"
        return "R-WPN-000"

    def _hold(
        self,
        rule_id: str,
        reason: str,
        facts: ReasoningFacts,
        inference_path: Tuple[InferenceStep, ...],
    ) -> Decision:
        return self._decision(
            Conclusion.HOLD,
            rule_id,
            reason,
            (reason,),
            facts,
            inference_path,
        )

    def _decision(
        self,
        conclusion: Conclusion,
        rule_id: str,
        reason: str,
        matched_facts: Tuple[str, ...],
        facts: ReasoningFacts,
        inference_path: Tuple[InferenceStep, ...],
    ) -> Decision:
        actions = self._empty_actions()

        if conclusion is Conclusion.EVADE_MISSILE:
            actions[WAYPOINT_ACTOR] = [
                ACTION_THRESHOLD,
                facts.evade_lon,
                facts.evade_lat,
                5,
                4,
            ]
        elif conclusion is Conclusion.REQUEST_ATTACK:
            # 支持动作：先给出攻击高度层，再提交是否发射；武器由系统选择。
            actions[MOBILITY_ACTOR] = [
                ACTION_THRESHOLD,
                4,
                facts.attack_altitude_level,
                0.0,
                0.0,
            ]
            actions[ATTACK_ACTOR] = [
                ACTION_THRESHOLD,
                facts.target_id,
                facts.target_lon,
                facts.target_lat,
                None,
            ]
        elif conclusion in (
            Conclusion.CHASE_TO_RANGE,
            Conclusion.CHASE_AND_ALIGN,
        ):
            actions[WAYPOINT_ACTOR] = [
                ACTION_THRESHOLD,
                facts.target_lon,
                facts.target_lat,
                facts.attack_altitude_level,
                facts.waypoint_velocity_level,
            ]
        elif conclusion is Conclusion.DEPLOY_SONOBUOY:
            actions[SONOBUOY_ACTOR] = [
                ACTION_THRESHOLD,
                1.0,
                1.0,
                None,
                None,
            ]
        elif conclusion is Conclusion.SEARCH:
            actions[SENSOR_ACTOR] = [ACTION_THRESHOLD, 1.0, 1.0, 0.0, None]
        elif conclusion is Conclusion.RETURN_TO_BASE:
            actions[RETURN_TO_BASE_ACTOR] = [
                ACTION_THRESHOLD,
                None,
                None,
                None,
                None,
            ]

        return Decision(
            conclusion=conclusion,
            rule_id=rule_id,
            reason=reason,
            matched_facts=matched_facts,
            inference_path=inference_path,
            actions=actions,
            target_id=facts.target_id,
            expected_weapon_type=facts.expected_weapon_type,
        )
