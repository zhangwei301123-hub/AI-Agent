"""符号推理智能体：输入事实，推出结论，并用 execute_actions 执行。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# 与 execute.execute_actions 保持一致：每个实体有 8 个 Actor，每个 Actor 5 个参数。
ACTOR_COUNT = 8
ACTION_THRESHOLD = 0.9
ACTION_DISABLED = 0.01

RETURN_TO_BASE_ACTOR = 1
WAYPOINT_ACTOR = 2
ATTACK_ACTOR = 4
SENSOR_ACTOR = 5


class Conclusion(str, Enum):
    """符号推理可以得到的五种结论。"""

    RETURN_TO_BASE = "RETURN_TO_BASE"
    ATTACK = "ATTACK"
    CHASE = "CHASE"
    SEARCH = "SEARCH"
    HOLD = "HOLD"


@dataclass(frozen=True)
class ReasoningFacts:
    """单个实体的一组输入事实。

    安全相关事实默认均为 ``False``，因此调用方漏传事实时不会错误放行攻击。
    """

    entity_id: str
    target_id: Optional[str] = None
    need_return_to_base: bool = False
    attack_authorized: bool = False
    target_type_allowed: bool = False
    weapon_available: bool = False
    within_attack_range: bool = False
    aimed_at_target: bool = False
    safety_clearance: bool = False
    chase_allowed: bool = False
    target_lon: float = 0.0
    target_lat: float = 0.0
    waypoint_altitude: float = 4.0
    waypoint_velocity: float = 4.0

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("entity_id 必须是非空字符串")
        for field_name in (
            "need_return_to_base",
            "attack_authorized",
            "target_type_allowed",
            "weapon_available",
            "within_attack_range",
            "aimed_at_target",
            "safety_clearance",
            "chase_allowed",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError("{} 必须是 bool".format(field_name))
        for field_name in (
            "target_lon",
            "target_lat",
            "waypoint_altitude",
            "waypoint_velocity",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("{} 必须是有限数值".format(field_name))


@dataclass(frozen=True)
class InferenceStep:
    """一条规则的匹配结果和当时使用的事实。"""

    rule_id: str
    rule: str
    matched: bool
    evidence: Tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    """一次符号推理的结论、依据和 execute 动作。"""

    conclusion: Conclusion
    rule_id: str
    reason: str
    matched_facts: Tuple[str, ...]
    inference_path: Tuple[InferenceStep, ...]
    actions: List[List[Any]]

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


Executor = Callable[[Dict[str, List[List[Any]]], Sequence[str], float, Any], Tuple[Any, Any]]


class SymbolicReasoningAgent:
    """按固定优先级执行规则并调用现有 ``execute_actions``。

    规则优先级：返航 > 搜索目标 > 安全拒绝 > 攻击 > 追击 > 保持。
    """

    def reason(self, facts: ReasoningFacts) -> Decision:
        """根据一组事实得到一个确定性结论。"""

        path: List[InferenceStep] = []

        def matches(
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

        if matches(
            "SR-001",
            "需要返航时优先返航",
            facts.need_return_to_base,
            ("need_return_to_base={}".format(facts.need_return_to_base),),
        ):
            return self._decision(
                Conclusion.RETURN_TO_BASE,
                "SR-001",
                "需要返航，返航优先于其他动作",
                ("need_return_to_base=True",),
                facts,
                tuple(path),
            )

        if matches(
            "SR-002",
            "没有目标时开启搜索",
            facts.target_id is None,
            ("target_id={}".format(facts.target_id),),
        ):
            return self._decision(
                Conclusion.SEARCH,
                "SR-002",
                "没有发现目标，开启雷达搜索",
                ("target_id=None",),
                facts,
                tuple(path),
            )

        if matches(
            "SR-003",
            "无攻击权限时保持",
            not facts.attack_authorized,
            ("attack_authorized={}".format(facts.attack_authorized),),
        ):
            return self._hold(
                "SR-003",
                "实体没有攻击权限",
                "attack_authorized=False",
                facts,
                tuple(path),
            )

        if matches(
            "SR-004",
            "安全约束不允许时保持",
            not facts.safety_clearance,
            ("safety_clearance={}".format(facts.safety_clearance),),
        ):
            return self._hold(
                "SR-004",
                "安全约束不允许攻击",
                "safety_clearance=False",
                facts,
                tuple(path),
            )

        if matches(
            "SR-005",
            "目标类型不允许时保持",
            not facts.target_type_allowed,
            ("target_type_allowed={}".format(facts.target_type_allowed),),
        ):
            return self._hold(
                "SR-005",
                "目标类型不允许攻击",
                "target_type_allowed=False",
                facts,
                tuple(path),
            )

        if matches(
            "SR-006",
            "没有可用武器时保持",
            not facts.weapon_available,
            ("weapon_available={}".format(facts.weapon_available),),
        ):
            return self._hold(
                "SR-006",
                "没有可用武器",
                "weapon_available=False",
                facts,
                tuple(path),
            )

        attack_matched = facts.within_attack_range and facts.aimed_at_target
        if matches(
            "SR-007",
            "在攻击范围内且已对准时攻击",
            attack_matched,
            (
                "within_attack_range={}".format(facts.within_attack_range),
                "aimed_at_target={}".format(facts.aimed_at_target),
            ),
        ):
            return self._decision(
                Conclusion.ATTACK,
                "SR-007",
                "攻击条件全部满足",
                (
                    "attack_authorized=True",
                    "safety_clearance=True",
                    "target_type_allowed=True",
                    "weapon_available=True",
                    "within_attack_range=True",
                    "aimed_at_target=True",
                ),
                facts,
                tuple(path),
            )

        if matches(
            "SR-008",
            "不能立即攻击但允许追击时机动",
            facts.chase_allowed,
            ("chase_allowed={}".format(facts.chase_allowed),),
        ):
            return self._decision(
                Conclusion.CHASE,
                "SR-008",
                "暂不满足攻击条件，允许向目标机动",
                (
                    "within_attack_range={}".format(facts.within_attack_range),
                    "aimed_at_target={}".format(facts.aimed_at_target),
                    "chase_allowed=True",
                ),
                facts,
                tuple(path),
            )

        matches(
            "SR-009",
            "不能攻击且不允许追击时保持",
            True,
            ("chase_allowed=False",),
        )
        return self._hold(
            "SR-009",
            "不能攻击且不允许追击",
            "chase_allowed=False",
            facts,
            tuple(path),
        )

    def build_actions(
        self, facts_list: Iterable[ReasoningFacts]
    ) -> Tuple[Dict[str, Decision], Dict[str, List[List[Any]]]]:
        """对多个实体推理，生成可直接传给 ``execute_actions`` 的字典。"""

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
        """完成“符号推理 → execute_actions 执行”的完整过程。

        ``executor`` 为空时使用包内 ``execute_actions.py``，由它校验动作后复用
        项目现有的 ``execute.execute_actions``；测试时可注入假执行器。
        """

        decisions, actions_dict = self.build_actions(facts_list)
        selected_executor = executor or self._load_execute_actions()
        execute_results, rewards = selected_executor(
            actions_dict, enemy_ids, probability, logger
        )
        return RunResult(
            decisions=decisions,
            actions_dict=actions_dict,
            execute_results=execute_results,
            rewards=rewards,
        )

    @staticmethod
    def _load_execute_actions() -> Executor:
        # 延迟导入，真正执行时再由包内执行层加载根目录 gRPC 接口。
        from .execute_actions import execute_actions

        return execute_actions

    @staticmethod
    def _empty_actions() -> List[List[Any]]:
        return [
            [ACTION_DISABLED, None, None, None, None]
            for _ in range(ACTOR_COUNT)
        ]

    def _hold(
        self,
        rule_id: str,
        reason: str,
        matched_fact: str,
        facts: ReasoningFacts,
        inference_path: Tuple[InferenceStep, ...],
    ) -> Decision:
        return self._decision(
            Conclusion.HOLD,
            rule_id,
            reason,
            (matched_fact,),
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

        if conclusion is Conclusion.RETURN_TO_BASE:
            actions[RETURN_TO_BASE_ACTOR] = [
                ACTION_THRESHOLD,
                None,
                None,
                None,
                None,
            ]
        elif conclusion is Conclusion.ATTACK:
            actions[ATTACK_ACTOR] = [
                ACTION_THRESHOLD,
                facts.target_id,
                facts.target_lon,
                facts.target_lat,
                None,
            ]
        elif conclusion is Conclusion.CHASE:
            actions[WAYPOINT_ACTOR] = [
                ACTION_THRESHOLD,
                facts.target_lon,
                facts.target_lat,
                facts.waypoint_altitude,
                facts.waypoint_velocity,
            ]
        elif conclusion is Conclusion.SEARCH:
            actions[SENSOR_ACTOR] = [ACTION_THRESHOLD, 1.0, 0.0, 0.0, None]

        return Decision(
            conclusion=conclusion,
            rule_id=rule_id,
            reason=reason,
            matched_facts=matched_facts,
            inference_path=inference_path,
            actions=actions,
        )
