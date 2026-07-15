"""自动执行符号推理正确性、覆盖性、可解释性和性能测试。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import tracemalloc
from time import perf_counter_ns
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .agent import (
    Conclusion,
    Decision,
    ReasoningFacts,
    SymbolicReasoningAgent,
    TargetEvaluation,
)
from .entity import EncodedEntity, TargetDomain
from .symbolic_reasoning4test import SymbolicReasoningEnv


BOOLEAN_FACT_NAMES = (
    "has_target",
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
    "target_is_missile",
    "interceptor_limit_reached",
    "is_aircraft",
    "buoy_conditions_met",
)

WORST_CASE_MAX_ENTITIES = 700
WORST_CASE_OWN_ENTITIES = WORST_CASE_MAX_ENTITIES // 2
WORST_CASE_TARGET_ENTITIES = (
    WORST_CASE_MAX_ENTITIES - WORST_CASE_OWN_ENTITIES
)
WORST_CASE_TARGET_EVALUATIONS = (
    WORST_CASE_OWN_ENTITIES * WORST_CASE_TARGET_ENTITIES
)


def _facts(**changes: Any) -> ReasoningFacts:
    values: Dict[str, Any] = {
        "entity_id": "our-aircraft-0001",
        "own_platform_type": TargetDomain.AIR,
        "target_id": "enemy-aircraft-0001",
        "target_domain": TargetDomain.AIR,
        "detected_target_count": 1,
        "attack_authorized": True,
        "target_type_allowed": True,
        "weapon_available": True,
        "compatible_weapon_count": 2,
        "expected_weapon_type": "AIR_DEFENCE_OR_AIR_TO_AIR_MISSILE",
        "within_attack_range": True,
        "distance_km": 50.0,
        "max_attack_range_km": 100.0,
        "aimed_at_target": True,
        "heading_difference_deg": 10.0,
        "safety_clearance": True,
        "chase_allowed": True,
        "concurrency_slot_available": True,
        "target_lon": 120.1,
        "target_lat": 30.2,
        "attack_altitude_level": 3,
    }
    values.update(changes)
    return ReasoningFacts(**values)


def correctness_cases() -> Tuple[Tuple[str, ReasoningFacts, Conclusion, str], ...]:
    """覆盖 rule.md 的关键允许、拒绝和边界路径。"""

    return (
        (
            "COR-MSL-001",
            _facts(
                incoming_missile=True,
                incoming_missile_id="enemy-missile-1",
                incoming_missile_distance_km=5.0,
                incoming_missile_heading_deg=90.0,
                evade_lon=120.0,
                evade_lat=29.95,
            ),
            Conclusion.EVADE_MISSILE,
            "R-MSL-002",
        ),
        (
            "COR-AIR-ATTACK",
            _facts(),
            Conclusion.REQUEST_ATTACK,
            "R-WPN-001",
        ),
        (
            "COR-RANGE-EQUAL",
            _facts(distance_km=100.0, max_attack_range_km=100.0),
            Conclusion.REQUEST_ATTACK,
            "R-WPN-001",
        ),
        (
            "COR-SURFACE-WEAPON",
            _facts(
                target_id="enemy-ship-1",
                target_domain=TargetDomain.SURFACE,
                expected_weapon_type="ANTI_SHIP_MISSILE",
                attack_altitude_level=1,
            ),
            Conclusion.REQUEST_ATTACK,
            "R-WPN-002",
        ),
        (
            "COR-SUBMARINE-WEAPON",
            _facts(
                target_id="enemy-submarine-1",
                target_domain=TargetDomain.SUBMARINE,
                expected_weapon_type="ANTI_SUBMARINE_WEAPON_OR_TORPEDO",
                attack_altitude_level=1,
            ),
            Conclusion.REQUEST_ATTACK,
            "R-WPN-003",
        ),
        (
            "COR-AIR-CHASE-RANGE",
            _facts(
                within_attack_range=False,
                distance_km=120.0,
                max_attack_range_km=100.0,
            ),
            Conclusion.CHASE_TO_RANGE,
            "R-RNG-004",
        ),
        (
            "COR-SHIP-NO-CHASE",
            _facts(
                own_platform_type=TargetDomain.SURFACE,
                within_attack_range=False,
                distance_km=120.0,
                max_attack_range_km=100.0,
            ),
            Conclusion.HOLD,
            "R-RNG-003",
        ),
        (
            "COR-AIR-ALIGN",
            _facts(aimed_at_target=False, heading_difference_deg=30.0),
            Conclusion.CHASE_AND_ALIGN,
            "R-AIM-002",
        ),
        (
            "COR-SUBMARINE-AIM",
            _facts(
                own_platform_type=TargetDomain.SUBMARINE,
                aimed_at_target=False,
                heading_difference_deg=30.0,
            ),
            Conclusion.HOLD,
            "R-AIM-002",
        ),
        (
            "COR-SHIP-AIM-EXEMPT",
            _facts(
                own_platform_type=TargetDomain.SURFACE,
                aimed_at_target=False,
                heading_difference_deg=180.0,
            ),
            Conclusion.REQUEST_ATTACK,
            "R-WPN-001",
        ),
        (
            "COR-CONCURRENCY",
            _facts(
                concurrency_slot_available=False,
                active_attackers_on_target=3,
            ),
            Conclusion.HOLD,
            "R-CON-001",
        ),
        (
            "COR-ACTIVE-ATTACK",
            _facts(already_attacking_target=True),
            Conclusion.HOLD,
            "R-CON-001",
        ),
        (
            "COR-INTERCEPTOR-LIMIT",
            _facts(target_is_missile=True, interceptors_launched=4),
            Conclusion.HOLD,
            "R-INT-001",
        ),
        (
            "COR-NO-RANGE",
            _facts(target_type_allowed=False, max_attack_range_km=0.0),
            Conclusion.HOLD,
            "R-RNG-001",
        ),
        (
            "COR-NO-WEAPON",
            _facts(weapon_available=False, compatible_weapon_count=0),
            Conclusion.HOLD,
            "R-WPN-001",
        ),
        (
            "COR-SAFETY",
            _facts(safety_clearance=False),
            Conclusion.HOLD,
            "R-VAL-001",
        ),
        (
            "COR-BUOY-0M",
            _facts(
                target_id=None,
                detected_target_count=0,
                is_patrol_aircraft=True,
                has_patrol_mission=True,
                inside_patrol_area=True,
                altitude_above_sea_m=0.0,
                sonobuoy_count=1,
            ),
            Conclusion.DEPLOY_SONOBUOY,
            "R-BUOY-001",
        ),
        (
            "COR-BUOY-500M",
            _facts(
                target_id=None,
                detected_target_count=0,
                is_patrol_aircraft=True,
                has_patrol_mission=True,
                inside_patrol_area=True,
                altitude_above_sea_m=500.0,
                sonobuoy_count=1,
            ),
            Conclusion.DEPLOY_SONOBUOY,
            "R-BUOY-001",
        ),
        (
            "COR-BUOY-TOO-HIGH",
            _facts(
                target_id=None,
                detected_target_count=0,
                is_patrol_aircraft=True,
                has_patrol_mission=True,
                inside_patrol_area=True,
                altitude_above_sea_m=500.01,
                sonobuoy_count=1,
            ),
            Conclusion.SEARCH,
            "R-SEARCH-001",
        ),
        (
            "COR-NO-LEGAL-TARGET",
            _facts(target_id=None, detected_target_count=2),
            Conclusion.HOLD,
            "R-TGT-001",
        ),
        (
            "COR-SEARCH",
            _facts(target_id=None, detected_target_count=0),
            Conclusion.SEARCH,
            "R-SEARCH-001",
        ),
    )


RULE_IDS = {case[3] for case in correctness_cases()}


def _valid_decision(decision: Decision) -> bool:
    return (
        isinstance(decision.conclusion, Conclusion)
        and isinstance(decision.rule_id, str)
        and bool(decision.rule_id)
        and len(decision.actions) == 8
        and all(len(action) == 5 for action in decision.actions)
    )


def _valid_explanation_text(decision: Decision, explanation: str) -> bool:
    if not decision.inference_path:
        return False
    if not any(step.rule_id == decision.rule_id for step in decision.inference_path):
        return False
    return (
        isinstance(explanation, str)
        and "推理路径" in explanation
        and "结论" in explanation
        and decision.rule_id in explanation
        and all(step.rule_id in explanation for step in decision.inference_path)
        and all(step.evidence for step in decision.inference_path)
    )


def _valid_explanation(decision: Decision) -> bool:
    return _valid_explanation_text(decision, decision.explanation)


def _complete_explanation_text(decision: Decision, explanation: str) -> bool:
    return _valid_explanation_text(decision, explanation) and all(
        evidence in explanation
        for step in decision.inference_path
        for evidence in step.evidence
    )


def test_correctness(agent: SymbolicReasoningAgent) -> Dict[str, Any]:
    failures = []
    cases = correctness_cases()
    for case_id, facts, expected_conclusion, expected_rule in cases:
        decision = agent.reason(facts)
        if (
            decision.conclusion is not expected_conclusion
            or decision.rule_id != expected_rule
            or not _valid_decision(decision)
        ):
            failures.append(
                {
                    "case_id": case_id,
                    "expected_conclusion": expected_conclusion.value,
                    "actual_conclusion": decision.conclusion.value,
                    "expected_rule": expected_rule,
                    "actual_rule": decision.rule_id,
                }
            )
    return {
        "passed": not failures,
        "total": len(cases),
        "passed_cases": len(cases) - len(failures),
        "failures": failures,
    }


def _all_fact_combinations() -> Iterable[ReasoningFacts]:
    """穷举 15 个标准化任务布尔事实，共 2^15=32768 种组合。"""

    for values in itertools.product((False, True), repeat=len(BOOLEAN_FACT_NAMES)):
        flags = dict(zip(BOOLEAN_FACT_NAMES, values))
        has_target = flags["has_target"]
        weapon_available = flags["weapon_available"]
        is_aircraft = flags["is_aircraft"]
        buoy = flags["buoy_conditions_met"]
        yield _facts(
            own_platform_type=(
                TargetDomain.AIR if is_aircraft else TargetDomain.SURFACE
            ),
            target_id="enemy-missile-1" if has_target else None,
            detected_target_count=1 if has_target else 0,
            incoming_missile=flags["incoming_missile"],
            incoming_missile_id=(
                "enemy-missile-1" if flags["incoming_missile"] else None
            ),
            incoming_missile_distance_km=(
                4.0 if flags["incoming_missile"] else -1.0
            ),
            attack_authorized=flags["attack_authorized"],
            target_type_allowed=flags["target_type_allowed"],
            weapon_available=weapon_available,
            compatible_weapon_count=1 if weapon_available else 0,
            within_attack_range=flags["within_attack_range"],
            aimed_at_target=flags["aimed_at_target"],
            safety_clearance=flags["safety_clearance"],
            chase_allowed=flags["chase_allowed"],
            concurrency_slot_available=flags["concurrency_slot_available"],
            already_attacking_target=flags["already_attacking_target"],
            target_is_missile=flags["target_is_missile"],
            interceptors_launched=(
                4 if flags["interceptor_limit_reached"] else 0
            ),
            is_patrol_aircraft=buoy,
            has_patrol_mission=buoy,
            inside_patrol_area=buoy,
            altitude_above_sea_m=500.0 if buoy else 501.0,
            sonobuoy_count=1 if buoy else 0,
        )


def test_coverage(agent: SymbolicReasoningAgent) -> Dict[str, Any]:
    total = 0
    invalid = 0
    fired_rules = set()
    conclusion_counts = {item.value: 0 for item in Conclusion}
    for facts in _all_fact_combinations():
        total += 1
        try:
            decision = agent.reason(facts)
        except Exception:
            invalid += 1
            continue
        if not _valid_decision(decision):
            invalid += 1
            continue
        fired_rules.add(decision.rule_id)
        conclusion_counts[decision.conclusion.value] += 1

    possible = 2 ** len(BOOLEAN_FACT_NAMES)
    correctness_fired_rules = {
        agent.reason(case_facts).rule_id
        for _, case_facts, _, _ in correctness_cases()
    }
    all_rules_covered = RULE_IDS.issubset(
        fired_rules | correctness_fired_rules
    )
    return {
        "passed": total == possible and invalid == 0 and all_rules_covered,
        "fact_axes": len(BOOLEAN_FACT_NAMES),
        "possible_combinations": possible,
        "tested_combinations": total,
        "valid_conclusions": total - invalid,
        "invalid_conclusions": invalid,
        "fired_rules": sorted(fired_rules),
        "correctness_rule_count": len(RULE_IDS),
        "all_decisive_rules_covered": all_rules_covered,
        "conclusion_counts": conclusion_counts,
    }


def test_explainability(agent: SymbolicReasoningAgent) -> Dict[str, Any]:
    total = 0
    explainable = 0
    failures = []
    for index, facts in enumerate(_all_fact_combinations(), 1):
        total += 1
        decision = agent.reason(facts)
        if _valid_explanation(decision):
            explainable += 1
        elif len(failures) < 20:
            failures.append(index)
    return {
        "passed": explainable == total,
        "total": total,
        "explainable": explainable,
        "failed_combination_indexes": failures,
    }


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _benchmark_operation(
    operation: Callable[[], Any],
    iterations: int,
    memory_iterations: Optional[int] = None,
) -> Tuple[Dict[str, Any], Any]:
    """计时并测量一次操作；返回指标和最后一次操作结果。"""

    if iterations <= 0:
        raise ValueError("iterations 必须大于 0")
    if memory_iterations is None:
        memory_iterations = iterations
    if memory_iterations <= 0:
        raise ValueError("memory_iterations 必须大于 0")

    for _ in range(min(10, iterations)):
        operation()

    durations_ms: List[float] = []
    last_result: Any = None
    for _ in range(iterations):
        start_ns = perf_counter_ns()
        last_result = operation()
        durations_ms.append((perf_counter_ns() - start_ns) / 1_000_000.0)

    tracemalloc.start()
    try:
        for _ in range(memory_iterations):
            last_result = operation()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return (
        {
            "iterations": iterations,
            "mean_ms": round(statistics.mean(durations_ms), 6),
            "p95_ms": round(_percentile(durations_ms, 0.95), 6),
            "max_ms": round(max(durations_ms), 6),
            "peak_memory_mib": round(
                peak_bytes / (1024.0 * 1024.0), 6
            ),
        },
        last_result,
    )


def _worst_case_payload() -> Dict[str, Any]:
    """构造 700 实体的最大笛卡尔积目标遍历场景。"""

    own_side_id = "acceptance-own-side"
    units: List[Dict[str, Any]] = []
    for index in range(WORST_CASE_OWN_ENTITIES):
        units.append(
            {
                "guid": "own-aircraft-{:03d}".format(index),
                "name": "性能测试我方飞机{:03d}".format(index),
                "SideId": own_side_id,
                "IsContact": False,
                "IsWeapon": False,
                "unitType": 0,
                "unitCategory": 0,
                "longitude": 120.0 + (index % 25) * 0.0001,
                "latitude": 30.0 + (index // 25) * 0.0001,
                "altitude": 8000.0,
                "heading": 90.0,
                "BloodAmount": 100.0,
                "rangeStrike_Air": 500.0,
                "weaponNumber": {"airNum": WORST_CASE_TARGET_ENTITIES},
                "Icon2D": "/Aircraft/acceptance-own.png",
            }
        )

    for index in range(WORST_CASE_TARGET_ENTITIES):
        units.append(
            {
                "guid": "enemy-aircraft-{:03d}".format(index),
                "contactGuid": "enemy-contact-{:03d}".format(index),
                "name": "性能测试敌方飞机{:03d}".format(index),
                "SideId": "acceptance-enemy-side",
                "IsContact": True,
                "IsWeapon": False,
                "unitType": 0,
                "unitCategory": 0,
                "longitude": 120.1 + (index % 25) * 0.0001,
                "latitude": 30.1 + (index // 25) * 0.0001,
                "altitude": 8000.0,
                "heading": 270.0,
                "BloodAmount": 100.0,
                "Icon2D": "/Aircraft/acceptance-target.png",
            }
        )

    return {
        "data": {
            "sideGuid": own_side_id,
            "data": {
                "ScenName": "symbolic-reasoning-worst-case",
                "CurrentTime": "2026-01-01T00:00:00+00:00",
                "UnitList": units,
            },
        }
    }


class _InstrumentedPerformanceEnv(SymbolicReasoningEnv):
    """核验性能样本实际经过了全部目标和导弹候选遍历。"""

    def __init__(self) -> None:
        super().__init__(max_entities=WORST_CASE_MAX_ENTITIES)
        self.target_evaluations = 0
        self.incoming_candidates_scanned = 0

    def reset_performance_counters(self) -> None:
        self.target_evaluations = 0
        self.incoming_candidates_scanned = 0

    def _evaluate_target(
        self,
        own: EncodedEntity,
        target: EncodedEntity,
        planned_attackers: Mapping[str, Set[str]],
        planned_interceptors: Mapping[str, int],
    ) -> TargetEvaluation:
        self.target_evaluations += 1
        return super()._evaluate_target(
            own,
            target,
            planned_attackers,
            planned_interceptors,
        )

    def _find_incoming_missile(
        self, own: EncodedEntity, targets: Sequence[EncodedEntity]
    ) -> Optional[EncodedEntity]:
        self.incoming_candidates_scanned += len(targets)
        return super()._find_incoming_missile(own, targets)


def test_performance(
    agent: SymbolicReasoningAgent,
    iterations: int,
    max_p95_ms: float,
    max_peak_memory_mib: float,
    worst_case_iterations: int = 5,
    max_worst_case_p95_ms: float = 5000.0,
    max_worst_case_peak_memory_mib: float = 128.0,
    max_explanation_p95_ms: float = 5.0,
) -> Dict[str, Any]:
    facts = _facts()
    reasoning_core, core_decision = _benchmark_operation(
        lambda: agent.reason(facts),
        iterations,
    )
    reasoning_core.update(
        {
            "passed": (
                reasoning_core["p95_ms"] <= max_p95_ms
                and reasoning_core["peak_memory_mib"]
                <= max_peak_memory_mib
            ),
            "max_p95_ms": max_p95_ms,
            "max_peak_memory_mib": max_peak_memory_mib,
        }
    )

    reasoning_with_explanation, full_explanation = _benchmark_operation(
        lambda: agent.reason(facts).explanation,
        iterations,
    )
    explanation_complete = _complete_explanation_text(
        core_decision,
        full_explanation,
    )
    reasoning_with_explanation.update(
        {
            "passed": (
                explanation_complete
                and reasoning_with_explanation["p95_ms"]
                <= max_explanation_p95_ms
                and reasoning_with_explanation["peak_memory_mib"]
                <= max_peak_memory_mib
            ),
            "complete_explanation_verified": explanation_complete,
            "explanation_characters": len(full_explanation),
            "max_p95_ms": max_explanation_p95_ms,
            "max_peak_memory_mib": max_peak_memory_mib,
        }
    )

    payload = _worst_case_payload()
    performance_env = _InstrumentedPerformanceEnv()

    def run_worst_case() -> Dict[str, Any]:
        performance_env.reset_performance_counters()
        result = performance_env.step(payload, execute_commands=False)
        explanations = tuple(
            decision.explanation for decision in result.decisions.values()
        )
        return {
            "entities": len(result.situation.entities),
            "own_entities": len(result.situation.own_entities),
            "target_entities": len(result.situation.targets),
            "target_evaluations": performance_env.target_evaluations,
            "incoming_candidates_scanned": (
                performance_env.incoming_candidates_scanned
            ),
            "decisions": len(result.decisions),
            "generated_explanations": len(explanations),
            "explanation_characters": sum(len(item) for item in explanations),
        }

    worst_case, observation = _benchmark_operation(
        run_worst_case,
        worst_case_iterations,
        memory_iterations=1,
    )
    verification_result = performance_env.step(payload, execute_commands=False)
    observation["complete_explanations"] = sum(
        _complete_explanation_text(decision, decision.explanation)
        for decision in verification_result.decisions.values()
    )
    workload_verified = (
        observation["entities"] == WORST_CASE_MAX_ENTITIES
        and observation["own_entities"] == WORST_CASE_OWN_ENTITIES
        and observation["target_entities"] == WORST_CASE_TARGET_ENTITIES
        and observation["target_evaluations"]
        == WORST_CASE_TARGET_EVALUATIONS
        and observation["incoming_candidates_scanned"]
        == WORST_CASE_TARGET_EVALUATIONS
        and observation["decisions"] == WORST_CASE_OWN_ENTITIES
        and observation["generated_explanations"]
        == WORST_CASE_OWN_ENTITIES
        and observation["complete_explanations"]
        == WORST_CASE_OWN_ENTITIES
        and observation["explanation_characters"] > 0
    )
    worst_case.update(observation)
    worst_case.update(
        {
            "passed": (
                workload_verified
                and worst_case["p95_ms"] <= max_worst_case_p95_ms
                and worst_case["peak_memory_mib"]
                <= max_worst_case_peak_memory_mib
            ),
            "workload_verified": workload_verified,
            "max_p95_ms": max_worst_case_p95_ms,
            "max_peak_memory_mib": max_worst_case_peak_memory_mib,
        }
    )

    return {
        "passed": all(
            item["passed"]
            for item in (
                reasoning_core,
                reasoning_with_explanation,
                worst_case,
            )
        ),
        "reasoning_core": reasoning_core,
        "reasoning_with_full_explanation": reasoning_with_explanation,
        "worst_case_end_to_end": worst_case,
    }


def run_acceptance(
    performance_iterations: int = 10000,
    max_p95_ms: float = 5.0,
    max_peak_memory_mib: float = 16.0,
    worst_case_iterations: int = 5,
    max_worst_case_p95_ms: float = 5000.0,
    max_worst_case_peak_memory_mib: float = 128.0,
    max_explanation_p95_ms: float = 5.0,
) -> Dict[str, Any]:
    agent = SymbolicReasoningAgent()
    requirements = {
        "correctness": test_correctness(agent),
        "coverage": test_coverage(agent),
        "explainability": test_explainability(agent),
        "performance": test_performance(
            agent,
            iterations=performance_iterations,
            max_p95_ms=max_p95_ms,
            max_peak_memory_mib=max_peak_memory_mib,
            worst_case_iterations=worst_case_iterations,
            max_worst_case_p95_ms=max_worst_case_p95_ms,
            max_worst_case_peak_memory_mib=(
                max_worst_case_peak_memory_mib
            ),
            max_explanation_p95_ms=max_explanation_p95_ms,
        ),
    }
    return {
        "passed": all(item["passed"] for item in requirements.values()),
        "requirements": requirements,
    }


def _print_report(report: Dict[str, Any]) -> None:
    req = report["requirements"]
    for name, label in (
        ("correctness", "正确性"),
        ("coverage", "覆盖性"),
        ("explainability", "可解释性"),
        ("performance", "效率/性能"),
    ):
        status = "PASS" if req[name]["passed"] else "FAIL"
        print(
            "[{}] {}: {}".format(
                status, label, json.dumps(req[name], ensure_ascii=False)
            )
        )
    print("总体结果：{}".format("PASS" if report["passed"] else "FAIL"))


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description="符号推理四项自动验收测试")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--max-p95-ms", type=float, default=5.0)
    parser.add_argument("--max-memory-mib", type=float, default=16.0)
    parser.add_argument("--worst-case-iterations", type=int, default=5)
    parser.add_argument(
        "--max-worst-case-p95-ms", type=float, default=5000.0
    )
    parser.add_argument(
        "--max-worst-case-memory-mib", type=float, default=128.0
    )
    parser.add_argument(
        "--max-explanation-p95-ms", type=float, default=5.0
    )
    args = parser.parse_args(argv)
    report = run_acceptance(
        performance_iterations=args.iterations,
        max_p95_ms=args.max_p95_ms,
        max_peak_memory_mib=args.max_memory_mib,
        worst_case_iterations=args.worst_case_iterations,
        max_worst_case_p95_ms=args.max_worst_case_p95_ms,
        max_worst_case_peak_memory_mib=args.max_worst_case_memory_mib,
        max_explanation_p95_ms=args.max_explanation_p95_ms,
    )
    _print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
