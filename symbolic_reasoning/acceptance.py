"""自动执行符号推理正确性、覆盖性、可解释性和性能测试。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import tracemalloc
from time import perf_counter_ns
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .agent import Conclusion, Decision, ReasoningFacts, SymbolicReasoningAgent


RULE_IDS = {"SR-{:03d}".format(index) for index in range(1, 10)}
BOOLEAN_FACT_NAMES = (
    "need_return_to_base",
    "attack_authorized",
    "target_type_allowed",
    "weapon_available",
    "within_attack_range",
    "aimed_at_target",
    "safety_clearance",
    "chase_allowed",
)


def _facts(**changes: Any) -> ReasoningFacts:
    values: Dict[str, Any] = {
        "entity_id": "our-aircraft-0001",
        "target_id": "enemy-aircraft-0001",
        "need_return_to_base": False,
        "attack_authorized": True,
        "target_type_allowed": True,
        "weapon_available": True,
        "within_attack_range": True,
        "aimed_at_target": True,
        "safety_clearance": True,
        "chase_allowed": True,
        "target_lon": 120.1,
        "target_lat": 30.2,
    }
    values.update(changes)
    return ReasoningFacts(**values)


def correctness_cases() -> Tuple[Tuple[str, ReasoningFacts, Conclusion, str], ...]:
    """九条规则各提供一个独立的期望结论。"""

    return (
        (
            "COR-001",
            _facts(need_return_to_base=True),
            Conclusion.RETURN_TO_BASE,
            "SR-001",
        ),
        ("COR-002", _facts(target_id=None), Conclusion.SEARCH, "SR-002"),
        (
            "COR-003",
            _facts(attack_authorized=False),
            Conclusion.HOLD,
            "SR-003",
        ),
        (
            "COR-004",
            _facts(safety_clearance=False),
            Conclusion.HOLD,
            "SR-004",
        ),
        (
            "COR-005",
            _facts(target_type_allowed=False),
            Conclusion.HOLD,
            "SR-005",
        ),
        (
            "COR-006",
            _facts(weapon_available=False),
            Conclusion.HOLD,
            "SR-006",
        ),
        ("COR-007", _facts(), Conclusion.ATTACK, "SR-007"),
        (
            "COR-008",
            _facts(within_attack_range=False),
            Conclusion.CHASE,
            "SR-008",
        ),
        (
            "COR-009",
            _facts(within_attack_range=False, chase_allowed=False),
            Conclusion.HOLD,
            "SR-009",
        ),
    )


def _valid_decision(decision: Decision) -> bool:
    return (
        isinstance(decision.conclusion, Conclusion)
        and decision.rule_id in RULE_IDS
        and len(decision.actions) == 8
        and all(len(action) == 5 for action in decision.actions)
    )


def _valid_explanation(decision: Decision) -> bool:
    if not decision.inference_path:
        return False
    if decision.inference_path[-1].rule_id != decision.rule_id:
        return False
    if not decision.inference_path[-1].matched:
        return False
    if any(step.matched for step in decision.inference_path[:-1]):
        return False
    explanation = decision.explanation
    return (
        "推理路径" in explanation
        and "结论" in explanation
        and all(step.rule_id in explanation for step in decision.inference_path)
        and all(step.evidence for step in decision.inference_path)
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
    # 8 个布尔事实 + 目标存在性，共 2^9 = 512 种任务域组合。
    for values in itertools.product((False, True), repeat=9):
        has_target = values[0]
        booleans = dict(zip(BOOLEAN_FACT_NAMES, values[1:]))
        yield _facts(
            target_id="enemy-aircraft-0001" if has_target else None,
            **booleans
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

    possible = 2 ** 9
    passed = total == possible and invalid == 0 and fired_rules == RULE_IDS
    return {
        "passed": passed,
        "possible_combinations": possible,
        "tested_combinations": total,
        "valid_conclusions": total - invalid,
        "invalid_conclusions": invalid,
        "fired_rules": len(fired_rules),
        "total_rules": len(RULE_IDS),
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
        else:
            failures.append(index)
    return {
        "passed": explainable == total,
        "total": total,
        "explainable": explainable,
        "failed_combination_indexes": failures[:20],
    }


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def test_performance(
    agent: SymbolicReasoningAgent,
    iterations: int,
    max_p95_ms: float,
    max_peak_memory_mib: float,
) -> Dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations 必须大于 0")
    facts = _facts()
    for _ in range(min(100, iterations)):
        agent.reason(facts)

    durations_ms: List[float] = []
    for _ in range(iterations):
        start_ns = perf_counter_ns()
        agent.reason(facts)
        durations_ms.append((perf_counter_ns() - start_ns) / 1_000_000.0)

    tracemalloc.start()
    for _ in range(iterations):
        agent.reason(facts)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mean_ms = statistics.mean(durations_ms)
    p95_ms = _percentile(durations_ms, 0.95)
    peak_memory_mib = peak_bytes / (1024.0 * 1024.0)
    return {
        "passed": p95_ms <= max_p95_ms and peak_memory_mib <= max_peak_memory_mib,
        "iterations": iterations,
        "mean_ms": round(mean_ms, 6),
        "p95_ms": round(p95_ms, 6),
        "max_ms": round(max(durations_ms), 6),
        "max_p95_ms": max_p95_ms,
        "peak_memory_mib": round(peak_memory_mib, 6),
        "max_peak_memory_mib": max_peak_memory_mib,
    }


def run_acceptance(
    performance_iterations: int = 10000,
    max_p95_ms: float = 5.0,
    max_peak_memory_mib: float = 16.0,
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
        print("[{}] {}: {}".format(status, label, json.dumps(req[name], ensure_ascii=False)))
    print("总体结果：{}".format("PASS" if report["passed"] else "FAIL"))


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description="符号推理四项自动验收测试")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--max-p95-ms", type=float, default=5.0)
    parser.add_argument("--max-memory-mib", type=float, default=16.0)
    args = parser.parse_args(argv)
    report = run_acceptance(
        performance_iterations=args.iterations,
        max_p95_ms=args.max_p95_ms,
        max_peak_memory_mib=args.max_memory_mib,
    )
    _print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

