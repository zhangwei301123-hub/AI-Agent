from __future__ import annotations

import unittest

from symbolic_reasoning.acceptance import (
    WORST_CASE_MAX_ENTITIES,
    WORST_CASE_OWN_ENTITIES,
    WORST_CASE_TARGET_ENTITIES,
    WORST_CASE_TARGET_EVALUATIONS,
    run_acceptance,
)


class SymbolicReasoningAcceptanceTests(unittest.TestCase):
    def test_all_four_required_test_groups_pass(self):
        report = run_acceptance(
            performance_iterations=500,
            worst_case_iterations=2,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["requirements"]["correctness"]["passed"])
        self.assertTrue(report["requirements"]["coverage"]["passed"])
        self.assertTrue(report["requirements"]["explainability"]["passed"])
        self.assertTrue(report["requirements"]["performance"]["passed"])
        self.assertEqual(
            report["requirements"]["coverage"]["tested_combinations"], 32768
        )
        self.assertEqual(
            report["requirements"]["coverage"]["invalid_conclusions"], 0
        )
        self.assertTrue(
            report["requirements"]["coverage"]["all_decisive_rules_covered"]
        )
        self.assertEqual(
            report["requirements"]["explainability"]["explainable"], 32768
        )

        performance = report["requirements"]["performance"]
        self.assertTrue(performance["reasoning_core"]["passed"])
        explanation = performance["reasoning_with_full_explanation"]
        self.assertTrue(explanation["passed"])
        self.assertTrue(explanation["complete_explanation_verified"])

        worst_case = performance["worst_case_end_to_end"]
        self.assertTrue(worst_case["passed"])
        self.assertTrue(worst_case["workload_verified"])
        self.assertEqual(worst_case["entities"], WORST_CASE_MAX_ENTITIES)
        self.assertEqual(
            worst_case["own_entities"], WORST_CASE_OWN_ENTITIES
        )
        self.assertEqual(
            worst_case["target_entities"], WORST_CASE_TARGET_ENTITIES
        )
        self.assertEqual(
            worst_case["target_evaluations"],
            WORST_CASE_TARGET_EVALUATIONS,
        )
        self.assertEqual(
            worst_case["incoming_candidates_scanned"],
            WORST_CASE_TARGET_EVALUATIONS,
        )
        self.assertEqual(
            worst_case["complete_explanations"],
            WORST_CASE_OWN_ENTITIES,
        )
        self.assertGreater(worst_case["explanation_characters"], 0)


if __name__ == "__main__":
    unittest.main()
