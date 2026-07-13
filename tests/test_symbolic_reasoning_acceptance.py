from __future__ import annotations

import unittest

from symbolic_reasoning.acceptance import run_acceptance


class SymbolicReasoningAcceptanceTests(unittest.TestCase):
    def test_all_four_required_test_groups_pass(self):
        report = run_acceptance(performance_iterations=500)

        self.assertTrue(report["passed"])
        self.assertTrue(report["requirements"]["correctness"]["passed"])
        self.assertTrue(report["requirements"]["coverage"]["passed"])
        self.assertTrue(report["requirements"]["explainability"]["passed"])
        self.assertTrue(report["requirements"]["performance"]["passed"])
        self.assertEqual(
            report["requirements"]["coverage"]["tested_combinations"], 512
        )
        self.assertEqual(report["requirements"]["coverage"]["fired_rules"], 9)
        self.assertEqual(
            report["requirements"]["explainability"]["explainable"], 512
        )


if __name__ == "__main__":
    unittest.main()

