import io
import unittest
from contextlib import redirect_stdout

from symbolic_reasoning.acceptance import _print_report


def _report(passed=True):
    failures = []
    if not passed:
        failures.append(
            {
                "case_id": "missile_evasion",
                "expected_conclusion": "EVADE_MISSILE",
                "actual_conclusion": "HOLD",
                "expected_rule": "R-EVA-001",
                "actual_rule": "R-DEF-001",
            }
        )
    return {
        "passed": passed,
        "requirements": {
            "correctness": {
                "passed": passed,
                "total": 8,
                "passed_cases": 8 if passed else 7,
                "failures": failures,
            },
            "coverage": {
                "passed": True,
                "possible_combinations": 32768,
                "valid_conclusions": 32768,
                "lifecycle_tested_combinations": 256,
                "lifecycle_invalid_conclusions": 0,
                "lifecycle_possible_combinations": 256,
            },
            "explainability": {
                "passed": True,
                "total": 33024,
                "explainable": 33024,
            },
            "performance": {
                "passed": True,
                "reasoning_core": {
                    "p95_ms": 0.1234,
                    "peak_memory_mib": 0.5678,
                },
                "worst_case_end_to_end": {"p95_ms": 123.4567},
            },
        },
    }


class AcceptanceOutputTests(unittest.TestCase):
    def _render(self, report, details=False):
        output = io.StringIO()
        with redirect_stdout(output):
            _print_report(report, details=details)
        return output.getvalue()

    def test_default_output_is_a_short_summary(self):
        output = self._render(_report())
        self.assertIn("正确性：8/8", output)
        self.assertIn("事实组合 32768/32768", output)
        self.assertIn("可解释性：33024/33024", output)
        self.assertIn("总体结果：PASS", output)
        self.assertNotIn('"requirements"', output)

    def test_default_output_only_expands_failed_correctness_cases(self):
        output = self._render(_report(passed=False))
        self.assertIn("[FAIL] 正确性：7/8", output)
        self.assertIn("missile_evasion", output)
        self.assertIn("EVADE_MISSILE/R-EVA-001", output)
        self.assertIn("HOLD/R-DEF-001", output)

    def test_details_option_keeps_machine_readable_data(self):
        output = self._render(_report(), details=True)
        self.assertIn('"passed": true', output)
        self.assertIn('"passed_cases": 8', output)


if __name__ == "__main__":
    unittest.main()
