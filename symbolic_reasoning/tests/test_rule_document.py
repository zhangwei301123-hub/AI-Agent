import re
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]


class RuleDocumentConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule_text = (PACKAGE_DIR / "rule.md").read_text(encoding="utf-8")
        cls.agent_text = (PACKAGE_DIR / "agent.py").read_text(encoding="utf-8")
        cls.actions_text = (PACKAGE_DIR / "execute_actions.py").read_text(
            encoding="utf-8"
        )

    def test_every_agent_rule_id_is_documented(self):
        rule_id_pattern = re.compile(r"R-[A-Z]+-[0-9]{3}")
        emitted_rule_ids = set(rule_id_pattern.findall(self.agent_text))
        documented_rule_ids = set(rule_id_pattern.findall(self.rule_text))
        self.assertFalse(
            emitted_rule_ids - documented_rule_ids,
            "rule.md 缺少代码规则：{}".format(
                sorted(emitted_rule_ids - documented_rule_ids)
            ),
        )

    def test_attack_pipeline_is_described_with_explicit_weapon_fields(self):
        for field in (
            "attacker_unit_id",
            "target_unit_id",
            "weapon_db_id",
            "quantity",
            "mode",
        ):
            self.assertIn(field, self.rule_text)
            self.assertIn(field, self.actions_text)
        self.assertIn("GetWeaponFiringInfo", self.rule_text)
        self.assertIn("suitable_weapons", self.rule_text)
        self.assertIn("selected_weapon_db_id", self.rule_text)
        self.assertIn('mode="manual"', self.rule_text)

    def test_obsolete_weapon_selection_claims_are_absent(self):
        obsolete_claims = (
            "执行接口只能选择“是否发射”",
            "真正发射何种武器由系统根据目标自动选择",
            "weapon_selection_delegated_to_system",
            '"weapon_selection": "SYSTEM"',
            '"execution_status": "PENDING_SYSTEM_FEEDBACK"',
            "## R-RNG-001～004",
            "依据：R-RNG-001～004",
        )
        for claim in obsolete_claims:
            self.assertNotIn(claim, self.rule_text)


if __name__ == "__main__":
    unittest.main()
