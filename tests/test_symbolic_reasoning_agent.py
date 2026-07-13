from __future__ import annotations

import unittest

from symbolic_reasoning import Conclusion, ReasoningFacts, SymbolicReasoningAgent


ENTITY_ID = "our-aircraft-0001"
TARGET_ID = "enemy-aircraft-0001"


def attack_facts(**changes):
    values = {
        "entity_id": ENTITY_ID,
        "target_id": TARGET_ID,
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


class SymbolicReasoningAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = SymbolicReasoningAgent()

    def test_return_to_base_has_highest_priority(self):
        decision = self.agent.reason(attack_facts(need_return_to_base=True))

        self.assertEqual(decision.conclusion, Conclusion.RETURN_TO_BASE)
        self.assertEqual(decision.rule_id, "SR-001")
        self.assertEqual(decision.actions[1][0], 0.9)
        self.assertEqual(decision.actions[4][0], 0.01)

    def test_attack_when_all_attack_facts_are_true(self):
        decision = self.agent.reason(attack_facts())

        self.assertEqual(decision.conclusion, Conclusion.ATTACK)
        self.assertEqual(decision.rule_id, "SR-007")
        self.assertEqual(decision.actions[4][1], TARGET_ID)
        self.assertIn("within_attack_range=True", decision.explanation)
        self.assertEqual(decision.inference_path[-1].rule_id, "SR-007")
        self.assertTrue(decision.inference_path[-1].matched)
        self.assertTrue(all(not step.matched for step in decision.inference_path[:-1]))

    def test_chase_when_target_is_out_of_range(self):
        decision = self.agent.reason(
            attack_facts(within_attack_range=False, aimed_at_target=False)
        )

        self.assertEqual(decision.conclusion, Conclusion.CHASE)
        self.assertEqual(decision.actions[2][1:3], [120.1, 30.2])
        self.assertEqual(decision.actions[4][0], 0.01)

    def test_search_when_there_is_no_target(self):
        decision = self.agent.reason(ReasoningFacts(entity_id=ENTITY_ID))

        self.assertEqual(decision.conclusion, Conclusion.SEARCH)
        self.assertEqual(decision.actions[5][0], 0.9)
        self.assertEqual(decision.actions[5][1], 1.0)

    def test_missing_safety_permission_fails_closed(self):
        decision = self.agent.reason(attack_facts(safety_clearance=False))

        self.assertEqual(decision.conclusion, Conclusion.HOLD)
        self.assertEqual(decision.rule_id, "SR-004")
        self.assertTrue(all(action[0] == 0.01 for action in decision.actions))

    def test_run_reuses_execute_actions_interface(self):
        captured = {}

        def fake_execute(actions_dict, enemy_ids, probability, logger):
            captured["actions_dict"] = actions_dict
            captured["enemy_ids"] = enemy_ids
            captured["probability"] = probability
            captured["logger"] = logger
            return {ENTITY_ID: [False, False, False, False, True]}, [0] * 8

        result = self.agent.run(
            [attack_facts()],
            enemy_ids=[TARGET_ID],
            probability=0.7,
            executor=fake_execute,
        )

        self.assertEqual(result.decisions[ENTITY_ID].conclusion, Conclusion.ATTACK)
        self.assertIs(result.actions_dict, captured["actions_dict"])
        self.assertEqual(captured["enemy_ids"], [TARGET_ID])
        self.assertEqual(captured["probability"], 0.7)
        self.assertEqual(result.execute_results[ENTITY_ID][-1], True)


if __name__ == "__main__":
    unittest.main()
