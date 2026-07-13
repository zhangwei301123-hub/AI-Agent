from __future__ import annotations

import unittest

from symbolic_reasoning import (
    ActionValidationError,
    ReasoningFacts,
    SymbolicReasoningAgent,
    TargetDomain,
    execute_actions,
    validate_actions_dict,
)


ENTITY_ID = "our-aircraft-0001"
TARGET_ID = "enemy-aircraft-0001"


class ExecuteActionsTests(unittest.TestCase):
    def setUp(self):
        facts = ReasoningFacts(
            entity_id=ENTITY_ID,
            own_platform_type=TargetDomain.AIR,
            target_id=TARGET_ID,
            target_domain=TargetDomain.AIR,
            detected_target_count=1,
            attack_authorized=True,
            target_type_allowed=True,
            weapon_available=True,
            compatible_weapon_count=2,
            within_attack_range=True,
            distance_km=50.0,
            max_attack_range_km=100.0,
            aimed_at_target=True,
            heading_difference_deg=10.0,
            safety_clearance=True,
            concurrency_slot_available=True,
            target_lon=120.1,
            target_lat=30.2,
            attack_altitude_level=3,
        )
        self.actions_dict = {
            ENTITY_ID: SymbolicReasoningAgent().reason(facts).actions
        }

    def test_validates_eight_by_five_action_matrix(self):
        validate_actions_dict(self.actions_dict)

        invalid = {ENTITY_ID: self.actions_dict[ENTITY_ID][:-1]}
        with self.assertRaises(ActionValidationError):
            validate_actions_dict(invalid)

    def test_calls_injected_execution_backend(self):
        captured = {}

        def fake_backend(actions_dict, enemy_ids, probability, logger):
            captured["actions_dict"] = actions_dict
            captured["enemy_ids"] = enemy_ids
            captured["probability"] = probability
            return {ENTITY_ID: [False, False, False, False, True]}, [0] * 8

        results, rewards = execute_actions(
            self.actions_dict,
            [TARGET_ID],
            probablity=0.75,
            backend=fake_backend,
        )

        self.assertIs(captured["actions_dict"], self.actions_dict)
        self.assertEqual(captured["enemy_ids"], [TARGET_ID])
        self.assertEqual(captured["probability"], 0.75)
        self.assertTrue(results[ENTITY_ID][-1])
        self.assertEqual(rewards, [0] * 8)

    def test_agent_loads_package_execution_layer(self):
        executor = SymbolicReasoningAgent._load_execute_actions()
        self.assertEqual(executor.__module__, "symbolic_reasoning.execute_actions")


if __name__ == "__main__":
    unittest.main()
