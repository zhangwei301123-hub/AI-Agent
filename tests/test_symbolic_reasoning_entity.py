from __future__ import annotations

import math
import unittest
from pathlib import Path

from symbolic_reasoning import EntityEncoder, FEATURE_NAMES, TargetDomain, load_situation
from symbolic_reasoning.symbolic_reasoning4test import SymbolicReasoningEnv, log_step


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "source" / "我方视角下态势完整响应.txt"


class EntityEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_situation(SAMPLE)
        cls.situation = EntityEncoder(max_entities=64).encode(cls.payload)

    def test_encodes_complete_sample_response(self):
        self.assertEqual(len(self.situation.entities), 34)
        self.assertEqual(self.situation.encoded_data.shape, (64, len(FEATURE_NAMES)))
        self.assertEqual(int(self.situation.mask.sum()), 34)
        self.assertEqual(len(self.situation.own_entities), 18)
        self.assertEqual(len(self.situation.targets), 16)

    def test_encodes_documented_fields_for_first_entity(self):
        entity = self.situation.entities[0]

        self.assertEqual(entity.entity_id, "7b768d56-cfa7-4116-849d-518d7d8b1319")
        self.assertTrue(entity.is_own)
        self.assertEqual(entity.unit_category, 0)
        self.assertEqual(entity.domain, TargetDomain.AIR)
        self.assertAlmostEqual(entity.longitude, 122.0376892697069)
        self.assertAlmostEqual(entity.health_pct, 100.0)
        self.assertEqual(entity.vector.shape, (len(FEATURE_NAMES),))

    def test_contact_uses_contact_guid_and_icon_domain(self):
        ship = next(
            entity
            for entity in self.situation.targets
            if "/Ship/" in entity.icon_2d
        )
        submarine = next(
            entity
            for entity in self.situation.targets
            if "/Submarine/" in entity.icon_2d
        )

        self.assertEqual(ship.command_id, ship.contact_id)
        self.assertEqual(ship.domain, TargetDomain.SURFACE)
        self.assertEqual(submarine.domain, TargetDomain.SUBMARINE)

    def test_non_finite_source_number_is_normalized(self):
        water_contact = next(
            entity
            for entity in self.situation.targets
            if entity.altitude_m == -1.0
        )
        self.assertTrue(math.isfinite(water_contact.altitude_m))


class SymbolicReasoningEnvironmentTests(unittest.TestCase):
    def test_info_log_contains_complete_explanation(self):
        class CapturingLogger:
            def __init__(self):
                self.messages = []

            def info(self, message, *args):
                self.messages.append(message % args if args else str(message))

        payload = load_situation(SAMPLE)
        result = SymbolicReasoningEnv(max_entities=64).step(
            payload, execute_commands=False
        )
        logger = CapturingLogger()

        log_step(result, step_index=0, logger=logger)

        log_text = "\n".join(logger.messages)
        self.assertIn("step=0", log_text)
        self.assertIn("推理路径：", log_text)
        self.assertIn("execution=DRY_RUN", log_text)
        self.assertIn("决定规则", log_text)

    def test_sample_runs_through_encoding_and_reasoning(self):
        payload = load_situation(SAMPLE)
        result = SymbolicReasoningEnv(max_entities=64).step(
            payload, execute_commands=False
        )

        self.assertEqual(len(result.decisions), 18)
        self.assertEqual(set(result.decisions), set(result.actions_dict))
        for actions in result.actions_dict.values():
            self.assertEqual(len(actions), 8)
            self.assertTrue(all(len(action) == 5 for action in actions))

    def test_step_executes_by_default_through_execute_interface(self):
        own_id = "our-aircraft-0001"
        target_id = "enemy-aircraft-0001"
        payload = {
            "data": {
                "sideGuid": "red-side",
                "data": {
                    "ScenName": "test",
                    "CurrentTime": "2026-07-13 12:00:00",
                    "UnitList": [
                        {
                            "guid": own_id,
                            "name": "own",
                            "longitude": 120.0,
                            "latitude": 30.0,
                            "altitude": 1000.0,
                            "heading": 90.0,
                            "SideId": "red-side",
                            "unitType": 0,
                            "unitCategory": 0,
                            "Icon2D": "/ArmyIcon/Aircraft/test.svg",
                            "Speed": 500,
                            "BloodAmount": 100,
                            "IsContact": False,
                            "IsWeapon": False,
                            "rangeStrike_Air": 100,
                            "rangeStrike_Surface": 0,
                            "rangeStrike_Land": 0,
                            "rangeStrike_Submarine": 0,
                            "weaponNumber": {"airNum": 2},
                            "CommText": "",
                            "JammText": ""
                        },
                        {
                            "guid": "target-physical-guid",
                            "contactGuid": target_id,
                            "name": "target",
                            "longitude": 120.01,
                            "latitude": 30.0,
                            "altitude": 1000.0,
                            "heading": 270.0,
                            "SideId": "blue-side",
                            "unitType": 0,
                            "unitCategory": 0,
                            "Icon2D": "/ArmyIcon/Aircraft/test.svg",
                            "Speed": 500,
                            "BloodAmount": 100,
                            "IsContact": True,
                            "IsWeapon": False
                        }
                    ]
                }
            }
        }
        captured = {}

        def fake_execute(actions_dict, enemy_ids, probability, logger):
            captured["actions_dict"] = actions_dict
            captured["enemy_ids"] = enemy_ids
            return {own_id: [False, False, False, False, True]}, [0] * 8

        result = SymbolicReasoningEnv(max_entities=8).step(
            payload,
            executor=fake_execute,
        )

        self.assertEqual(
            result.decisions[own_id].conclusion.value,
            "REQUEST_ATTACK",
        )
        self.assertEqual(captured["enemy_ids"], [target_id])
        self.assertEqual(captured["actions_dict"][own_id][4][1], target_id)
        self.assertEqual(result.execution_status[own_id], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
