import unittest

import numpy as np
import torch

from actor_rules import handle_attack_decision, handle_deploy


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class MaddpgActorRuleIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.actions = np.zeros((8, 1, 5), dtype=np.float32)
        self.actions[4, 0, 0] = 0.9
        self.list_actions = self.actions[:, 0, :].tolist()
        self.raw_data = [
            {
                "mdlID": "own",
                "mdlType": "AIRCRAFT",
                "unitCategory": "Aircraft_Fighter",
                "unitTarget": [0],
                "weaponNumber": {"airNum": 4, "shipNum": 0, "subNum": 0},
                "maxRange": {"maxAir": 50, "maxSurface": 0, "maxSubsurface": 0},
                "logisticStates": {"oil": 0.9},
                "innerstates": {"IsJamReaction": 0},
            },
            {
                "mdlID": "target",
                "mdlType": "AIRCRAFT",
                "contactType": 0,
            },
        ]
        encoded = torch.zeros((2, 38), dtype=torch.float32)
        encoded[0, 4] = 0.0
        encoded[0, 5] = 1000.0
        encoded[0, 13] = 0.0
        encoded[1, 5] = 1000.0
        encoded[1, 13] = 0.0
        self.now_state = {"encoded_data": encoded}

    def test_attack_decision_writes_one_target_id_and_quantity(self):
        counts = handle_attack_decision(
            self.actions,
            self.now_state,
            0,
            0,
            "own",
            [0],
            [0],
            ["target"],
            [1],
            [(0.0, 0.05)],
            {},
            0,
            0,
            0.6,
            self.list_actions,
            self.raw_data,
            [(0.0, 0.0)],
            14,
            _Logger(),
            False,
            0,
        )
        self.assertEqual((0, 0, 0), counts)
        self.assertGreaterEqual(self.list_actions[4][0], 0.6)
        self.assertEqual("target", self.list_actions[4][1])
        self.assertEqual(2, self.list_actions[4][4])

    def test_sonobuoy_deployment_forces_active_shallow_and_disables_route(self):
        raw_data = [
            {
                "mdlID": "patrol",
                "mdlType": "AIRCRAFT",
                "unitCategory": "Aircraft_ASW",
                "missionId": "mission",
                "weaponNumber": {"buoyNum": 2},
            }
        ]
        actions = np.zeros((8, 1, 5), dtype=np.float32)
        actions[2, 0, 0] = 0.9
        actions[6, 0, 0] = 0.9
        list_actions = actions[:, 0, :].tolist()
        rejected = handle_deploy(
            list_actions,
            actions,
            0,
            0,
            [(0.0, 0.5)],
            raw_data,
            {
                "mission": {
                    "area_points": [
                        (0.0, 0.0),
                        (1.0, 0.0),
                        (1.0, 1.0),
                        (0.0, 1.0),
                    ]
                }
            },
            500.0,
            0.6,
        )
        self.assertEqual(0, rejected)
        self.assertEqual(1.0, list_actions[6][1])
        self.assertEqual(1.0, list_actions[6][2])
        self.assertLess(list_actions[2][0], 0.6)


if __name__ == "__main__":
    unittest.main()
