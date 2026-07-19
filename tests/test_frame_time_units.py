from __future__ import annotations

import unittest

from symbolic_reasoning.state import ATTACK_SLOT_TIMEOUT_FRAMES, EngagementState
from symbolic_reasoning.symbolic_reasoning4test import SymbolicReasoningEnv


class _Target:
    command_id = "target-1"
    entity_id = "target-1"
    is_own = False
    is_weapon = False
    weapon_impact = 0
    weapon_target_id = None


class _Situation:
    targets = (_Target(),)
    entities = targets
    deleted_entity_ids = ()

    @staticmethod
    def find_entity(_entity_id):
        return None


class FrameTimeUnitTests(unittest.TestCase):
    @staticmethod
    def _empty_payload(current_time):
        return {
            "data": {
                "sideGuid": "red-side",
                "CurrentTime": current_time,
                "data": {"UnitList": []},
            }
        }

    def test_attack_slot_timeout_is_counted_in_frames(self):
        self.assertEqual(ATTACK_SLOT_TIMEOUT_FRAMES, 600)
        state = EngagementState()
        state.record_successful_attack(
            "attacker-1", "target-1", started_frame=0, target_is_missile=False
        )

        state.update_from_situation(_Situation(), current_frame=599)
        self.assertEqual(state.active_attackers("target-1"), 1)

        state.update_from_situation(_Situation(), current_frame=600)
        self.assertEqual(state.active_attackers("target-1"), 0)

    def test_environment_ignores_scenario_time_jump_for_frame_timeout(self):
        state = EngagementState(timeout_frames=2)
        state.record_successful_attack(
            "attacker-1", "target-1", started_frame=0, target_is_missile=False
        )
        env = SymbolicReasoningEnv(max_entities=1, engagement_state=state)

        env.step(self._empty_payload("2026-01-01 00:00:00"), execute_commands=False)
        env.step(self._empty_payload("2036-01-01 00:00:00"), execute_commands=False)
        self.assertEqual(state.active_attackers("target-1"), 1)

        env.step(self._empty_payload("2036-01-01 00:00:01"), execute_commands=False)
        self.assertEqual(state.active_attackers("target-1"), 0)

if __name__ == "__main__":
    unittest.main()
