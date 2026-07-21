import unittest
from pathlib import Path
from types import SimpleNamespace

from symbolic_reasoning.control import (
    FrontendControl,
    FrontendStatus,
    TIME_COMPRESSION_MULTIPLIERS,
    frontend_status_from_engine,
)
from symbolic_reasoning.entity import EntityEncoder
from symbolic_reasoning.state import EngagementState
from symbolic_reasoning.symbolic_reasoning4test import (
    SymbolicReasoningEnv,
    main_loop,
)


def _payload():
    return {"sideGuid": "red-side", "UnitList": []}


class QuietLogger:
    def debug(self, message, *args):
        return None

    def info(self, message, *args):
        return None

    def warning(self, message, *args):
        return None


class TimeCompressionTests(unittest.TestCase):
    def test_engine_status_exposes_time_compression_and_sim_step(self):
        status = frontend_status_from_engine(
            SimpleNamespace(run_status=2, time_compress=7, sim_step="0.1")
        )

        self.assertEqual(status.signal, "running")
        self.assertEqual(status.time_compress_code, 7)
        self.assertEqual(status.time_compression_multiplier, 50)
        self.assertEqual(status.sim_step, "0.1")

    def test_frontend_control_updates_multiplier_when_ui_changes(self):
        control = FrontendControl(logger=QuietLogger())

        control.handle_signal(FrontendStatus("running", 1, "0.1"))
        self.assertEqual(control.time_compression_multiplier, 2)
        control.handle_signal(FrontendStatus("running", 7, "0.1"))

        self.assertEqual(control.state, "running")
        self.assertEqual(control.time_compress_code, 7)
        self.assertEqual(control.time_compression_multiplier, 50)

    def test_main_loop_advances_cooldown_clock_by_current_multiplier(self):
        env = SymbolicReasoningEnv(max_entities=2)
        control = FrontendControl(logger=QuietLogger())
        control.handle_signal(FrontendStatus("running", 7, "0.1"))

        main_loop(
            env=env,
            input_path=Path("unused.json"),
            steps=1,
            interval=0.0,
            execute_commands=False,
            logger=QuietLogger(),
            frontend_control=control,
            situation_provider=_payload,
        )

        self.assertEqual(env.current_step, 50)

    def test_multiplier_changes_do_not_reset_existing_cooldown_clock(self):
        env = SymbolicReasoningEnv(max_entities=2)

        env.step(_payload(), execute_commands=False, time_compression_multiplier=1)
        env.step(_payload(), execute_commands=False, time_compression_multiplier=50)
        env.step(_payload(), execute_commands=False, time_compression_multiplier=2)

        self.assertEqual(env.current_step, 53)

    def test_six_hundred_frame_timeout_expires_after_twelve_50x_frames(self):
        state = EngagementState(
            timeout_frames=600,
            weapon_appearance_grace_frames=10000,
        )
        state.record_successful_attack(
            attacker_id="red-1",
            target_id="blue-1",
            started_frame=0,
            target_is_missile=False,
        )
        situation = EntityEncoder(max_entities=2).encode(_payload())

        state.update_from_situation(situation, current_frame=550)
        self.assertTrue(state.is_attacking("red-1", "blue-1"))
        state.update_from_situation(situation, current_frame=600)

        self.assertFalse(state.is_attacking("red-1", "blue-1"))

    def test_contact_loss_debounce_counts_received_snapshots_not_multiplier(self):
        state = EngagementState(
            timeout_frames=10000,
            weapon_appearance_grace_frames=10000,
        )
        state.record_successful_attack(
            attacker_id="red-1",
            target_id="blue-1",
            started_frame=0,
            target_is_missile=False,
        )
        situation = EntityEncoder(max_entities=2).encode(_payload())

        state.update_from_situation(situation, current_frame=50)
        self.assertEqual(state.target_missing_frames("red-1", "blue-1"), 1)
        state.update_from_situation(situation, current_frame=100)
        self.assertEqual(state.target_missing_frames("red-1", "blue-1"), 2)
        state.update_from_situation(situation, current_frame=150)

        self.assertEqual(state.target_missing_frames("red-1", "blue-1"), 3)

    def test_turbo_uses_documented_sixty_times_fallback(self):
        self.assertEqual(TIME_COMPRESSION_MULTIPLIERS[9], 60)


if __name__ == "__main__":
    unittest.main()
