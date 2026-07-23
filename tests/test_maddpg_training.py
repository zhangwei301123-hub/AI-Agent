import random
import unittest
from unittest.mock import patch

import numpy as np

from MADDPG import maddpg_ as maddpg_training
from MADDPG import maddpg_live_adapter
from symbolic_reasoning.state import EngagementState


class MaddpgTrainingConfigurationTest(unittest.TestCase):
    def test_training_controls_red_side(self):
        self.assertEqual(1, maddpg_training.OUR_SIDE)
        self.assertEqual(0, maddpg_training.ENEMY_SIDE)

    def test_training_env_uses_one_hundred_frame_engagement_state(self):
        self.assertEqual(100, maddpg_training.ATTACK_SLOT_TIMEOUT_FRAMES)

        env = maddpg_training.SimulatedEnv()

        self.assertIsInstance(env.engagement_state, EngagementState)
        self.assertEqual(100, env.engagement_state.timeout_frames)

    def test_keyboard_interrupt_exit_does_not_wait_for_library_threads(self):
        with patch.object(maddpg_training, "pause", return_value=None), patch.object(
            maddpg_training.os,
            "_exit",
            side_effect=SystemExit(130),
        ) as hard_exit:
            with self.assertRaises(SystemExit):
                maddpg_training._exit_after_keyboard_interrupt(pause_timeout=0.1)

        hard_exit.assert_called_once_with(130)


class LiveAdapterReportTimeTest(unittest.TestCase):
    def test_iso_scenario_time_is_converted_to_elapsed_seconds(self):
        maddpg_live_adapter.set_report_time_origin("2026年07月17日 02:06:07")

        elapsed = maddpg_live_adapter._numeric_report_time(
            "2026-07-17T02:13:40.5Z"
        )

        self.assertEqual(453.5, elapsed)

    def test_legacy_numeric_report_time_is_unchanged(self):
        maddpg_live_adapter.set_report_time_origin(None)

        self.assertEqual(251.0, maddpg_live_adapter._numeric_report_time("251"))


class ReplayBufferInvalidMaskTest(unittest.TestCase):
    @staticmethod
    def _experience(index, *, valid):
        mask = np.array([valid], dtype=np.bool_)
        state = {
            "encoded_data": np.zeros((1, 38), dtype=np.float32),
            "mask": mask,
        }
        next_state = {
            "encoded_data": np.zeros((1, 38), dtype=np.float32),
            "mask": mask.copy(),
        }
        return {
            "states": state,
            "actions": np.zeros((8, 1, 5), dtype=np.float32),
            "actions_mask": np.ones((8, 1), dtype=np.bool_),
            "rewards": np.zeros(8, dtype=np.float32),
            "next_states": next_state,
            "dones": False,
            "step": index,
            "action_entity_id": [f"entity-{index}"],
            "actions_executed": np.zeros((1, 8), dtype=np.bool_),
        }

    @staticmethod
    def _bounded_sample():
        calls = 0

        def sample(population, count):
            nonlocal calls
            calls += 1
            if calls > 32:
                raise AssertionError(
                    "ReplayBuffer.sample kept retrying invalid samples"
                )
            if calls == 1:
                return list(population)[-count:]
            if calls == 2:
                return list(population)[:count]
            valid = [
                item
                for item in population
                if maddpg_training.ReplayBuffer._has_valid_mask(None, item)
            ]
            if len(valid) < count:
                return random.sample(list(population), count)
            return valid[:count]

        return sample

    def _buffer(self, experiences):
        buffer = maddpg_training.ReplayBuffer(capacity=100)
        buffer.start_new_episode()
        for experience in experiences:
            buffer.add(experience)
        return buffer

    def test_sample_replaces_invalid_masks_without_type_error_or_retry_loop(self):
        experiences = [
            self._experience(index, valid=index < 24)
            for index in range(40)
        ]
        buffer = self._buffer(experiences)

        with patch.object(
            maddpg_training.random,
            "sample",
            side_effect=self._bounded_sample(),
        ):
            batch = buffer.sample(batch_size=16)

        self.assertEqual(16, len(batch["states"]))
        self.assertTrue(
            all(
                np.asarray(state["mask"], dtype=bool).any()
                and np.asarray(next_state["mask"], dtype=bool).any()
                for state, next_state in zip(
                    batch["states"], batch["next_states"]
                )
            )
        )

    def test_sample_raises_clear_value_error_when_valid_samples_are_insufficient(self):
        experiences = [
            self._experience(index, valid=index < 8)
            for index in range(40)
        ]
        buffer = self._buffer(experiences)

        with patch.object(
            maddpg_training.random,
            "sample",
            side_effect=self._bounded_sample(),
        ):
            with self.assertRaises(ValueError) as caught:
                buffer.sample(batch_size=16)

        self.assertTrue(str(caught.exception).strip())


if __name__ == "__main__":
    unittest.main()
