import unittest

from maddpg4test import ATTACK_SLOT_TIMEOUT_FRAMES, SignalListener, SimulatedEnv
from symbolic_reasoning.entity import EntityEncoder as SymbolicEntityEncoder


class _Env:
    def __init__(self):
        self.calls = []

    def reset(self, logger):
        self.calls.append("reset")

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")

    def stop(self):
        self.calls.append("stop")


class _Logger:
    pass


class SignalListenerTest(unittest.TestCase):
    def setUp(self):
        self.env = _Env()
        self.controller = SignalListener(self.env, _Logger())

    def test_pause_and_running_resume_the_main_loop(self):
        self.controller.handle_signal("start")
        self.controller.handle_signal("pause")

        self.assertTrue(self.controller.simulating)
        self.assertFalse(self.controller.control_event.is_set())

        self.controller.handle_signal("running")
        self.assertTrue(self.controller.simulating)
        self.assertTrue(self.controller.control_event.is_set())
        self.assertEqual(["reset", "pause", "resume"], self.env.calls)

    def test_stop_wakes_a_paused_loop_without_exiting_the_process(self):
        self.controller.handle_signal("start")
        self.controller.handle_signal("pause")
        self.controller.handle_signal("stop")

        self.assertFalse(self.controller.simulating)
        self.assertTrue(self.controller.control_event.is_set())
        self.assertFalse(self.controller.shutdown_event.is_set())

    def test_shutdown_wakes_waiters_and_marks_process_for_exit(self):
        self.controller.handle_signal("start")
        self.controller.handle_signal("pause")
        self.controller.shutdown()

        self.assertFalse(self.controller.simulating)
        self.assertTrue(self.controller.control_event.is_set())
        self.assertTrue(self.controller.shutdown_event.is_set())


class MaddpgEngagementLifecycleTest(unittest.TestCase):
    @staticmethod
    def _situation(with_weapon):
        units = [
            {
                "guid": "red-ship",
                "SideId": "red-side",
                "forceSide": "红方",
                "IsContact": False,
                "IsWeapon": False,
                "unitType": 1,
                "unitCategory": 1,
                "longitude": 120.0,
                "latitude": 30.0,
                "altitude": 0.0,
                "BloodAmount": 100.0,
            },
            {
                "guid": "blue-stable",
                "contactGuid": "blue-contact",
                "SideId": "blue-side",
                "forceSide": "蓝方",
                "IsContact": True,
                "IsWeapon": False,
                "unitType": 1,
                "unitCategory": 1,
                "longitude": 121.0,
                "latitude": 30.0,
                "altitude": 0.0,
                "BloodAmount": 100.0,
            },
        ]
        links = []
        if with_weapon:
            units.append(
                {
                    "guid": "red-weapon",
                    "SideId": "red-side",
                    "forceSide": "红方",
                    "IsContact": False,
                    "IsWeapon": True,
                    "unitType": 0,
                    "unitCategory": 0,
                    "longitude": 120.001,
                    "latitude": 30.0,
                    "altitude": 50.0,
                    "BloodAmount": 100.0,
                }
            )
            links.append(
                {
                    "Arr": [
                        {"unitguid": "red-weapon"},
                        {"unitguid": "blue-stable"},
                    ]
                }
            )
        return SymbolicEntityEncoder().encode(
            {
                "sideGuid": "red-side",
                "UnitList": units,
                "radiationAndDataLinkLine": {"WeaponTarget": links},
            }
        )

    @staticmethod
    def _record_attack(env):
        env.engagement_state.record_successful_attack(
            attacker_id="red-ship",
            target_id="blue-contact",
            started_frame=0,
            target_is_missile=False,
            target_aliases=("blue-stable",),
            attack_quantity=1,
        )

    def test_maddpg_uses_one_hundred_frame_final_fallback(self):
        env = SimulatedEnv()
        self.assertEqual(100, ATTACK_SLOT_TIMEOUT_FRAMES)
        self.assertEqual(100, env.engagement_state.timeout_frames)
        self._record_attack(env)

        env.engagement_state.update_from_situation(
            self._situation(with_weapon=True),
            current_frame=99,
        )
        self.assertTrue(
            env.engagement_state.is_attacking("red-ship", "blue-contact")
        )

        env.engagement_state.update_from_situation(
            self._situation(with_weapon=True),
            current_frame=100,
        )
        self.assertFalse(
            env.engagement_state.is_attacking("red-ship", "blue-contact")
        )

    def test_maddpg_releases_slot_when_last_in_flight_weapon_disappears(self):
        env = SimulatedEnv()
        self._record_attack(env)

        env.engagement_state.update_from_situation(
            self._situation(with_weapon=True),
            current_frame=1,
        )
        self.assertTrue(
            env.engagement_state.is_attacking("red-ship", "blue-contact")
        )

        env.engagement_state.update_from_situation(
            self._situation(with_weapon=False),
            current_frame=2,
        )
        self.assertFalse(
            env.engagement_state.is_attacking("red-ship", "blue-contact")
        )


if __name__ == "__main__":
    unittest.main()
