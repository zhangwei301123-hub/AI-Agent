import importlib
import unittest
from types import SimpleNamespace

from symbolic_reasoning.execute_actions import (
    execute_attack_pipeline,
    precheck_weapon_fire,
)
from symbolic_reasoning.control import FrontendControl
from symbolic_reasoning.symbolic_reasoning4test import (
    SymbolicReasoningEnv,
    log_step,
)


actions_module = importlib.import_module("symbolic_reasoning.execute_actions")


class CapturingLogger:
    def __init__(self):
        self.debug_messages = []
        self.info_messages = []
        self.warning_messages = []

    @staticmethod
    def _format(message, args):
        return message % args if args else str(message)

    def debug(self, message, *args):
        self.debug_messages.append(self._format(message, args))

    def info(self, message, *args):
        self.info_messages.append(self._format(message, args))

    def warning(self, message, *args):
        self.warning_messages.append(self._format(message, args))


class FiringStub:
    def __init__(self, can_fire=True, error=None):
        self.can_fire = can_fire
        self.error = error
        self.attack_calls = 0

    def GetWeaponFiringInfo(self, request, timeout):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            suitable_weapons=[
                SimpleNamespace(
                    weapon_db_id=2001,
                    weapon_name="HHQ-9 防空导弹",
                    total_quantity=4,
                    fire_evaluations=[
                        SimpleNamespace(
                            quantity=2,
                            can_fire=self.can_fire,
                            evaluation=("目标条件不满足" if not self.can_fire else ""),
                        )
                    ],
                    auto_fire_denied_reason="",
                )
            ]
        )

    def AttackTarget(self, request, timeout):
        self.attack_calls += 1
        return SimpleNamespace()


class SensorStub:
    def controlUnitSensorw(self, request, timeout):
        return SimpleNamespace(code=0, error_message="")


def _hold_payload():
    return {
        "sideGuid": "red-side",
        "UnitList": [
            {
                "guid": "quiet-red-1",
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
            }
        ],
    }


class LoggingPolicyTests(unittest.TestCase):
    def setUp(self):
        actions_module._MISSING_CONTACT_WARNED.clear()
        actions_module._ATTACK_RPC_WARNED.clear()
        actions_module._ACTION_FAILURE_WARNED.clear()
        actions_module._MISSING_ATTACK_TARGET_WARNED.clear()
        actions_module._LAST_POSITIVE_ACTION_LOGS.clear()
        actions_module._LAST_SENSOR_ACTIONS.clear()

    def test_unmatched_reasoning_is_debug_not_info(self):
        logger = CapturingLogger()
        result = SymbolicReasoningEnv(max_entities=4).step(
            _hold_payload(), execute_commands=False
        )

        log_step(result, 0, logger)

        self.assertEqual(logger.info_messages, [])
        self.assertEqual(logger.warning_messages, [])
        self.assertTrue(any("推理路径" in item for item in logger.debug_messages))
        self.assertFalse(any("未命中" in item for item in logger.debug_messages))

    def test_ui_state_transition_remains_info(self):
        logger = CapturingLogger()
        control = FrontendControl(
            signal_provider=lambda: "running", logger=logger
        )

        control.handle_signal("running")

        self.assertEqual(
            logger.info_messages, ["[UI控制] waiting -> running"]
        )

    def test_expected_weapon_rejection_is_not_warning(self):
        logger = CapturingLogger()
        result = execute_attack_pipeline(
            attacker_id="quiet-attacker-1",
            target_id="quiet-target-1",
            logger=logger,
            stub=FiringStub(can_fire=False),
        )

        self.assertFalse(result.success)
        self.assertEqual(logger.info_messages, [])
        self.assertEqual(logger.warning_messages, [])
        self.assertTrue(any("本帧未发射" in item for item in logger.debug_messages))

    def test_successful_weapon_launch_is_info(self):
        logger = CapturingLogger()
        stub = FiringStub(can_fire=True)

        result = execute_attack_pipeline(
            attacker_id="quiet-attacker-2",
            target_id="quiet-target-2",
            logger=logger,
            stub=stub,
        )

        self.assertTrue(result.success)
        self.assertEqual(stub.attack_calls, 1)
        self.assertEqual(len(logger.info_messages), 1)
        self.assertIn("[武器发射] 成功", logger.info_messages[0])
        self.assertEqual(logger.warning_messages, [])

    def test_successful_sensor_switch_has_no_log(self):
        logger = CapturingLogger()
        actions = [[0.01, None, None, None, None] for _ in range(8)]
        actions[5] = [0.9, 1.0, 0.0, 0.0, None]

        actions_module._execute_symbolic_rpc_actions(
            {"quiet-sensor-1": actions},
            enemy_ids=[],
            probability=0.7,
            logger=logger,
            stub=SensorStub(),
        )

        self.assertEqual(logger.info_messages, [])
        self.assertEqual(logger.warning_messages, [])

    def test_repeated_rpc_error_warns_only_once(self):
        logger = CapturingLogger()
        stub = FiringStub(error=RuntimeError("rpc unavailable"))

        for _ in range(2):
            precheck_weapon_fire(
                attacker_id="quiet-attacker-3",
                target_id="quiet-target-3",
                logger=logger,
                stub=stub,
            )

        self.assertEqual(len(logger.warning_messages), 1)
        self.assertTrue(any("RPC 失败" in item for item in logger.warning_messages))


if __name__ == "__main__":
    unittest.main()
