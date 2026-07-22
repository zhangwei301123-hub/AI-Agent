import unittest
from types import SimpleNamespace

from symbolic_reasoning.agent import (
    ACTION_DISABLED,
    ACTION_THRESHOLD,
    Conclusion,
    SONOBUOY_ACTOR,
    WAYPOINT_ACTOR,
)
from symbolic_reasoning.execute_actions import _execute_symbolic_rpc_actions
from symbolic_reasoning.symbolic_reasoning4test import (
    SymbolicReasoningEnv,
    log_step,
)


MISSION = {
    "mission-patrol-1": {
        "mission_name": "反潜巡逻",
        "mission_type": 2,
        "is_patrol": True,
        "area_points": [
            (120.0, 30.0),
            (121.0, 30.0),
            (121.0, 31.0),
            (120.0, 31.0),
        ],
    }
}


def _aircraft(lon=120.5, lat=30.5, altitude=300.0):
    return {
        "guid": "asw-aircraft-1",
        "SideId": "red-side",
        "forceSide": "红方",
        "name": "直-9C #1",
        "unitname": "直-9C",
        "unitType": 0,
        "unitCategory": 13,
        "longitude": lon,
        "latitude": lat,
        "altitude": altitude,
        "Speed": 100.0,
        "BloodAmount": 100.0,
        "buoyNum": 8,
    }


def _buoy(name, lon, lat, radius=7.408):
    return {
        "guid": "buoy-" + name,
        "name": name,
        "unitname": name,
        "longitude": lon,
        "latitude": lat,
        "altitude": 0.0,
        "BloodAmount": 100.0,
        "rangeSensor_UnderWater": [{"Radius": radius}],
    }


def _payload(*units):
    return {"sideGuid": "red-side", "UnitList": list(units)}


class BuoyStub:
    def __init__(self):
        self.request = None

    def delpoySonobuoyw(self, request, timeout):
        self.request = request
        return SimpleNamespace(code=0, error_message="")


class PatrolSonobuoyDeploymentTests(unittest.TestCase):
    def test_unique_patrol_mission_is_inferred_without_unit_mission_id(self):
        env = SymbolicReasoningEnv(max_entities=8, mission_areas=MISSION)

        result = env.step(_payload(_aircraft()), execute_commands=False)
        facts = result.facts["asw-aircraft-1"]

        self.assertEqual(facts.mission_id, "mission-patrol-1")
        self.assertTrue(facts.is_patrol_aircraft)
        self.assertTrue(facts.inside_patrol_area)

    def test_inside_area_deploys_only_active_shallow_buoy(self):
        env = SymbolicReasoningEnv(max_entities=8, mission_areas=MISSION)

        result = env.step(_payload(_aircraft()), execute_commands=False)
        decision = result.decisions["asw-aircraft-1"]

        self.assertEqual(decision.conclusion, Conclusion.DEPLOY_SONOBUOY)
        self.assertEqual(
            decision.actions[SONOBUOY_ACTOR],
            [ACTION_THRESHOLD, 1.0, 1.0, None, None],
        )
        self.assertEqual(
            decision.actions[WAYPOINT_ACTOR][0], ACTION_DISABLED
        )

    def test_outside_area_never_generates_route_or_buoy_action(self):
        env = SymbolicReasoningEnv(max_entities=8, mission_areas=MISSION)

        result = env.step(
            _payload(_aircraft(lon=122.0, lat=32.0)),
            execute_commands=False,
        )
        decision = result.decisions["asw-aircraft-1"]

        self.assertEqual(decision.conclusion, Conclusion.HOLD)
        self.assertEqual(decision.rule_id, "R-BUOY-001")
        self.assertNotIn(
            "R-BUOY-002",
            [step.rule_id for step in decision.inference_path],
        )
        self.assertEqual(decision.actions[SONOBUOY_ACTOR][0], ACTION_DISABLED)
        self.assertEqual(decision.actions[WAYPOINT_ACTOR][0], ACTION_DISABLED)

    def test_system_patrol_hold_is_not_printed_as_inference_path(self):
        env = SymbolicReasoningEnv(max_entities=8, mission_areas=MISSION)
        result = env.step(
            _payload(_aircraft(lon=122.0, lat=32.0)),
            execute_commands=False,
        )
        messages = []
        logger = SimpleNamespace(
            info=lambda *args: None,
            debug=lambda message, *args: messages.append(
                message % args if args else str(message)
            ),
        )

        log_step(result, 0, logger)

        self.assertFalse(any("R-BUOY-002" in item for item in messages))
        self.assertFalse(any("巡逻航路" in item for item in messages))

    def test_existing_active_buoy_does_not_block_inventory_based_deployment(self):
        env = SymbolicReasoningEnv(max_entities=8, mission_areas=MISSION)

        result = env.step(
            _payload(
                _aircraft(),
                _buoy("AN/SSQ-62B 主动声呐浮标", 120.5, 30.52),
            ),
            execute_commands=False,
        )
        decision = result.decisions["asw-aircraft-1"]

        self.assertEqual(decision.conclusion, Conclusion.DEPLOY_SONOBUOY)
        self.assertEqual(
            decision.actions[SONOBUOY_ACTOR],
            [ACTION_THRESHOLD, 1.0, 1.0, None, None],
        )

    def test_existing_passive_buoy_does_not_block_deployment(self):
        env = SymbolicReasoningEnv(max_entities=8, mission_areas=MISSION)

        result = env.step(
            _payload(
                _aircraft(),
                _buoy("AN/SSQ-53B 被动声呐浮标", 120.5, 30.5),
            ),
            execute_commands=False,
        )

        self.assertEqual(
            result.decisions["asw-aircraft-1"].conclusion,
            Conclusion.DEPLOY_SONOBUOY,
        )

    def test_executor_sends_active_and_shallow_flags(self):
        stub = BuoyStub()
        actions = [[ACTION_DISABLED, None, None, None, None] for _ in range(8)]
        actions[SONOBUOY_ACTOR] = [
            ACTION_THRESHOLD,
            1.0,
            1.0,
            None,
            None,
        ]

        result, _ = _execute_symbolic_rpc_actions(
            {"asw-aircraft-1": actions},
            enemy_ids=[],
            probability=0.7,
            logger=None,
            stub=stub,
        )

        self.assertTrue(result["asw-aircraft-1"][SONOBUOY_ACTOR])
        self.assertTrue(stub.request.passiveOrActive)
        self.assertTrue(stub.request.shallowOrDeep)

    def test_successful_deployment_does_not_block_next_inventory_based_decision(self):
        env = SymbolicReasoningEnv(max_entities=8, mission_areas=MISSION)

        env.step(
            _payload(_aircraft()),
            execute_commands=True,
            executor=lambda actions, *args, **kwargs: (
                {"asw-aircraft-1": [False] * 6 + [True, False]},
                {"asw-aircraft-1": [0.0] * 8},
            ),
        )
        second = env.step(_payload(_aircraft()), execute_commands=False)

        self.assertEqual(
            second.decisions["asw-aircraft-1"].conclusion,
            Conclusion.DEPLOY_SONOBUOY,
        )


if __name__ == "__main__":
    unittest.main()
