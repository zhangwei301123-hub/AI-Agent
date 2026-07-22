import unittest
from types import SimpleNamespace

from symbolic_reasoning.agent import (
    ACTION_DISABLED,
    ACTION_THRESHOLD,
    Conclusion,
    ReasoningFacts,
    SENSOR_ACTOR,
    WAYPOINT_ACTOR,
    SymbolicReasoningAgent,
)
from symbolic_reasoning.entity import EntityEncoder, FEATURE_NAMES, TargetDomain
from symbolic_reasoning.live import inventory_from_unit_data
from symbolic_reasoning.symbolic_reasoning4test import SymbolicReasoningEnv


def _sensor(sensor_type, name="sensor", active=False):
    return SimpleNamespace(
        sensor_type=sensor_type,
        sensor_name=name,
        active_status=active,
    )


def _facts(**changes):
    values = {
        "entity_id": "red-1",
        "own_platform_type": TargetDomain.SURFACE,
        "target_id": None,
        "detected_target_count": 0,
    }
    values.update(changes)
    return ReasoningFacts(**values)


class SensorCapabilityTests(unittest.TestCase):
    def test_get_unit_data_uses_inventory_not_active_state(self):
        response = SimpleNamespace(
            unir_sensor_params=SimpleNamespace(
                unir_sensor_params=[
                    _sensor("雷达, 对空搜索", active=False),
                    _sensor("船体声纳, 主动/被动搜索", active=False),
                ]
            ),
            unit_weapons=[],
            unit_current_status=SimpleNamespace(unit_name="052D"),
        )

        inventory = inventory_from_unit_data(response)

        self.assertTrue(inventory.has_radar_sensor)
        self.assertTrue(inventory.has_sonar_sensor)

    def test_entity_encoder_prefers_explicit_capability_flags(self):
        payload = {
            "sideGuid": "red-side",
            "UnitList": [
                {
                    "guid": "red-1",
                    "SideId": "red-side",
                    "forceSide": "红方",
                    "unitType": 1,
                    "unitCategory": 1,
                    "longitude": 120.0,
                    "latitude": 30.0,
                    "hasRadarSensor": True,
                    "hasSonarSensor": False,
                    # Explicit False must override a stale legacy range hint.
                    "rangeSensor_UnderWater": [{"range": 10}],
                }
            ],
        }

        situation = EntityEncoder(max_entities=4).encode(payload)
        entity = situation.entities[0]

        self.assertTrue(entity.has_radar_sensor)
        self.assertFalse(entity.has_sonar_sensor)
        self.assertEqual(entity.vector.shape[0], len(FEATURE_NAMES))

    def test_search_enables_radar_only(self):
        decision = SymbolicReasoningAgent().reason(
            _facts(radar_available=True, sonar_available=False)
        )

        self.assertEqual(decision.conclusion, Conclusion.SEARCH)
        self.assertEqual(
            decision.actions[SENSOR_ACTOR],
            [ACTION_THRESHOLD, 1.0, 0.0, 0.0, None],
        )

    def test_search_enables_sonar_only(self):
        decision = SymbolicReasoningAgent().reason(
            _facts(radar_available=False, sonar_available=True)
        )

        self.assertEqual(decision.conclusion, Conclusion.SEARCH)
        self.assertEqual(
            decision.actions[SENSOR_ACTOR],
            [ACTION_THRESHOLD, 0.0, 1.0, 0.0, None],
        )

    def test_search_without_supported_sensor_holds(self):
        decision = SymbolicReasoningAgent().reason(_facts())

        self.assertEqual(decision.conclusion, Conclusion.HOLD)
        self.assertEqual(decision.rule_id, "R-SEARCH-002")
        self.assertEqual(decision.actions[SENSOR_ACTOR][0], ACTION_DISABLED)

    def test_patrol_does_not_generate_route_or_sensor_action(self):
        decision = SymbolicReasoningAgent().reason(
            _facts(
                is_patrol_aircraft=True,
                has_patrol_mission=True,
                mission_id="patrol-1",
            )
        )

        self.assertEqual(decision.conclusion, Conclusion.HOLD)
        self.assertEqual(decision.rule_id, "R-BUOY-001")
        self.assertNotIn(
            "R-BUOY-002",
            [step.rule_id for step in decision.inference_path],
        )
        self.assertEqual(decision.actions[WAYPOINT_ACTOR][0], ACTION_DISABLED)
        self.assertEqual(decision.actions[SENSOR_ACTOR][0], ACTION_DISABLED)

    def test_end_to_end_facts_preserve_sensor_capabilities(self):
        payload = {
            "sideGuid": "red-side",
            "UnitList": [
                {
                    "guid": "red-1",
                    "SideId": "red-side",
                    "forceSide": "红方",
                    "unitType": 1,
                    "unitCategory": 1,
                    "longitude": 120.0,
                    "latitude": 30.0,
                    "hasRadarSensor": True,
                    "hasSonarSensor": False,
                }
            ],
        }

        result = SymbolicReasoningEnv(max_entities=4).step(
            payload, execute_commands=False
        )

        facts = result.facts["red-1"]
        self.assertTrue(facts.radar_available)
        self.assertFalse(facts.sonar_available)
        self.assertEqual(result.decisions["red-1"].conclusion, Conclusion.SEARCH)


if __name__ == "__main__":
    unittest.main()
