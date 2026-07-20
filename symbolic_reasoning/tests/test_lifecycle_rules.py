import unittest
from types import SimpleNamespace

from symbolic_reasoning.agent import (
    ACTION_DISABLED,
    ACTION_THRESHOLD,
    CANCEL_ATTACK_ACTOR,
    RETURN_TO_BASE_ACTOR,
    TAKEOFF_ACTOR,
    Conclusion,
    ReasoningFacts,
    SymbolicReasoningAgent,
)
from symbolic_reasoning.live import inventory_from_unit_data
from symbolic_reasoning.symbolic_reasoning4test import SymbolicReasoningEnv


def _payload(**changes):
    unit = {
        "guid": "red-aircraft-1",
        "SideId": "red-side",
        "forceSide": "红方",
        "IsContact": False,
        "IsWeapon": False,
        "unitType": 0,
        "unitCategory": 0,
        "Icon2D": "/ArmyIcon/Aircraft/live.svg",
        "longitude": 120.0,
        "latitude": 30.0,
        "altitude": 1000.0,
        "Speed": 200.0,
        "BloodAmount": 100.0,
        "fuelPercentage": 80.0,
        "hasStrikeWeaponSystem": True,
        "strikeWeaponCount": 2,
        "weaponNumber": {"airNum": 2},
        "rangeStrike_Air": 100.0,
        "hasRadarSensor": True,
        "hasSonarSensor": False,
    }
    unit.update(changes)
    return {"sideGuid": "red-side", "UnitList": [unit]}


def _facts(**changes):
    values = {
        "entity_id": "red-aircraft-1",
        "target_id": None,
        "detected_target_count": 0,
        "attack_authorized": True,
        "safety_clearance": True,
        "is_aircraft": True,
        "radar_available": True,
    }
    values.update(changes)
    return ReasoningFacts(**values)


def _successful_executor(actor_index):
    def execute(actions_dict, enemy_ids, probability, logger):
        results = {}
        for entity_id in actions_dict:
            actor_results = [False] * 8
            actor_results[actor_index] = True
            results[entity_id] = actor_results
        return results, [0.0] * 8

    return execute


class LifecycleRuleTests(unittest.TestCase):
    def test_zero_inventory_preserves_installed_strike_weapon_capability(self):
        weapon = SimpleNamespace(
            num=0,
            weapon_type=2001,
            text="air-to-air missile",
            air_range_strike=50.0,
            surface_range_strike=0.0,
            land_range_strike=0.0,
            submarine_range_strike=0.0,
        )
        response = SimpleNamespace(
            unir_sensor_params=SimpleNamespace(unir_sensor_params=[]),
            unit_weapons=[weapon],
            unit_current_status=SimpleNamespace(
                unit_name="fighter",
                text_block_unit_status="停放",
                text_block_assigned_host="airbase-1",
            ),
            selected_unit_fuel=SimpleNamespace(percen_tage=75),
        )

        inventory = inventory_from_unit_data(response)

        self.assertTrue(inventory.has_strike_weapon_system)
        self.assertEqual(inventory.strike_weapon_count, 0)
        self.assertEqual(inventory.fuel_percentage, 75.0)
        self.assertEqual(inventory.unit_status, "停放")

    def test_takeoff_rule_generates_actor_zero(self):
        decision = SymbolicReasoningAgent().reason(
            _facts(is_parked=True, is_airborne=False)
        )

        self.assertEqual(decision.conclusion, Conclusion.TAKEOFF)
        self.assertEqual(decision.rule_id, "R-TAKEOFF-001")
        self.assertEqual(decision.actions[TAKEOFF_ACTOR][0], ACTION_THRESHOLD)

    def test_low_fuel_rule_generates_return_actor(self):
        decision = SymbolicReasoningAgent().reason(
            _facts(
                is_airborne=True,
                fuel_percentage=20.0,
                fuel_low=True,
            )
        )

        self.assertEqual(decision.conclusion, Conclusion.RETURN_TO_BASE)
        self.assertEqual(decision.rule_id, "R-RTB-001")
        self.assertEqual(
            decision.actions[RETURN_TO_BASE_ACTOR][0], ACTION_THRESHOLD
        )

    def test_depleted_ammunition_returns_only_installed_weapon_aircraft(self):
        agent = SymbolicReasoningAgent()
        armed = agent.reason(
            _facts(
                is_airborne=True,
                fuel_percentage=80.0,
                has_strike_weapon_system=True,
                strike_weapon_count=0,
                ammunition_low=True,
            )
        )
        unarmed_mission_aircraft = agent.reason(
            _facts(
                is_airborne=True,
                fuel_percentage=80.0,
                has_strike_weapon_system=False,
                strike_weapon_count=0,
                ammunition_low=False,
            )
        )

        self.assertEqual(armed.conclusion, Conclusion.RETURN_TO_BASE)
        self.assertNotEqual(
            unarmed_mission_aircraft.conclusion, Conclusion.RETURN_TO_BASE
        )

    def test_invalid_active_attack_generates_cancel_actor(self):
        decision = SymbolicReasoningAgent().reason(
            _facts(
                currently_attacking=True,
                current_attack_target_id="enemy-contact-1",
                attack_conditions_valid=False,
            )
        )

        self.assertEqual(decision.conclusion, Conclusion.CANCEL_ATTACK)
        self.assertEqual(decision.rule_id, "R-CANCEL-001")
        self.assertEqual(
            decision.actions[CANCEL_ATTACK_ACTOR][0], ACTION_THRESHOLD
        )

    def test_valid_active_attack_does_not_cancel(self):
        decision = SymbolicReasoningAgent().reason(
            _facts(
                currently_attacking=True,
                current_attack_target_id="enemy-contact-1",
                attack_conditions_valid=True,
            )
        )

        self.assertEqual(decision.conclusion, Conclusion.HOLD)
        self.assertEqual(decision.rule_id, "R-CANCEL-002")
        self.assertEqual(
            decision.actions[CANCEL_ATTACK_ACTOR][0], ACTION_DISABLED
        )

    def test_end_to_end_parked_aircraft_takeoff_and_pending_guard(self):
        env = SymbolicReasoningEnv(max_entities=4)
        payload = _payload(
            altitude=0.0,
            Speed=0.0,
            airStatus="停放",
        )

        first = env.step(
            payload,
            execute_commands=True,
            executor=_successful_executor(TAKEOFF_ACTOR),
        )
        second = env.step(payload, execute_commands=False)

        self.assertEqual(
            first.decisions["red-aircraft-1"].conclusion,
            Conclusion.TAKEOFF,
        )
        self.assertEqual(first.execution_status["red-aircraft-1"], "SUCCESS")
        self.assertEqual(
            second.decisions["red-aircraft-1"].rule_id,
            "R-TAKEOFF-002",
        )

    def test_end_to_end_low_fuel_aircraft_returns(self):
        result = SymbolicReasoningEnv(max_entities=4).step(
            _payload(fuelPercentage=20.0), execute_commands=False
        )

        facts = result.facts["red-aircraft-1"]
        self.assertTrue(facts.is_airborne)
        self.assertTrue(facts.fuel_low)
        self.assertEqual(
            result.decisions["red-aircraft-1"].conclusion,
            Conclusion.RETURN_TO_BASE,
        )

    def test_end_to_end_lost_target_cancels_and_releases_attack_slot(self):
        env = SymbolicReasoningEnv(max_entities=4)
        env.engagement_state.record_successful_attack(
            attacker_id="red-aircraft-1",
            target_id="enemy-contact-1",
            started_frame=0,
            target_is_missile=False,
        )

        first = env.step(_payload(), execute_commands=False)
        second = env.step(_payload(), execute_commands=False)
        result = env.step(
            _payload(),
            execute_commands=True,
            executor=_successful_executor(CANCEL_ATTACK_ACTOR),
        )

        self.assertEqual(
            first.decisions["red-aircraft-1"].rule_id, "R-CANCEL-002"
        )
        self.assertEqual(
            second.decisions["red-aircraft-1"].rule_id, "R-CANCEL-002"
        )
        self.assertEqual(
            result.decisions["red-aircraft-1"].conclusion,
            Conclusion.CANCEL_ATTACK,
        )
        self.assertIsNone(
            env.engagement_state.attack_target_for("red-aircraft-1")
        )

    def test_last_weapon_in_flight_is_not_immediately_cancelled(self):
        env = SymbolicReasoningEnv(max_entities=4)
        env.engagement_state.record_successful_attack(
            attacker_id="red-aircraft-1",
            target_id="enemy-contact-1",
            started_frame=0,
            target_is_missile=False,
        )
        payload = _payload(
            hasStrikeWeaponSystem=True,
            strikeWeaponCount=0,
            weaponNumber={"airNum": 0},
        )
        payload["UnitList"].append(
            {
                "guid": "enemy-entity-1",
                "contactGuid": "enemy-contact-1",
                "SideId": "blue-side",
                "forceSide": "蓝方",
                "IsContact": True,
                "IsWeapon": False,
                "unitType": 0,
                "unitCategory": 0,
                "Icon2D": "/ArmyIcon/Aircraft/live.svg",
                "longitude": 120.1,
                "latitude": 30.1,
                "altitude": 1000.0,
                "BloodAmount": 100.0,
            }
        )

        result = env.step(payload, execute_commands=False)

        facts = result.facts["red-aircraft-1"]
        self.assertTrue(facts.ammunition_low)
        self.assertTrue(facts.attack_conditions_valid)
        self.assertEqual(
            result.decisions["red-aircraft-1"].rule_id,
            "R-CANCEL-002",
        )


if __name__ == "__main__":
    unittest.main()
