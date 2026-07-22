import unittest

from symbolic_reasoning.agent import (
    ACTION_DISABLED,
    ACTION_THRESHOLD,
    ASW_TORPEDO_RELEASE_DISTANCE_KM,
    ATTACK_ACTOR,
    WAYPOINT_ACTOR,
    Conclusion,
    ReasoningFacts,
    SymbolicReasoningAgent,
)
from symbolic_reasoning.entity import TargetDomain
from symbolic_reasoning.symbolic_reasoning4test import SymbolicReasoningEnv


def _payload(target_longitude, heading):
    return {
        "sideGuid": "red-side",
        "UnitList": [
            {
                "guid": "red-asw-1",
                "name": "直-8J #1",
                "SideId": "red-side",
                "forceSide": "红方",
                "IsContact": False,
                "IsWeapon": False,
                "unitType": 0,
                "unitCategory": 0,
                "UnitSpecificType": 7401,
                "Icon2D": "/ArmyIcon/Aircraft/live.svg",
                "longitude": 120.0,
                "latitude": 30.0,
                "altitude": 100.0,
                "heading": heading,
                "Speed": 60.0,
                "BloodAmount": 100.0,
                "fuelPercentage": 80.0,
                "weaponNumber": {"subNum": 2},
                "rangeStrike_Submarine": 20.0,
                "hasStrikeWeaponSystem": True,
                "strikeWeaponCount": 2,
            },
            {
                "guid": "blue-submarine-1",
                "contactGuid": "blue-sub-contact-1",
                "name": "敌方潜艇 #1",
                "SideId": "blue-side",
                "forceSide": "蓝方",
                "IsContact": True,
                "IsWeapon": False,
                "unitType": 2,
                "unitCategory": 2,
                "Icon2D": "/ArmyIcon/Submarine/live.svg",
                "longitude": target_longitude,
                "latitude": 30.0,
                "altitude": -20.0,
                "heading": 180.0,
                "Speed": 5.0,
                "BloodAmount": 0.0,
            },
        ],
    }


class AswTorpedoRuleTests(unittest.TestCase):
    def test_outside_point_four_nm_chases_without_attack(self):
        result = SymbolicReasoningEnv(max_entities=4).step(
            _payload(target_longitude=120.01, heading=90.0),
            execute_commands=False,
        )
        decision = result.decisions["red-asw-1"]

        self.assertEqual(decision.conclusion, Conclusion.CHASE_TO_RANGE)
        self.assertEqual(decision.rule_id, "R-ASW-001")
        self.assertEqual(decision.actions[ATTACK_ACTOR][0], ACTION_DISABLED)
        self.assertEqual(decision.actions[WAYPOINT_ACTOR][0], ACTION_THRESHOLD)

    def test_inside_point_four_nm_allows_torpedo_attack(self):
        result = SymbolicReasoningEnv(max_entities=4).step(
            _payload(target_longitude=120.0, heading=0.0),
            execute_commands=False,
        )
        decision = result.decisions["red-asw-1"]

        self.assertEqual(decision.conclusion, Conclusion.REQUEST_ATTACK)
        self.assertEqual(decision.rule_id, "R-WPN-003")
        self.assertEqual(decision.actions[ATTACK_ACTOR][0], ACTION_THRESHOLD)
        self.assertTrue(
            any(
                step.rule_id == "R-ASW-001" and step.matched
                for step in decision.inference_path
            )
        )

    def test_exact_point_four_nm_is_inside_release_boundary(self):
        facts = ReasoningFacts(
            entity_id="red-asw-1",
            own_platform_type=TargetDomain.AIR,
            is_asw_aircraft=True,
            target_id="blue-sub-contact-1",
            target_domain=TargetDomain.SUBMARINE,
            detected_target_count=1,
            attack_authorized=True,
            target_type_allowed=True,
            weapon_available=True,
            compatible_weapon_count=2,
            expected_weapon_type="ANTI_SUBMARINE_WEAPON_OR_TORPEDO",
            within_attack_range=True,
            distance_km=ASW_TORPEDO_RELEASE_DISTANCE_KM,
            max_attack_range_km=ASW_TORPEDO_RELEASE_DISTANCE_KM,
            aimed_at_target=True,
            heading_difference_deg=0.0,
            safety_clearance=True,
            chase_allowed=True,
            concurrency_slot_available=True,
            target_lon=120.0,
            target_lat=30.0,
            attack_altitude_level=1,
        )

        decision = SymbolicReasoningAgent().reason(facts)

        self.assertEqual(decision.conclusion, Conclusion.REQUEST_ATTACK)
        self.assertEqual(decision.rule_id, "R-WPN-003")


if __name__ == "__main__":
    unittest.main()
