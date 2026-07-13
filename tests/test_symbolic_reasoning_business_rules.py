from __future__ import annotations

import unittest

from symbolic_reasoning import Conclusion, EngagementState, EntityEncoder
from symbolic_reasoning.symbolic_reasoning4test import SymbolicReasoningEnv


RED = "red-side"
BLUE = "blue-side"


def unit(
    guid,
    name,
    side_id,
    lon,
    lat,
    altitude,
    heading,
    icon,
    is_contact=False,
    is_weapon=False,
    contact_id=None,
    ranges=None,
    inventory=None,
    **extra
):
    ranges = ranges or {}
    value = {
        "guid": guid,
        "contactGuid": contact_id,
        "name": name,
        "unitname": name,
        "longitude": lon,
        "latitude": lat,
        "altitude": altitude,
        "heading": heading,
        "SideId": side_id,
        "unitType": 0,
        "unitCategory": 0,
        "UnitSpecificType": 0,
        "Icon2D": icon,
        "Speed": 500,
        "BloodAmount": 100,
        "IsContact": is_contact,
        "IsWeapon": is_weapon,
        "rangeStrike_Air": ranges.get("air", 0),
        "rangeStrike_Surface": ranges.get("surface", 0),
        "rangeStrike_Land": ranges.get("land", 0),
        "rangeStrike_Submarine": ranges.get("submarine", 0),
        "WeaponImpact": 0,
        "WeaponTargetDistance": 0,
        "WeaponTargetName": None,
        "CommText": "",
        "JammText": "",
    }
    if inventory is not None:
        value["weaponNumber"] = inventory
    value.update(extra)
    return value


def payload(units, current_time="2026-07-13T12:00:00Z", radiation=None):
    send_msg = {
        "ScenName": "symbolic-test",
        "CurrentTime": current_time,
        "UnitList": units,
        "DeleteUnitIdList": [],
    }
    if radiation is not None:
        send_msg["radiationAndDataLinkLine"] = radiation
    return {"data": {"sideGuid": RED, "data": send_msg}}


class SymbolicBusinessRuleTests(unittest.TestCase):
    def test_weapon_target_link_is_encoded(self):
        own = unit(
            "own-1",
            "我方飞机",
            RED,
            120.0,
            30.0,
            1000,
            90,
            "/ArmyIcon/Aircraft/a.svg",
        )
        missile = unit(
            "missile-physical",
            "敌方导弹",
            BLUE,
            120.03,
            30.0,
            1000,
            270,
            "/ArmyIcon/Weapon/missile.svg",
            is_contact=True,
            is_weapon=True,
            contact_id="missile-contact",
        )
        radiation = {
            "WeaponTarget": [
                {
                    "Arr": [
                        {"unitguid": "missile-physical"},
                        {"unitguid": "own-1"},
                    ]
                }
            ]
        }

        situation = EntityEncoder(max_entities=8).encode(
            payload([own, missile], radiation=radiation)
        )
        encoded_missile = situation.find_entity("missile-contact")

        self.assertIsNotNone(encoded_missile)
        self.assertEqual(encoded_missile.weapon_target_id, "own-1")

    def test_heading_fallback_triggers_fixed_right_90_degree_evasion(self):
        own = unit(
            "own-1",
            "我方飞机",
            RED,
            120.0,
            30.0,
            1000,
            90,
            "/ArmyIcon/Aircraft/a.svg",
        )
        # 导弹位于我方以东、向西飞行，无目标 ID/名称，距离约 3.85 km。
        missile = unit(
            "missile-physical",
            "敌方导弹",
            BLUE,
            120.04,
            30.0,
            1000,
            270,
            "/ArmyIcon/Weapon/missile.svg",
            is_contact=True,
            is_weapon=True,
            contact_id="missile-contact",
        )

        result = SymbolicReasoningEnv(max_entities=8).step(
            payload([own, missile]), execute_commands=False
        )
        decision = result.decisions["own-1"]

        self.assertEqual(decision.conclusion, Conclusion.EVADE_MISSILE)
        self.assertEqual(decision.actions[2][3:], [5, 4])
        # 导弹航向 270° 右转 90° 为 0°，规避点应位于我方以北。
        self.assertGreater(decision.actions[2][2], 30.0)

    def test_nearest_in_range_target_and_attack_altitude_are_selected(self):
        own = unit(
            "own-1",
            "我方飞机",
            RED,
            120.0,
            30.0,
            2000,
            90,
            "/ArmyIcon/Aircraft/a.svg",
            ranges={"air": 100, "surface": 100, "submarine": 100},
            inventory={"airNum": 2, "shipNum": 2, "subNum": 2},
        )
        farther_air = unit(
            "air-physical",
            "敌机",
            BLUE,
            120.02,
            30.0,
            4000,
            270,
            "/ArmyIcon/Aircraft/a.svg",
            is_contact=True,
            contact_id="air-contact",
        )
        nearer_ship = unit(
            "ship-physical",
            "敌舰",
            BLUE,
            120.01,
            30.0,
            0,
            270,
            "/ArmyIcon/Ship/s.svg",
            is_contact=True,
            contact_id="ship-contact",
        )

        result = SymbolicReasoningEnv(max_entities=8).step(
            payload([own, farther_air, nearer_ship]), execute_commands=False
        )
        facts = result.facts["own-1"]
        decision = result.decisions["own-1"]

        self.assertEqual(facts.target_id, "ship-contact")
        self.assertEqual(facts.attack_altitude_level, 1)
        self.assertEqual(facts.expected_weapon_type, "ANTI_SHIP_MISSILE")
        self.assertEqual(decision.conclusion, Conclusion.REQUEST_ATTACK)

    def test_5000m_air_target_height_boundary_maps_to_levels_3_and_5(self):
        own = unit(
            "own-1",
            "我方飞机",
            RED,
            120.0,
            30.0,
            3000,
            90,
            "/ArmyIcon/Aircraft/a.svg",
            ranges={"air": 100},
            inventory={"airNum": 2},
        )
        target_at_boundary = unit(
            "air-physical",
            "敌机",
            BLUE,
            120.01,
            30.0,
            5000.0,
            270,
            "/ArmyIcon/Aircraft/a.svg",
            is_contact=True,
            contact_id="air-contact",
        )
        target_above_boundary = dict(target_at_boundary)
        target_above_boundary["altitude"] = 5000.01

        at_boundary = SymbolicReasoningEnv(max_entities=4).step(
            payload([own, target_at_boundary]), execute_commands=False
        )
        above_boundary = SymbolicReasoningEnv(max_entities=4).step(
            payload([own, target_above_boundary]), execute_commands=False
        )

        self.assertEqual(at_boundary.facts["own-1"].attack_altitude_level, 3)
        self.assertEqual(above_boundary.facts["own-1"].attack_altitude_level, 5)

    def test_same_frame_allows_at_most_three_attackers_for_one_target(self):
        own_units = [
            unit(
                "own-{}".format(index),
                "我方飞机{}".format(index),
                RED,
                120.0,
                30.0,
                1000,
                90,
                "/ArmyIcon/Aircraft/a.svg",
                ranges={"air": 100},
                inventory={"airNum": 2},
            )
            for index in range(1, 5)
        ]
        target = unit(
            "target-physical",
            "敌机",
            BLUE,
            120.01,
            30.0,
            1000,
            270,
            "/ArmyIcon/Aircraft/a.svg",
            is_contact=True,
            contact_id="target-contact",
        )

        result = SymbolicReasoningEnv(max_entities=8).step(
            payload(own_units + [target]), execute_commands=False
        )
        attacks = [
            decision
            for decision in result.decisions.values()
            if decision.conclusion is Conclusion.REQUEST_ATTACK
        ]

        self.assertEqual(len(attacks), 3)
        self.assertEqual(
            result.decisions["own-4"].conclusion,
            Conclusion.HOLD,
        )

    def test_patrol_area_boundary_and_500m_allow_sonobuoy(self):
        patrol_aircraft = unit(
            "patrol-1",
            "海上巡逻机",
            RED,
            120.0,
            30.0,
            500.0,
            90,
            "/ArmyIcon/Aircraft/patrol.svg",
            inventory={"buoyNum": 2},
            missionId="mission-patrol",
            missionType="巡逻任务",
        )
        mission_areas = {
            "mission-patrol": {
                "is_patrol": True,
                "area_points": [
                    [120.0, 30.0],
                    [121.0, 30.0],
                    [121.0, 31.0],
                    [120.0, 31.0],
                ],
            }
        }

        result = SymbolicReasoningEnv(
            max_entities=8, mission_areas=mission_areas
        ).step(payload([patrol_aircraft]), execute_commands=False)

        self.assertEqual(
            result.decisions["patrol-1"].conclusion,
            Conclusion.DEPLOY_SONOBUOY,
        )


class EngagementStateTests(unittest.TestCase):
    class _Target:
        def __init__(
            self,
            target_id,
            impact=0,
            is_own=False,
            is_weapon=True,
            weapon_target_id=None,
        ):
            self.command_id = target_id
            self.entity_id = target_id
            self.is_own = is_own
            self.is_weapon = is_weapon
            self.weapon_impact = impact
            self.weapon_target_id = weapon_target_id

    class _Situation:
        def __init__(self, targets=(), deleted=(), entities=None):
            self.targets = targets
            self.deleted_entity_ids = deleted
            self.entities = tuple(targets if entities is None else entities)

        def find_entity(self, entity_id):
            for entity in self.entities:
                if entity_id in (entity.entity_id, entity.command_id):
                    return entity
            return None

    def test_attack_slots_timeout_and_interceptor_count_is_cumulative(self):
        state = EngagementState()
        target_id = "missile-target"
        for index in range(1, 4):
            state.record_successful_attack(
                "attacker-{}".format(index), target_id, 0.0, True
            )

        self.assertEqual(state.active_attackers(target_id), 3)
        self.assertEqual(state.interceptors_launched(target_id), 3)
        self.assertFalse(state.slot_available("attacker-4", target_id))

        # 目标仍在态势中；满 10 分钟只释放并发槽位，不清空累计发射数。
        state.update_from_situation(
            self._Situation((self._Target(target_id),)), 600.0
        )
        self.assertEqual(state.active_attackers(target_id), 0)
        self.assertEqual(state.interceptors_launched(target_id), 3)

        state.record_successful_attack("attacker-4", target_id, 600.0, True)
        self.assertEqual(state.interceptors_launched(target_id), 4)
        self.assertFalse(state.interceptor_available(target_id))

    def test_kinetic_hit_feedback_releases_target_slots(self):
        state = EngagementState()
        target_id = "target-1"
        state.record_successful_attack("attacker-1", target_id, 0.0, False)
        target = self._Target(target_id, is_weapon=False)
        our_weapon = self._Target(
            "our-missile",
            impact=2,
            is_own=True,
            is_weapon=True,
            weapon_target_id=target_id,
        )

        state.update_from_situation(
            self._Situation((target,), entities=(target, our_weapon)), 10.0
        )

        self.assertEqual(state.active_attackers(target_id), 0)


if __name__ == "__main__":
    unittest.main()
