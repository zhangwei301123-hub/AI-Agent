import unittest
from types import SimpleNamespace

from symbolic_reasoning.agent import Conclusion
from symbolic_reasoning.execute_actions import (
    WeaponFirePrecheck,
    execute_attack_pipeline,
    precheck_weapon_fire,
)
from symbolic_reasoning.entity import EntityEncoder
from symbolic_reasoning.state import EngagementState
from symbolic_reasoning.symbolic_reasoning4test import SymbolicReasoningEnv


def _payload(contact_id=None, last_detect="1秒", uncertainty=None):
    units = [
        {
            "guid": "red-ship-1",
            "SideId": "red-side",
            "forceSide": "红方",
            "IsContact": False,
            "IsWeapon": False,
            "unitType": 1,
            "unitCategory": 1,
            "Icon2D": "/ArmyIcon/Ship/red.svg",
            "longitude": 120.0,
            "latitude": 30.0,
            "altitude": 0.0,
            "heading": 0.0,
            "Speed": 10.0,
            "BloodAmount": 100.0,
            "weaponNumber": {"shipNum": 4},
            "rangeStrike_Surface": 100.0,
            "hasStrikeWeaponSystem": True,
            "strikeWeaponCount": 4,
        }
    ]
    if contact_id is not None:
        units.append(
            {
                "guid": "blue-stable-1",
                "contactGuid": contact_id,
                "SideId": "blue-side",
                "forceSide": "蓝方",
                "IsContact": True,
                "IsWeapon": False,
                "unitType": 1,
                "unitCategory": 1,
                "Icon2D": "/ArmyIcon/Ship/blue.svg",
                "longitude": 120.01,
                "latitude": 30.0,
                "altitude": 0.0,
                "heading": 180.0,
                "Speed": 10.0,
                "BloodAmount": 0.0,
                "ContactLastDetectTimeStr": last_detect,
                "UncertainAreaList": uncertainty,
            }
        )
    return {"sideGuid": "red-side", "UnitList": units}


def _multi_ship_salvo_payload(weapon_owners):
    """构造三舰对同一目标齐射的态势；武器位置靠近对应发射舰。"""

    ship_longitudes = {
        "red-ship-a": 120.0,
        "red-ship-b": 120.2,
        "red-ship-c": 120.4,
    }
    units = []
    for ship_id, longitude in ship_longitudes.items():
        units.append(
            {
                "guid": ship_id,
                "SideId": "red-side",
                "forceSide": "红方",
                "IsContact": False,
                "IsWeapon": False,
                "unitType": 1,
                "unitCategory": 1,
                "Icon2D": "/ArmyIcon/Ship/red.svg",
                "longitude": longitude,
                "latitude": 30.0,
                "altitude": 0.0,
                "BloodAmount": 100.0,
            }
        )
    units.append(
        {
            "guid": "blue-target",
            "contactGuid": "blue-contact",
            "SideId": "blue-side",
            "forceSide": "蓝方",
            "IsContact": True,
            "IsWeapon": False,
            "unitType": 1,
            "unitCategory": 1,
            "Icon2D": "/ArmyIcon/Ship/blue.svg",
            "longitude": 121.0,
            "latitude": 30.0,
            "altitude": 0.0,
            "BloodAmount": 100.0,
        }
    )

    links = []
    for owner_id in weapon_owners:
        for salvo_index in range(2):
            weapon_id = "weapon-{}-{}".format(owner_id, salvo_index)
            units.append(
                {
                    "guid": weapon_id,
                    "SideId": "red-side",
                    "forceSide": "红方",
                    "IsContact": False,
                    "IsWeapon": True,
                    "unitType": 0,
                    "unitCategory": 0,
                    "Icon2D": "/ArmyIcon/Weapon/missile.svg",
                    "longitude": ship_longitudes[owner_id]
                    + 0.001 * (salvo_index + 1),
                    "latitude": 30.0,
                    "altitude": 50.0,
                    "BloodAmount": 100.0,
                }
            )
            links.append(
                {
                    "Arr": [
                        {"unitguid": weapon_id},
                        {"unitguid": "blue-target"},
                    ]
                }
            )
    return {
        "sideGuid": "red-side",
        "UnitList": units,
        "radiationAndDataLinkLine": {"WeaponTarget": links},
    }


class _FiringInfoStub:
    def __init__(self, can_fire):
        self.can_fire = can_fire
        self.calls = 0
        self.attack_calls = 0

    def GetWeaponFiringInfo(self, request, timeout):
        self.calls += 1
        evaluation = SimpleNamespace(
            quantity=2,
            can_fire=self.can_fire,
            evaluation=(
                "cross-range ambiguity 不满足武器要求"
                if not self.can_fire
                else ""
            ),
        )
        weapon = SimpleNamespace(
            weapon_db_id=2001,
            weapon_name="YJ-18",
            total_quantity=2,
            fire_evaluations=[evaluation],
            auto_fire_denied_reason="",
        )
        return SimpleNamespace(suitable_weapons=[weapon])

    def AttackTarget(self, request, timeout):
        self.attack_calls += 1
        return SimpleNamespace()


class AttackStabilityTests(unittest.TestCase):
    def test_intercepted_salvos_release_only_their_ship_slots(self):
        state = EngagementState(weapon_appearance_grace_frames=10)
        for attacker_id in (
            "red-ship-a",
            "red-ship-b",
            "red-ship-c",
        ):
            state.record_successful_attack(
                attacker_id=attacker_id,
                target_id="blue-contact",
                target_aliases=("blue-target",),
                started_frame=0,
                target_is_missile=False,
                attack_quantity=2,
            )

        encoder = EntityEncoder(max_entities=16)
        all_six_in_flight = encoder.encode(
            _multi_ship_salvo_payload(
                ("red-ship-a", "red-ship-b", "red-ship-c")
            )
        )
        state.update_from_situation(all_six_in_flight, current_frame=1)
        self.assertEqual(state.active_attackers("blue-contact"), 3)

        # A、B 的四发被拦截，只剩 C 的两发在途：A、B 应立即腾出两个槽位。
        only_c_in_flight = encoder.encode(
            _multi_ship_salvo_payload(("red-ship-c",))
        )
        state.update_from_situation(only_c_in_flight, current_frame=2)

        self.assertEqual(
            state.attacker_ids("blue-contact"), {"red-ship-c"}
        )
        self.assertEqual(state.active_attackers("blue-contact"), 1)
        self.assertTrue(
            state.slot_available("red-ship-d", "blue-contact")
        )

    def test_contact_quality_signature_ignores_sub_meter_float_jitter(self):
        encoder = EntityEncoder(max_entities=4)
        first = encoder.encode(
            _payload(
                "contact-1",
                uncertainty=[{"longitude": 120.12340001}],
            )
        ).targets[0]
        jittered = encoder.encode(
            _payload(
                "contact-1",
                uncertainty=[{"longitude": 120.12340002}],
            )
        ).targets[0]
        changed = encoder.encode(
            _payload(
                "contact-1",
                uncertainty=[{"longitude": 120.124}],
            )
        ).targets[0]

        self.assertEqual(
            first.contact_quality_signature,
            jittered.contact_quality_signature,
        )
        self.assertNotEqual(
            first.contact_quality_signature,
            changed.contact_quality_signature,
        )

    def test_transient_contact_loss_requires_three_consecutive_frames(self):
        state = EngagementState(target_contact_loss_grace_frames=3)
        env = SymbolicReasoningEnv(max_entities=4, engagement_state=state)
        state.record_successful_attack(
            attacker_id="red-ship-1",
            target_id="contact-old",
            started_frame=0,
            target_is_missile=False,
            target_aliases=("blue-stable-1",),
        )

        first = env.step(_payload(), execute_commands=False)
        second = env.step(_payload(), execute_commands=False)
        third = env.step(_payload(), execute_commands=False)

        self.assertEqual(
            first.decisions["red-ship-1"].rule_id, "R-CANCEL-002"
        )
        self.assertEqual(first.facts["red-ship-1"].attack_target_missing_frames, 1)
        self.assertEqual(
            second.decisions["red-ship-1"].rule_id, "R-CANCEL-002"
        )
        self.assertEqual(second.facts["red-ship-1"].attack_target_missing_frames, 2)
        self.assertEqual(
            third.decisions["red-ship-1"].conclusion,
            Conclusion.CANCEL_ATTACK,
        )
        self.assertEqual(third.facts["red-ship-1"].attack_target_missing_frames, 3)

    def test_reacquired_contact_with_new_id_resets_loss_counter(self):
        state = EngagementState(target_contact_loss_grace_frames=3)
        env = SymbolicReasoningEnv(max_entities=4, engagement_state=state)
        state.record_successful_attack(
            attacker_id="red-ship-1",
            target_id="contact-old",
            started_frame=0,
            target_is_missile=False,
            target_aliases=("blue-stable-1",),
        )

        env.step(_payload(), execute_commands=False)
        reacquired = env.step(
            _payload(contact_id="contact-new"), execute_commands=False
        )

        facts = reacquired.facts["red-ship-1"]
        self.assertEqual(facts.attack_target_missing_frames, 0)
        self.assertTrue(facts.attack_conditions_valid)
        self.assertEqual(
            reacquired.decisions["red-ship-1"].rule_id, "R-CANCEL-002"
        )
        self.assertTrue(state.same_target("contact-old", "contact-new"))

    def test_fire_rejection_holds_and_suppresses_repeated_prechecks(self):
        calls = []

        def reject(**kwargs):
            calls.append((kwargs["attacker_id"], kwargs["target_id"]))
            return WeaponFirePrecheck(
                can_fire=False,
                attacker_id=kwargs["attacker_id"],
                target_id=kwargs["target_id"],
                reason="cross-range ambiguity 不满足武器要求",
                reason_key="NO_READY_WEAPON:CROSS_RANGE_AMBIGUITY",
            )

        state = EngagementState(fire_control_rejection_cooldown_frames=3)
        env = SymbolicReasoningEnv(
            max_entities=4,
            engagement_state=state,
            weapon_fire_prechecker=reject,
        )

        results = [
            env.step(_payload("contact-1"), execute_commands=False)
            for _ in range(3)
        ]
        retried = env.step(_payload("contact-1"), execute_commands=False)

        self.assertEqual(len(calls), 2)
        for result in results:
            self.assertEqual(
                result.decisions["red-ship-1"].rule_id, "R-FIRE-001"
            )
            self.assertEqual(
                result.decisions["red-ship-1"].conclusion, Conclusion.HOLD
            )
        self.assertTrue(results[1].facts["red-ship-1"].fire_control_cooldown)
        self.assertEqual(
            retried.decisions["red-ship-1"].rule_id, "R-FIRE-001"
        )

    def test_contact_quality_change_retries_before_cooldown_expires(self):
        calls = []

        def evaluate(**kwargs):
            calls.append(kwargs["target_id"])
            allowed = kwargs["target_id"] == "contact-2"
            return WeaponFirePrecheck(
                can_fire=allowed,
                attacker_id=kwargs["attacker_id"],
                target_id=kwargs["target_id"],
                weapon_db_id=2001 if allowed else None,
                weapon_name="YJ-18" if allowed else "",
                ready_quantity=2 if allowed else 0,
                reason=("can_fire" if allowed else "cross-range ambiguity"),
                reason_key=("CAN_FIRE" if allowed else "CROSS_RANGE"),
            )

        state = EngagementState(fire_control_rejection_cooldown_frames=10)
        env = SymbolicReasoningEnv(
            max_entities=4,
            engagement_state=state,
            weapon_fire_prechecker=evaluate,
        )

        rejected = env.step(_payload("contact-1"), execute_commands=False)
        accepted = env.step(_payload("contact-2"), execute_commands=False)

        self.assertEqual(calls, ["contact-1", "contact-2"])
        self.assertEqual(
            rejected.decisions["red-ship-1"].rule_id, "R-FIRE-001"
        )
        self.assertEqual(
            accepted.decisions["red-ship-1"].conclusion,
            Conclusion.REQUEST_ATTACK,
        )
        self.assertTrue(accepted.facts["red-ship-1"].fire_control_available)

    def test_get_weapon_can_fire_is_normalized_with_denial_reason(self):
        rejected_stub = _FiringInfoStub(can_fire=False)
        rejected = precheck_weapon_fire(
            attacker_id="red-ship-1",
            target_id="contact-1",
            stub=rejected_stub,
        )
        accepted_stub = _FiringInfoStub(can_fire=True)
        accepted = precheck_weapon_fire(
            attacker_id="red-ship-1",
            target_id="contact-1",
            stub=accepted_stub,
        )

        self.assertFalse(rejected.can_fire)
        self.assertIn("cross-range ambiguity", rejected.reason)
        self.assertEqual(rejected_stub.calls, 1)
        self.assertTrue(accepted.can_fire)
        self.assertEqual(accepted.weapon_db_id, 2001)
        self.assertEqual(accepted.ready_quantity, 2)

    def test_attack_pipeline_never_submits_when_precheck_rejects(self):
        stub = _FiringInfoStub(can_fire=False)

        result = execute_attack_pipeline(
            attacker_id="red-ship-1",
            target_id="contact-1",
            stub=stub,
        )

        self.assertFalse(result.success)
        self.assertEqual(stub.calls, 1)
        self.assertEqual(stub.attack_calls, 0)

    def test_inventory_without_explicit_can_fire_evaluation_is_rejected(self):
        weapon = SimpleNamespace(
            weapon_db_id=2001,
            weapon_name="YJ-18",
            total_quantity=4,
            fire_evaluations=[],
            auto_fire_denied_reason="",
        )
        stub = SimpleNamespace(
            GetWeaponFiringInfo=lambda request, timeout: SimpleNamespace(
                suitable_weapons=[weapon]
            )
        )

        result = precheck_weapon_fire(
            attacker_id="red-ship-1",
            target_id="contact-1",
            stub=stub,
        )

        self.assertFalse(result.can_fire)
        self.assertIn("fire_evaluations=empty", result.reason)


if __name__ == "__main__":
    unittest.main()
