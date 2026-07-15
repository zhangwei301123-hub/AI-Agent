from __future__ import annotations

import unittest

from symbolic_reasoning import (
    Conclusion,
    ReasoningFacts,
    SymbolicReasoningAgent,
    TargetEvaluation,
    TargetDomain,
)


ENTITY_ID = "our-aircraft-0001"
TARGET_ID = "enemy-aircraft-0001"


def attack_facts(**changes):
    values = {
        "entity_id": ENTITY_ID,
        "own_platform_type": TargetDomain.AIR,
        "target_id": TARGET_ID,
        "target_domain": TargetDomain.AIR,
        "detected_target_count": 1,
        "attack_authorized": True,
        "target_type_allowed": True,
        "weapon_available": True,
        "compatible_weapon_count": 2,
        "expected_weapon_type": "AIR_DEFENCE_OR_AIR_TO_AIR_MISSILE",
        "within_attack_range": True,
        "distance_km": 50.0,
        "max_attack_range_km": 100.0,
        "aimed_at_target": True,
        "heading_difference_deg": 10.0,
        "safety_clearance": True,
        "chase_allowed": True,
        "concurrency_slot_available": True,
        "target_lon": 120.1,
        "target_lat": 30.2,
        "attack_altitude_level": 3,
    }
    values.update(changes)
    return ReasoningFacts(**values)


class SymbolicReasoningAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = SymbolicReasoningAgent()

    def test_incoming_missile_evasion_has_highest_action_priority(self):
        decision = self.agent.reason(
            attack_facts(
                incoming_missile=True,
                incoming_missile_id="missile-1",
                incoming_missile_distance_km=5.0,
                incoming_missile_heading_deg=90.0,
                evade_lon=120.0,
                evade_lat=29.95,
            )
        )

        self.assertEqual(decision.conclusion, Conclusion.EVADE_MISSILE)
        self.assertEqual(decision.rule_id, "R-MSL-002")
        self.assertEqual(decision.actions[2], [0.9, 120.0, 29.95, 5, 4])
        self.assertEqual(decision.actions[4][0], 0.01)
        self.assertEqual(decision.actions[6][0], 0.01)

    def test_attack_when_all_business_constraints_pass(self):
        facts = attack_facts()
        evaluation = TargetEvaluation.from_facts(facts)
        decision = self.agent.reason(facts)

        self.assertTrue(evaluation.immediate_candidate)
        self.assertTrue(evaluation.attack_request_allowed)
        self.assertEqual(decision.conclusion, Conclusion.REQUEST_ATTACK)
        self.assertEqual(decision.rule_id, "R-WPN-001")
        self.assertEqual(decision.actions[3][1:3], [4, 3])
        self.assertEqual(decision.actions[4][1], TARGET_ID)
        self.assertIn("within_attack_range=True", decision.explanation)
        self.assertIn("R-CON-001", decision.explanation)

    def test_distance_equal_to_max_range_allows_attack_request(self):
        decision = self.agent.reason(
            attack_facts(distance_km=100.0, max_attack_range_km=100.0)
        )

        self.assertEqual(decision.conclusion, Conclusion.REQUEST_ATTACK)

    def test_only_aircraft_chases_target_outside_range(self):
        aircraft = self.agent.reason(
            attack_facts(
                within_attack_range=False,
                distance_km=120.0,
                max_attack_range_km=100.0,
            )
        )
        ship = self.agent.reason(
            attack_facts(
                own_platform_type=TargetDomain.SURFACE,
                within_attack_range=False,
                distance_km=120.0,
                max_attack_range_km=100.0,
            )
        )

        self.assertEqual(aircraft.conclusion, Conclusion.CHASE_TO_RANGE)
        self.assertEqual(aircraft.actions[2][1:3], [120.1, 30.2])
        self.assertEqual(aircraft.actions[4][0], 0.01)
        self.assertEqual(ship.conclusion, Conclusion.HOLD)
        self.assertEqual(ship.rule_id, "R-RNG-003")

    def test_aim_boundary_is_strict_and_submarine_is_not_exempt(self):
        aircraft = self.agent.reason(
            attack_facts(aimed_at_target=False, heading_difference_deg=30.0)
        )
        submarine = self.agent.reason(
            attack_facts(
                own_platform_type=TargetDomain.SUBMARINE,
                aimed_at_target=False,
                heading_difference_deg=30.0,
            )
        )
        ship = self.agent.reason(
            attack_facts(
                own_platform_type=TargetDomain.SURFACE,
                aimed_at_target=False,
                heading_difference_deg=180.0,
            )
        )

        self.assertEqual(aircraft.conclusion, Conclusion.CHASE_AND_ALIGN)
        self.assertEqual(submarine.conclusion, Conclusion.HOLD)
        self.assertEqual(submarine.rule_id, "R-AIM-002")
        self.assertEqual(ship.conclusion, Conclusion.REQUEST_ATTACK)

    def test_concurrency_and_interceptor_limits_fail_closed(self):
        concurrency = self.agent.reason(
            attack_facts(
                concurrency_slot_available=False,
                active_attackers_on_target=3,
            )
        )
        interceptor = self.agent.reason(
            attack_facts(target_is_missile=True, interceptors_launched=4)
        )

        self.assertEqual(concurrency.rule_id, "R-CON-001")
        self.assertEqual(concurrency.conclusion, Conclusion.HOLD)
        self.assertEqual(interceptor.rule_id, "R-INT-001")
        self.assertEqual(interceptor.conclusion, Conclusion.HOLD)

    def test_target_evaluation_centralizes_attack_blockers(self):
        concurrency = TargetEvaluation.from_facts(
            attack_facts(
                concurrency_slot_available=False,
                active_attackers_on_target=3,
            )
        )
        interceptor = TargetEvaluation.from_facts(
            attack_facts(target_is_missile=True, interceptors_launched=4)
        )
        out_of_range = TargetEvaluation.from_facts(
            attack_facts(
                within_attack_range=False,
                distance_km=120.0,
                max_attack_range_km=100.0,
            )
        )

        self.assertTrue(concurrency.concurrency_blocked)
        self.assertFalse(concurrency.candidate_eligible)
        self.assertTrue(interceptor.interceptor_blocked)
        self.assertFalse(interceptor.attack_request_allowed)
        self.assertTrue(out_of_range.pursuit_candidate)
        self.assertTrue(out_of_range.can_chase)

    def test_patrol_aircraft_can_deploy_buoy_on_500m_boundary(self):
        facts = ReasoningFacts(
            entity_id=ENTITY_ID,
            own_platform_type=TargetDomain.AIR,
            is_patrol_aircraft=True,
            has_patrol_mission=True,
            inside_patrol_area=True,
            altitude_above_sea_m=500.0,
            sonobuoy_count=2,
        )
        decision = self.agent.reason(facts)

        self.assertEqual(decision.conclusion, Conclusion.DEPLOY_SONOBUOY)
        self.assertEqual(decision.actions[6][0], 0.9)

    def test_search_when_no_target_and_buoy_conditions_fail(self):
        decision = self.agent.reason(ReasoningFacts(entity_id=ENTITY_ID))

        self.assertEqual(decision.conclusion, Conclusion.SEARCH)
        self.assertEqual(decision.actions[5][0], 0.9)

    def test_missing_safety_permission_fails_closed(self):
        decision = self.agent.reason(attack_facts(safety_clearance=False))

        self.assertEqual(decision.conclusion, Conclusion.HOLD)
        self.assertEqual(decision.rule_id, "R-VAL-001")
        self.assertTrue(all(action[0] == 0.01 for action in decision.actions))

    def test_run_reuses_execute_actions_interface_and_reports_feedback(self):
        captured = {}

        def fake_execute(actions_dict, enemy_ids, probability, logger):
            captured["actions_dict"] = actions_dict
            captured["enemy_ids"] = enemy_ids
            captured["probability"] = probability
            captured["logger"] = logger
            return {
                ENTITY_ID: [False, False, False, True, True, False, False, False]
            }, [0] * 8

        result = self.agent.run(
            [attack_facts()],
            enemy_ids=[TARGET_ID],
            probability=0.7,
            executor=fake_execute,
        )

        self.assertEqual(
            result.decisions[ENTITY_ID].conclusion,
            Conclusion.REQUEST_ATTACK,
        )
        self.assertIs(result.actions_dict, captured["actions_dict"])
        self.assertEqual(captured["enemy_ids"], [TARGET_ID])
        self.assertEqual(captured["probability"], 0.7)
        self.assertEqual(result.execution_status[ENTITY_ID], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
