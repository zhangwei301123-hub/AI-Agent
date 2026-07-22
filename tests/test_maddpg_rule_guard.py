import math
import unittest

from maddpg_rule_guard import (
    ASW_TORPEDO_RELEASE_DISTANCE_M,
    AttackTargetInput,
    AttackThrottle,
    DEFAULT_ATTACK_QUANTITY,
    decide_attack,
    missile_points_at_entity,
    sonobuoy_deployment_allowed,
)


def target(
    index,
    target_id,
    entity_type,
    contact_type,
    longitude,
    latitude,
    altitude=0.0,
):
    return AttackTargetInput(
        index=index,
        target_id=target_id,
        raw={"contactType": contact_type},
        entity_type=entity_type,
        altitude_m=altitude,
        longitude=longitude,
        latitude=latitude,
    )


class MaddpgRuleGuardTest(unittest.TestCase):
    def setUp(self):
        self.aircraft = {
            "mdlType": "AIRCRAFT",
            "unitCategory": "Aircraft_Fighter",
            "unitTarget": [0, 1, 2, 3],
            "weaponNumber": {"airNum": 8, "shipNum": 4, "subNum": 2},
            "maxRange": {"maxAir": 60, "maxSurface": 60, "maxSubsurface": 20},
        }

    def decide(self, own_raw, own_type, targets, heading=0.0):
        return decide_attack(
            own_raw=own_raw,
            own_type=own_type,
            own_altitude_m=0.0,
            own_heading_deg=heading,
            own_longitude=0.0,
            own_latitude=0.0,
            targets=targets,
        )

    def test_priority_missile_is_selected_before_nearer_surface_target(self):
        result = self.decide(
            self.aircraft,
            0,
            [
                target(0, "surface-near", 1, 2, 0.0, 0.03),
                target(1, "missile-priority", 4, 1, 0.0, 0.20, 1000.0),
            ],
        )
        self.assertEqual("REQUEST_ATTACK", result.conclusion)
        self.assertEqual("missile-priority", result.candidate.target_id)

    def test_air_and_surface_targets_generate_expected_weapon_rules(self):
        air = self.decide(
            self.aircraft,
            0,
            [target(0, "air-target", 0, 0, 0.0, 0.05, 1000.0)],
        )
        surface = self.decide(
            self.aircraft,
            0,
            [target(0, "surface-target", 1, 2, 0.0, 0.05)],
        )
        self.assertEqual(("REQUEST_ATTACK", "R-WPN-001"), (air.conclusion, air.rule_id))
        self.assertEqual(("REQUEST_ATTACK", "R-WPN-002"), (surface.conclusion, surface.rule_id))
        self.assertLessEqual(air.quantity, 2)

    def test_only_aircraft_can_chase_out_of_range_target(self):
        short_range = dict(self.aircraft)
        short_range["maxRange"] = {"maxAir": 5, "maxSurface": 5, "maxSubsurface": 5}
        far_target = [target(0, "far", 0, 0, 0.0, 0.20, 1000.0)]
        aircraft = self.decide(short_range, 0, far_target)

        ship = dict(short_range)
        ship["mdlType"] = "SHIP"
        ship_result = self.decide(ship, 1, far_target)
        self.assertEqual(("CHASE_TO_RANGE", "R-RNG-004"), (aircraft.conclusion, aircraft.rule_id))
        self.assertEqual(("HOLD", "R-RNG-003"), (ship_result.conclusion, ship_result.rule_id))

    def test_aircraft_heading_must_be_strictly_less_than_30_degrees(self):
        result = self.decide(
            self.aircraft,
            0,
            [target(0, "north", 0, 0, 0.0, 0.05, 1000.0)],
            heading=30.0,
        )
        self.assertEqual(("CHASE_AND_ALIGN", "R-AIM-002"), (result.conclusion, result.rule_id))

    def test_asw_aircraft_releases_only_within_point_four_nautical_miles(self):
        asw = dict(self.aircraft)
        asw["unitCategory"] = "Aircraft_ASW"
        asw["unitTarget"] = [3]
        boundary_latitude = math.degrees(ASW_TORPEDO_RELEASE_DISTANCE_M / 6_371_000.0)
        boundary = self.decide(
            asw,
            0,
            [target(0, "sub-boundary", 2, 3, 0.0, boundary_latitude, 0.0)],
        )
        outside = self.decide(
            asw,
            0,
            [target(0, "sub-outside", 2, 3, 0.0, boundary_latitude * 1.01, 0.0)],
        )
        self.assertEqual("REQUEST_ATTACK", boundary.conclusion)
        self.assertEqual("R-WPN-003", boundary.rule_id)
        self.assertEqual(("CHASE_TO_RANGE", "R-ASW-001"), (outside.conclusion, outside.rule_id))

    def test_attack_throttle_limits_parallel_attackers_and_commits_only_success(self):
        throttle = AttackThrottle(cooldown_frames=600, max_parallel=3)
        throttle.begin_frame(["target"])
        for attacker in ("a1", "a2", "a3"):
            self.assertEqual(2, throttle.reserve(attacker, "target"))
        self.assertEqual(0, throttle.reserve("a4", "target"))

        throttle.commit("a1", "target", False)
        self.assertEqual(2, throttle.reserve("a4", "target"))
        throttle.commit("a2", "target", True)
        self.assertIn("a2", throttle.table["target"])

    def test_missile_lifecycle_allows_at_most_four_interceptors(self):
        throttle = AttackThrottle(max_interceptors=4)
        throttle.begin_frame(["missile"])
        self.assertEqual(2, throttle.reserve("a1", "missile", True, 2))
        self.assertEqual(2, throttle.reserve("a2", "missile", True, 2))
        self.assertEqual(0, throttle.reserve("a3", "missile", True, 2))
        throttle.commit("a1", "missile", True)
        throttle.commit("a2", "missile", True)
        self.assertEqual(4, throttle.interceptor_totals["missile"])

    def test_incoming_missile_requires_target_or_heading_toward_entity(self):
        toward = {"attitude": {"yaw": 0.0}}
        away = {"attitude": {"yaw": 180.0}}
        args = dict(
            own_id="own",
            missile_longitude=0.0,
            missile_latitude=-0.02,
            own_longitude=0.0,
            own_latitude=0.0,
        )
        self.assertTrue(missile_points_at_entity(toward, **args))
        self.assertFalse(missile_points_at_entity(away, **args))
        self.assertTrue(missile_points_at_entity({"targetId": "own"}, **args))

    def test_sonobuoy_rule_accepts_boundary_at_500_m_and_forces_inventory(self):
        patrol = {
            "mdlType": "AIRCRAFT",
            "unitCategory": "Aircraft_ASW",
            "weaponNumber": {"buoyNum": 1},
        }
        area = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.assertTrue(sonobuoy_deployment_allowed(patrol, 0.0, 0.5, 500.0, area))
        self.assertFalse(sonobuoy_deployment_allowed(patrol, 0.0, 0.5, 500.1, area))
        patrol["weaponNumber"]["buoyNum"] = 0
        self.assertFalse(sonobuoy_deployment_allowed(patrol, 0.0, 0.5, 100.0, area))

    def test_default_attack_quantity_matches_symbolic_salvo_limit(self):
        self.assertGreaterEqual(DEFAULT_ATTACK_QUANTITY, 1)
        self.assertLessEqual(DEFAULT_ATTACK_QUANTITY, 2)


if __name__ == "__main__":
    unittest.main()
