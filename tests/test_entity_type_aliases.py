import unittest

from entity import EntityEncoder, normalize_entity_type
from maddpg_live_adapter import legacy_entities_from_symbolic_payload


class EntityTypeAliasTest(unittest.TestCase):
    def test_live_interface_aliases(self):
        aliases = {
            "AIR": 0,
            "AIRCRAFT": 0,
            "飞机": 0,
            "SURFACE": 1,
            "SHIP": 1,
            "水面舰艇": 1,
            "航母": 1,
            "SUBMARINE": 2,
            "SUBSURFACE": 2,
            "潜艇": 2,
            "FACILITY": 3,
            "LAND": 3,
            "MISSILE": 4,
            "WEAPON": 4,
            "导弹": 4,
        }
        for label, expected in aliases.items():
            with self.subTest(label=label):
                self.assertEqual(expected, normalize_entity_type(label))

    def test_weapon_alias_wins_over_air_or_surface_words(self):
        self.assertEqual(4, normalize_entity_type("AIR-TO-AIR MISSILE"))
        self.assertEqual(4, normalize_entity_type("SURFACE-TO-AIR MISSILE"))

    def test_encoder_uses_normalized_type(self):
        encoded, mask = EntityEncoder(max_entities=2).encode([
            {"forceSide": "蓝方", "mdlType": "AIR", "isCanManaged": True},
            {"forceSide": "红方", "mdlType": "水面舰艇", "isCanManaged": False},
        ])
        self.assertEqual([0, 1], encoded[:2, 13].astype(int).tolist())
        self.assertEqual([1, 1], mask.astype(int).tolist())

    def test_symbolic_live_payload_converts_red_guid_and_blue_contact_id(self):
        payload = {
            "data": {
                "sideGuid": "red-side",
                "data": {
                    "UnitList": [
                        {
                            "guid": "red-ship",
                            "SideId": "red-side",
                            "IsContact": False,
                            "IsWeapon": False,
                            "unitType": 1,
                            "unitCategory": 1,
                            "UnitSpecificType": 3201,
                            "Icon2D": "/ArmyIcon/Ship/live.svg",
                            "longitude": 120.0,
                            "latitude": 30.0,
                            "altitude": 0.0,
                            "heading": 90.0,
                            "Speed": 20.0,
                            "BloodAmount": 100.0,
                            "weaponNumber": {"airNum": 8, "shipNum": 4},
                            "rangeStrike_Air": 185.2,
                            "rangeStrike_Surface": 92.6,
                            "fuelPercentage": 80.0,
                        },
                        {
                            "guid": "blue-real-guid",
                            "contactGuid": "blue-contact-guid",
                            "SideId": "blue-side",
                            "IsContact": True,
                            "IsWeapon": False,
                            "unitType": 1,
                            "unitCategory": 1,
                            "UnitSpecificType": 3201,
                            "Icon2D": "/ArmyIcon/Ship/live.svg",
                            "longitude": 121.0,
                            "latitude": 31.0,
                            "altitude": 0.0,
                            "BloodAmount": 0.0,
                        },
                    ]
                },
            }
        }
        entities = legacy_entities_from_symbolic_payload(payload)
        self.assertEqual("red-ship", entities[0]["mdlID"])
        self.assertTrue(entities[0]["isCanManaged"])
        self.assertAlmostEqual(100.0, entities[0]["maxRange"]["maxAir"])
        self.assertEqual("blue-contact-guid", entities[1]["mdlID"])
        self.assertFalse(entities[1]["isCanManaged"])
        self.assertEqual("SHIP", entities[1]["mdlType"])


if __name__ == "__main__":
    unittest.main()
