from __future__ import annotations

import unittest
import subprocess
import sys

from symbolic_reasoning.entity import EntityEncoder as SymbolicEntityEncoder
from symbolic_reasoning.control import TIME_COMPRESSION_LABELS


class SymbolicTimeCompressionMappingTests(unittest.TestCase):
    def test_symbolic_ui_time_compression_codes(self):
        self.assertEqual(
            TIME_COMPRESSION_LABELS,
            {
                0: "1x",
                1: "2x",
                2: "5x",
                3: "10x",
                4: "15x",
                5: "30x",
                6: "40x",
                7: "50x",
                8: "60x",
                9: "Turbo",
            },
        )

    def test_compiled_proto_uses_same_zero_to_nine_mapping(self):
        code = (
            "from symbolic_reasoning import engine_pb2 as pb; "
            "print(','.join(pb.EnumTimeCompression.Name(i) for i in range(10)))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.stdout.strip(),
            "OneSec,TwoSec,FiveSec,TenSec,FifteenSec,ThirtySec,"
            "FortySec,FiftySec,SixtySec,Turbo",
        )

class SymbolicSideMappingTests(unittest.TestCase):
    def test_red_side_overrides_payload_view_and_only_blue_is_targeted(self):
        payload = {
            "data": {
                # 即使上游错误地给了蓝方视角，固定业务口径仍以红方为我方。
                "sideGuid": "blue-side",
                "data": {
                    "UnitList": [
                        {
                            "guid": "red-controlled",
                            "SideId": "red-side",
                            "forceSide": "红方",
                            "IsContact": False,
                            "isCanManaged": True,
                            "BloodAmount": 100,
                            "unitCategory": 0,
                        },
                        {
                            "guid": "red-not-controlled",
                            "SideId": "red-side",
                            "forceSide": "红方",
                            "IsContact": False,
                            "isCanManaged": False,
                            "BloodAmount": 100,
                            "unitCategory": 1,
                        },
                        {
                            "guid": "blue-unit",
                            "contactGuid": "blue-contact",
                            "SideId": "blue-side",
                            "forceSide": "蓝方",
                            "IsContact": True,
                            "BloodAmount": 100,
                            "unitCategory": 0,
                        },
                    ]
                },
            }
        }

        situation = SymbolicEntityEncoder(max_entities=3).encode(payload)

        self.assertEqual(situation.own_side_id, "red-side")
        self.assertEqual(
            [entity.entity_id for entity in situation.own_entities],
            ["red-controlled"],
        )
        self.assertEqual(
            [entity.command_id for entity in situation.targets],
            ["blue-contact"],
        )
        red_locked = situation.find_entity("red-not-controlled")
        self.assertTrue(red_locked.is_own)
        self.assertFalse(red_locked.commandable)


if __name__ == "__main__":
    unittest.main()
