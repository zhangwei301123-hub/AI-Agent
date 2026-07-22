import copy
import unittest
from unittest.mock import patch

import grpc

import execute


class _TransientRpcError(grpc.RpcError):
    def __init__(self, code):
        super().__init__()
        self._code = code

    def code(self):
        return self._code


class _Source:
    def __init__(self, results):
        self.results = list(results)

    def fetch_payload(self):
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class LiveSituationResilienceTest(unittest.TestCase):
    def setUp(self):
        self.old_source = execute._LIVE_SOURCE
        self.old_entities = copy.deepcopy(execute._LAST_LIVE_ENTITIES)
        self.old_symbolic = execute._LAST_SYMBOLIC_SITUATION
        self.old_stale = execute._LIVE_SITUATION_STALE
        execute._LAST_LIVE_ENTITIES = None
        execute._LAST_SYMBOLIC_SITUATION = None
        execute._LIVE_SITUATION_STALE = False

    def tearDown(self):
        execute._LIVE_SOURCE = self.old_source
        execute._LAST_LIVE_ENTITIES = self.old_entities
        execute._LAST_SYMBOLIC_SITUATION = self.old_symbolic
        execute._LIVE_SITUATION_STALE = self.old_stale

    def test_runtime_deadline_uses_cached_state_and_marks_it_stale(self):
        live_entities = [{"mdlID": "own", "isCanManaged": True}]
        payload = {
            "data": {
                "sideGuid": "red-side",
                "data": {"UnitList": []},
            }
        }
        execute._LIVE_SOURCE = _Source([
            payload,
            _TransientRpcError(grpc.StatusCode.DEADLINE_EXCEEDED),
        ])

        with patch.object(
            execute,
            "legacy_entities_from_symbolic_payload",
            return_value=live_entities,
        ):
            first = execute.get_Situaction4test(_Logger())
            cached = execute.get_Situaction4test(_Logger())

        self.assertEqual(live_entities, first)
        self.assertEqual(live_entities, cached)
        self.assertIsNot(first, cached)
        self.assertTrue(execute.is_live_situation_stale())

    def test_successful_refresh_clears_stale_marker(self):
        execute._LAST_LIVE_ENTITIES = [{"mdlID": "old"}]
        execute._LIVE_SITUATION_STALE = True
        execute._LIVE_SOURCE = _Source([
            {
                "data": {
                    "sideGuid": "red-side",
                    "data": {"UnitList": []},
                }
            }
        ])
        fresh_entities = [{"mdlID": "fresh", "isCanManaged": True}]

        with patch.object(
            execute,
            "legacy_entities_from_symbolic_payload",
            return_value=fresh_entities,
        ):
            result = execute.get_Situaction4test(_Logger())

        self.assertEqual(fresh_entities, result)
        self.assertFalse(execute.is_live_situation_stale())


if __name__ == "__main__":
    unittest.main()
