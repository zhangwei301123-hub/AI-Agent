"""符号推理使用的任务列表和巡逻区域加载。"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple


MissionFetcher = Callable[[], Any]
MissionInfo = Dict[str, Any]


def _load_project_fetcher() -> MissionFetcher:
    """构造只使用本包 protobuf 的任务列表读取器。"""

    import grpc
    from google.protobuf.empty_pb2 import Empty
    from . import engine_pb2_grpc

    endpoint = os.environ.get(
        "SYMBOLIC_REASONING_RPC_TARGET", "10.2.0.106:50051"
    )

    def fetch() -> Any:
        channel = grpc.insecure_channel(endpoint)
        try:
            stub = engine_pb2_grpc.SimulationServiceStub(channel)
            return stub.getMissionList(Empty(), timeout=5.0)
        finally:
            channel.close()

    return fetch


def load_project_mission_areas(
    fetcher: Optional[MissionFetcher] = None,
) -> Dict[str, MissionInfo]:
    """调用 ``getMissionList`` 并保留巡逻判断和区域航点需要的字段。"""

    response = (fetcher or _load_project_fetcher())()
    result: Dict[str, MissionInfo] = {}
    for mission in getattr(response, "mission", ()):
        mission_id = str(getattr(mission, "missionId", "") or "").strip()
        if not mission_id:
            continue

        points: List[Tuple[float, float]] = []
        for point in getattr(mission, "areaPoints", ()):
            try:
                lon = float(getattr(point, "lon"))
                lat = float(getattr(point, "lat"))
            except (AttributeError, TypeError, ValueError):
                continue
            candidate = (lon, lat)
            if not points or points[-1] != candidate:
                points.append(candidate)
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(points) < 3:
            continue

        mission_name = str(
            getattr(mission, "missionName", "") or ""
        ).strip()
        mission_type = int(getattr(mission, "missionType", 0) or 0)
        patrol_text = mission_name.lower()
        result[mission_id] = {
            "mission_name": mission_name,
            "mission_type": mission_type,
            "is_patrol": "巡逻" in mission_name or "patrol" in patrol_text,
            "area_points": points,
        }
    return result
