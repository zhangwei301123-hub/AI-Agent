"""实时 RPC 态势适配与详细武器库存读取。"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import StringValue

from . import engine_pb2, engine_pb2_grpc


RANGE_NM_TO_KM = 1.852
GUIDED_WEAPON_TYPE = 2001


@dataclass(frozen=True)
class DetailedWeaponInventory:
    """由 ``GetUnitData.unit_weapons`` 聚合出的真实武器能力。"""

    air_count: int = 0
    surface_count: int = 0
    land_count: int = 0
    submarine_count: int = 0
    sonobuoy_count: int = 0
    max_air_range_km: float = 0.0
    max_surface_range_km: float = 0.0
    max_land_range_km: float = 0.0
    max_submarine_range_km: float = 0.0
    unit_name: str = ""

    @property
    def weapon_number(self) -> Dict[str, int]:
        return {
            "airNum": self.air_count,
            "shipNum": self.surface_count,
            "landNum": self.land_count,
            "subNum": self.submarine_count,
            "sonobuoyNum": self.sonobuoy_count,
        }


def inventory_from_unit_data(response: Any) -> DetailedWeaponInventory:
    """聚合详细武器；导弹判断不再使用 ``SituationDataw.weaponNumber``。"""

    air_count = 0
    surface_count = 0
    land_count = 0
    submarine_count = 0
    sonobuoy_count = 0
    max_air = 0.0
    max_surface = 0.0
    max_land = 0.0
    max_submarine = 0.0

    for weapon in getattr(response, "unit_weapons", ()):
        quantity = max(0, int(getattr(weapon, "num", 0) or 0))
        if quantity <= 0:
            continue
        weapon_type = int(getattr(weapon, "weapon_type", 0) or 0)
        name = str(getattr(weapon, "text", "") or "")
        name_lower = name.casefold()
        air_range = max(0.0, float(getattr(weapon, "air_range_strike", 0.0)))
        surface_range = max(
            0.0, float(getattr(weapon, "surface_range_strike", 0.0))
        )
        land_range = max(0.0, float(getattr(weapon, "land_range_strike", 0.0)))
        submarine_range = max(
            0.0, float(getattr(weapon, "submarine_range_strike", 0.0))
        )

        # 对空、对海、对陆规则要求使用导弹；舰炮即使有对空射程也不计入导弹量。
        if weapon_type == GUIDED_WEAPON_TYPE:
            if air_range > 0.0:
                air_count += quantity
                max_air = max(max_air, air_range * RANGE_NM_TO_KM)
            if surface_range > 0.0:
                surface_count += quantity
                max_surface = max(max_surface, surface_range * RANGE_NM_TO_KM)
            if land_range > 0.0:
                land_count += quantity
                max_land = max(max_land, land_range * RANGE_NM_TO_KM)

        if submarine_range > 0.0:
            submarine_count += quantity
            max_submarine = max(
                max_submarine, submarine_range * RANGE_NM_TO_KM
            )
        if "浮标" in name or "sonobuoy" in name_lower:
            sonobuoy_count += quantity

    status = getattr(response, "unit_current_status", None)
    unit_name = str(getattr(status, "unit_name", "") or "")
    return DetailedWeaponInventory(
        air_count=air_count,
        surface_count=surface_count,
        land_count=land_count,
        submarine_count=submarine_count,
        sonobuoy_count=sonobuoy_count,
        max_air_range_km=max_air,
        max_surface_range_km=max_surface,
        max_land_range_km=max_land,
        max_submarine_range_km=max_submarine,
        unit_name=unit_name,
    )


def _type_flags(model_type: str) -> Tuple[bool, bool, int, str]:
    text = str(model_type or "").strip()
    normalized = text.casefold()
    contact_types = {
        "air": (0, "/ArmyIcon/Aircraft/live.svg"),
        "surface": (1, "/ArmyIcon/Ship/live.svg"),
        "subsurface": (2, "/ArmyIcon/Submarine/live.svg"),
        "submarine": (2, "/ArmyIcon/Submarine/live.svg"),
        "land": (3, "/ArmyIcon/Facility/live.svg"),
    }
    if normalized in contact_types:
        unit_type, icon = contact_types[normalized]
        return True, False, unit_type, icon

    is_weapon = any(
        marker in normalized
        for marker in ("导弹", "鱼雷", "missile", "torpedo", "weapon")
    )
    if is_weapon:
        underwater = "鱼雷" in normalized or "torpedo" in normalized
        return (
            False,
            True,
            2 if underwater else 0,
            (
                "/ArmyIcon/Submarine/torpedo.svg"
                if underwater
                else "/ArmyIcon/Aircraft/missile.svg"
            ),
        )
    if any(marker in normalized for marker in ("飞机", "航空器", "aircraft")):
        return False, False, 0, "/ArmyIcon/Aircraft/live.svg"
    if any(marker in normalized for marker in ("水面", "舰", "船", "航母", "ship")):
        return False, False, 1, "/ArmyIcon/Ship/live.svg"
    if any(marker in normalized for marker in ("潜艇", "水下", "submarine")):
        return False, False, 2, "/ArmyIcon/Submarine/live.svg"
    if any(marker in normalized for marker in ("地面", "设施", "facility", "land")):
        return False, False, 3, "/ArmyIcon/Facility/live.svg"
    return False, False, -1, ""


def _normalize_unit(
    unit: Any,
    inventory: Optional[DetailedWeaponInventory],
) -> Dict[str, Any]:
    is_contact, is_weapon, unit_type, icon = _type_flags(unit.mdlType)
    side = str(unit.forceSide or "")
    velocity = unit.velocity
    speed = math.sqrt(
        float(velocity.vx) ** 2
        + float(velocity.vy) ** 2
        + float(velocity.vz) ** 2
    )
    max_range = unit.maxRange
    inventory = inventory or DetailedWeaponInventory()

    range_air = max(
        float(max_range.maxAir), inventory.max_air_range_km
    )
    range_surface = max(
        float(max_range.maxSurface), inventory.max_surface_range_km
    )
    range_land = max(
        float(max_range.maxLand), inventory.max_land_range_km
    )
    range_submarine = max(
        float(max_range.maxSubsurface), inventory.max_submarine_range_km
    )
    name = inventory.unit_name or str(unit.mdlType or unit.mdlID)
    is_red = side.strip().casefold().replace(" ", "") in ("红方", "red", "redside")

    return {
        "guid": str(unit.mdlID),
        "name": name,
        "unitname": name,
        "forceSide": side,
        "SideId": side,
        "IsContact": is_contact,
        "IsWeapon": is_weapon,
        "isCanManaged": bool(is_red and not is_contact and not is_weapon),
        "unitType": unit_type,
        "unitCategory": int(unit.unitCategory),
        "UnitSpecificType": int(unit.contactType),
        "Icon2D": icon,
        "longitude": float(unit.entitySpatialCoord.longitude),
        "latitude": float(unit.entitySpatialCoord.latitude),
        "altitude": float(unit.entitySpatialCoord.altitude),
        "heading": float(unit.attitude.yaw),
        "Speed": speed,
        "BloodAmount": float(unit.activeLvl),
        "rangeStrike_Air": range_air,
        "rangeStrike_Surface": range_surface,
        "rangeStrike_Land": range_land,
        "rangeStrike_Submarine": range_submarine,
        "weaponNumber": inventory.weapon_number,
        "missionId": str(unit.missionId or ""),
        "missionType": int(unit.missionType),
        "CommText": "",
        "JammText": "被干扰" if int(unit.innerstates.IsJamReaction) else "",
        "CurrentScenarioTime": str(unit.reportTime),
    }


class RpcSituationSource:
    """从当前推演服务获取实时态势，并用详细武器接口补齐编码。"""

    def __init__(
        self,
        rpc_target: str,
        timeout: float = 20.0,
        logger: Any = None,
        stub: Any = None,
    ) -> None:
        if not isinstance(rpc_target, str) or not rpc_target.strip():
            raise ValueError("rpc_target 必须是非空字符串")
        if timeout <= 0.0:
            raise ValueError("timeout 必须大于 0")
        self.rpc_target = rpc_target.strip()
        self.timeout = float(timeout)
        self.logger = logger
        self._red_side_guid = ""
        self._situation_user_id = int(time.time() * 1000000)
        self._channel = None
        if stub is None:
            import grpc

            self._channel = grpc.insecure_channel(self.rpc_target)
            stub = engine_pb2_grpc.SimulationServiceStub(self._channel)
        self.stub = stub

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def control_signal(self) -> str:
        status = self.stub.GetEngineStatus(
            engine_pb2.EmptyRequest(), timeout=self.timeout
        )
        return {
            1: "pause",
            2: "running",
            3: "pause",
            4: "stop",
        }.get(int(status.run_status), "pause")

    def fetch_missions(self) -> Any:
        return self.stub.getMissionList(Empty(), timeout=self.timeout)

    def fetch_payload(self) -> Mapping[str, Any]:
        """优先读取红方视角态势，确保目标命令使用真实 Contact ID。"""

        try:
            return self._fetch_red_view_payload()
        except AttributeError:
            # 仅供未实现 GetThreeSituation 的旧测试 stub/旧 SDK 兼容。
            return self._fetch_legacy_payload()

    def _red_side(self) -> str:
        if self._red_side_guid:
            return self._red_side_guid
        status = self.stub.GetEngineStatus(
            engine_pb2.EmptyRequest(), timeout=self.timeout
        )
        for side in status.side_list:
            side_name = str(side.side_name or "").strip().casefold()
            color = str(side.color_name or "").strip().upper()
            if side_name in ("红方", "red", "redside") or color == "#FF0000":
                self._red_side_guid = str(side.guid_str)
                break
        if not self._red_side_guid:
            raise RuntimeError("GetEngineStatus 未返回红方 side_guid")
        return self._red_side_guid

    def _fetch_red_view_payload(self) -> Mapping[str, Any]:
        red_side_guid = self._red_side()
        response = self.stub.GetThreeSituation(
            engine_pb2.ThreeSituationRequest(
                user_id=self._situation_user_id,
                side_guid=red_side_guid,
                is_god_view=False,
                is_distribute=False,
            ),
            timeout=self.timeout,
        )
        wrapped = StringValue()
        if not response.data.Unpack(wrapped):
            raise ValueError(
                "GetThreeSituation.data 不是 google.protobuf.StringValue"
            )
        send_msg = json.loads(wrapped.value)
        if not isinstance(send_msg, Mapping):
            raise ValueError("GetThreeSituation JSON 顶层不是对象")
        raw_units = send_msg.get("UnitList")
        if not isinstance(raw_units, list):
            raise ValueError("GetThreeSituation JSON 缺少 UnitList")

        units = [dict(unit) for unit in raw_units if isinstance(unit, Mapping)]
        inventory_ids = [
            str(unit.get("guid") or "")
            for unit in units
            if self._needs_view_inventory(unit, red_side_guid)
        ]
        inventories = self._fetch_inventories(inventory_ids)
        for unit in units:
            unit_id = str(unit.get("guid") or "")
            inventory = inventories.get(unit_id)
            if inventory is None:
                continue
            unit["isCanManaged"] = True
            unit["weaponNumber"] = inventory.weapon_number
            unit["rangeStrike_Air"] = max(
                float(unit.get("rangeStrike_Air") or 0.0),
                inventory.max_air_range_km,
            )
            unit["rangeStrike_Surface"] = max(
                float(unit.get("rangeStrike_Surface") or 0.0),
                inventory.max_surface_range_km,
            )
            unit["rangeStrike_Land"] = max(
                float(unit.get("rangeStrike_Land") or 0.0),
                inventory.max_land_range_km,
            )
            unit["rangeStrike_Submarine"] = max(
                float(unit.get("rangeStrike_Submarine") or 0.0),
                inventory.max_submarine_range_km,
            )

        normalized_send_msg = dict(send_msg)
        normalized_send_msg["UnitList"] = units
        contacts = [
            unit
            for unit in units
            if bool(unit.get("IsContact")) and unit.get("contactGuid")
        ]
        self._log(
            "info",
            "[实时态势] source=GetThreeSituation(red-view) entities=%s "
            "contacts=%s red_inventory=%s",
            len(units),
            len(contacts),
            len(inventories),
        )
        return {
            "data": {
                "sideGuid": red_side_guid,
                "data": normalized_send_msg,
            }
        }

    @staticmethod
    def _needs_view_inventory(unit: Mapping[str, Any], red_side_guid: str) -> bool:
        side_id = str(unit.get("SideId") or unit.get("sideId") or "")
        side_name = str(unit.get("side") or unit.get("forceSide") or "")
        is_red = side_id == red_side_guid or side_name.strip().casefold() in (
            "红方",
            "red",
            "redside",
        )
        return bool(
            is_red
            and not unit.get("IsContact")
            and not unit.get("IsWeapon")
            and unit.get("guid")
        )

    def _fetch_legacy_payload(self) -> Mapping[str, Any]:
        """旧 getSituation 仅作为 SDK 测试兼容，不用于当前真实服务。"""

        response = self.stub.getSituation(Empty(), timeout=self.timeout)
        units = tuple(response.situaction)
        inventory_ids = [
            str(unit.mdlID)
            for unit in units
            if self._needs_inventory(unit)
        ]
        inventories = self._fetch_inventories(inventory_ids)
        normalized = [
            _normalize_unit(unit, inventories.get(str(unit.mdlID)))
            for unit in units
        ]
        report_time = max((int(unit.reportTime) for unit in units), default=0)
        self._log(
            "info",
            "[实时态势] entities=%s red_inventory=%s report_frame=%s",
            len(normalized),
            len(inventories),
            report_time,
        )
        return {
            "data": {
                "sideGuid": "红方",
                "data": {
                    "ScenName": "live-rpc",
                    "CurrentTime": str(report_time),
                    "UnitList": normalized,
                },
            }
        }

    @staticmethod
    def _needs_inventory(unit: Any) -> bool:
        is_contact, is_weapon, _, _ = _type_flags(unit.mdlType)
        side = str(unit.forceSide or "").strip().casefold().replace(" ", "")
        return (
            side in ("红方", "red", "redside")
            and not is_contact
            and not is_weapon
        )

    def _fetch_inventories(
        self, unit_ids: Sequence[str]
    ) -> Dict[str, DetailedWeaponInventory]:
        result: Dict[str, DetailedWeaponInventory] = {}
        if not unit_ids:
            return result
        workers = min(8, len(unit_ids))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._fetch_inventory, unit_id): unit_id
                for unit_id in unit_ids
            }
            for future in as_completed(futures):
                unit_id = futures[future]
                try:
                    inventory = future.result()
                except Exception as error:
                    self._log(
                        "warning",
                        "[实时武器] GetUnitData 失败 id=%s error=%s",
                        unit_id,
                        error,
                    )
                    continue
                result[unit_id] = inventory
                self._log(
                    "info",
                    "[实时武器] id=%s air=%s surface=%s land=%s sub=%s buoy=%s",
                    unit_id,
                    inventory.air_count,
                    inventory.surface_count,
                    inventory.land_count,
                    inventory.submarine_count,
                    inventory.sonobuoy_count,
                )
        return result

    def _fetch_inventory(self, unit_id: str) -> DetailedWeaponInventory:
        response = self.stub.GetUnitData(
            engine_pb2.GetUnitDataRequest(unit_id=unit_id),
            timeout=self.timeout,
        )
        return inventory_from_unit_data(response)

    def _log(self, level: str, message: str, *args: Any) -> None:
        if self.logger is not None:
            getattr(self.logger, level)(message, *args)
