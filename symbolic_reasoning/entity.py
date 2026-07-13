"""对三维态势 ``SendMsg.UnitList`` 进行字段校验和数值编码。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


class TargetDomain(IntEnum):
    """用于选择对应打击距离的目标域。"""

    AIR = 0
    SURFACE = 1
    SUBMARINE = 2
    LAND = 3
    UNKNOWN = 4


# 字段均来自 source/态势数据字段说明.md 的 UnitInfo。
FEATURE_NAMES: Tuple[str, ...] = (
    "is_own",
    "is_contact",
    "is_weapon",
    "unit_type",
    "unit_category",
    "target_domain",
    "longitude_deg",
    "latitude_deg",
    "altitude_m",
    "heading_deg",
    "speed",
    "health_pct",
    "range_detect_km",
    "range_strike_air_km",
    "range_strike_surface_km",
    "range_strike_land_km",
    "range_strike_submarine_km",
    "range_fly_max_km",
    "communication_ok",
    "radar_jammed",
    "commandable",
)


def _number(value: Any, default: float = -1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _integer(value: Any, default: int = -1) -> int:
    number = _number(value, float(default))
    return int(number) if number != float(default) else default


@dataclass(frozen=True)
class EncodedEntity:
    """保留符号推理所需的实体字段和对应编码。"""

    entity_id: str
    contact_id: Optional[str]
    name: str
    side_id: str
    is_own: bool
    is_contact: bool
    is_weapon: bool
    unit_type: int
    unit_category: int
    domain: TargetDomain
    longitude: float
    latitude: float
    altitude_m: float
    heading_deg: float
    speed: float
    health_pct: float
    range_detect_km: float
    range_strike_air_km: float
    range_strike_surface_km: float
    range_strike_land_km: float
    range_strike_submarine_km: float
    range_fly_max_km: float
    communication_ok: bool
    radar_jammed: bool
    commandable: bool
    icon_2d: str
    vector: np.ndarray

    @property
    def command_id(self) -> str:
        """己方命令使用 guid，接触目标命令优先使用 contactGuid。"""

        if self.is_contact and self.contact_id:
            return self.contact_id
        return self.entity_id

    def strike_range_for(self, target: "EncodedEntity") -> float:
        if target.domain is TargetDomain.AIR:
            return max(0.0, self.range_strike_air_km)
        if target.domain is TargetDomain.SURFACE:
            return max(0.0, self.range_strike_surface_km)
        if target.domain is TargetDomain.SUBMARINE:
            return max(0.0, self.range_strike_submarine_km)
        if target.domain is TargetDomain.LAND:
            return max(0.0, self.range_strike_land_km)
        return 0.0

    def distance_to_km(self, target: "EncodedEntity") -> float:
        """计算包含高度差的近似空间距离，返回公里。"""

        radius_km = 6371.0088
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(target.latitude)
        dlat = lat2 - lat1
        dlon = math.radians(target.longitude - self.longitude)
        value = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        )
        ground_km = 2.0 * radius_km * math.asin(min(1.0, math.sqrt(value)))
        altitude_km = (target.altitude_m - self.altitude_m) / 1000.0
        return math.hypot(ground_km, altitude_km)


@dataclass(frozen=True)
class EncodedSituation:
    own_side_id: str
    scene_name: str
    current_time: str
    entities: Tuple[EncodedEntity, ...]
    encoded_data: np.ndarray
    mask: np.ndarray

    @property
    def own_entities(self) -> Tuple[EncodedEntity, ...]:
        return tuple(entity for entity in self.entities if entity.commandable)

    @property
    def targets(self) -> Tuple[EncodedEntity, ...]:
        return tuple(
            entity
            for entity in self.entities
            # 接触目标的 BloodAmount 可能因通信/识别不足而为 0，不能据此丢弃。
            if not entity.is_own and (entity.is_contact or entity.health_pct != 0)
        )


class EntityEncoder:
    """编码 ``我方视角下态势完整响应`` 中的 UnitList。"""

    def __init__(self, max_entities: int = 700) -> None:
        if max_entities <= 0:
            raise ValueError("max_entities 必须大于 0")
        self.max_entities = max_entities
        self.feature_dim = len(FEATURE_NAMES)

    def encode(self, payload: Mapping[str, Any]) -> EncodedSituation:
        send_msg, own_side_id = self._extract_send_msg(payload)
        units = send_msg.get("UnitList")
        if not isinstance(units, list):
            raise ValueError("态势数据缺少 data.data.UnitList 列表")

        if not own_side_id:
            own_side_id = self._infer_own_side_id(units)
        if not own_side_id:
            raise ValueError("态势数据缺少 sideGuid，且无法从己方实体推断")

        encoded_data = np.full(
            (self.max_entities, self.feature_dim), -1.0, dtype=np.float32
        )
        mask = np.zeros(self.max_entities, dtype=np.bool_)
        entities: List[EncodedEntity] = []

        for index, unit in enumerate(units[: self.max_entities]):
            if not isinstance(unit, Mapping):
                raise ValueError("UnitList[{}] 不是对象".format(index))
            entity = self._encode_unit(unit, own_side_id)
            entities.append(entity)
            encoded_data[index] = entity.vector
            mask[index] = True

        return EncodedSituation(
            own_side_id=own_side_id,
            scene_name=str(send_msg.get("ScenName") or ""),
            current_time=str(send_msg.get("CurrentTime") or ""),
            entities=tuple(entities),
            encoded_data=encoded_data,
            mask=mask,
        )

    @staticmethod
    def _extract_send_msg(
        payload: Mapping[str, Any]
    ) -> Tuple[Mapping[str, Any], str]:
        if not isinstance(payload, Mapping):
            raise ValueError("态势数据必须是 JSON 对象")

        # 完整响应：data.sideGuid + data.data(UnitList)。
        outer_data = payload.get("data")
        if isinstance(outer_data, Mapping):
            inner_data = outer_data.get("data")
            if isinstance(inner_data, Mapping) and "UnitList" in inner_data:
                side_id = str(
                    outer_data.get("sideGuid")
                    or outer_data.get("SideGuid")
                    or inner_data.get("SideGuid")
                    or ""
                )
                return inner_data, side_id

        # 也允许调用方直接传 SendMsg。
        if "UnitList" in payload:
            side_id = str(payload.get("SideGuid") or payload.get("sideGuid") or "")
            return payload, side_id

        raise ValueError("不支持的态势结构，期望完整响应 data.data.UnitList 或 SendMsg")

    @staticmethod
    def _infer_own_side_id(units: Sequence[Mapping[str, Any]]) -> str:
        for unit in units:
            if bool(unit.get("IsContact")):
                continue
            side_id = unit.get("SideId") or unit.get("sideId")
            if side_id and str(side_id) != "00000000-0000-0000-0000-000000000000":
                return str(side_id)
        return ""

    def _encode_unit(
        self, unit: Mapping[str, Any], own_side_id: str
    ) -> EncodedEntity:
        entity_id = str(unit.get("guid") or "").strip()
        if not entity_id:
            raise ValueError("UnitInfo 缺少必填字段 guid")

        contact_value = unit.get("contactGuid")
        contact_id = str(contact_value).strip() if contact_value else None
        side_id = str(unit.get("SideId") or unit.get("sideId") or "")
        is_contact = bool(unit.get("IsContact"))
        is_own = side_id == own_side_id and not is_contact
        is_weapon = bool(unit.get("IsWeapon"))
        unit_type = _integer(unit.get("unitType"))
        unit_category = _integer(unit.get("unitCategory"))
        altitude_m = _number(unit.get("altitude"))
        icon_2d = str(unit.get("Icon2D") or "")
        domain = self._infer_domain(
            unit_category=unit_category,
            unit_type=unit_type,
            is_weapon=is_weapon,
            altitude_m=altitude_m,
            icon_2d=icon_2d,
        )

        longitude = _number(unit.get("longitude"))
        latitude = _number(unit.get("latitude"))
        heading_deg = _number(unit.get("heading"))
        speed = _number(unit.get("Speed"), 0.0)
        health_pct = _number(unit.get("BloodAmount"), 100.0)
        range_detect_km = _number(unit.get("rangeDetect"), 0.0)
        range_strike_air_km = _number(unit.get("rangeStrike_Air"), 0.0)
        range_strike_surface_km = _number(unit.get("rangeStrike_Surface"), 0.0)
        range_strike_land_km = _number(unit.get("rangeStrike_Land"), 0.0)
        range_strike_submarine_km = _number(
            unit.get("rangeStrike_Submarine"), 0.0
        )
        range_fly_max_km = _number(unit.get("range_FlyMax"), 0.0)

        comm_text = str(unit.get("CommText") or "")
        jam_text = str(unit.get("JammText") or "")
        communication_ok = "通信中断" not in comm_text
        radar_jammed = "被干扰" in jam_text
        commandable = is_own and health_pct > 0.0

        vector = np.asarray(
            [
                float(is_own),
                float(is_contact),
                float(is_weapon),
                float(unit_type),
                float(unit_category),
                float(domain),
                longitude,
                latitude,
                altitude_m,
                heading_deg,
                speed,
                health_pct,
                range_detect_km,
                range_strike_air_km,
                range_strike_surface_km,
                range_strike_land_km,
                range_strike_submarine_km,
                range_fly_max_km,
                float(communication_ok),
                float(radar_jammed),
                float(commandable),
            ],
            dtype=np.float32,
        )

        return EncodedEntity(
            entity_id=entity_id,
            contact_id=contact_id,
            name=str(unit.get("name") or unit.get("unitname") or entity_id),
            side_id=side_id,
            is_own=is_own,
            is_contact=is_contact,
            is_weapon=is_weapon,
            unit_type=unit_type,
            unit_category=unit_category,
            domain=domain,
            longitude=longitude,
            latitude=latitude,
            altitude_m=altitude_m,
            heading_deg=heading_deg,
            speed=speed,
            health_pct=health_pct,
            range_detect_km=range_detect_km,
            range_strike_air_km=range_strike_air_km,
            range_strike_surface_km=range_strike_surface_km,
            range_strike_land_km=range_strike_land_km,
            range_strike_submarine_km=range_strike_submarine_km,
            range_fly_max_km=range_fly_max_km,
            communication_ok=communication_ok,
            radar_jammed=radar_jammed,
            commandable=commandable,
            icon_2d=icon_2d,
            vector=vector,
        )

    @staticmethod
    def _infer_domain(
        unit_category: int,
        unit_type: int,
        is_weapon: bool,
        altitude_m: float,
        icon_2d: str,
    ) -> TargetDomain:
        # 我方视角样例中接触目标的 unitCategory/unitType 可能统一为 0，
        # 因此先用同一 UnitInfo 中的 Icon2D，再使用枚举，最后用高度兜底。
        icon = icon_2d.replace("\\", "/").lower()
        if "/aircraft/" in icon:
            return TargetDomain.AIR
        if "/ship/" in icon:
            return TargetDomain.SURFACE
        if "/submarine/" in icon:
            return TargetDomain.SUBMARINE
        if "/facility/" in icon or "/land/" in icon:
            return TargetDomain.LAND

        if not is_weapon:
            if unit_category == 0 or unit_type == 0:
                return TargetDomain.AIR
            if unit_category == 1 or unit_type in (1, 7):
                return TargetDomain.SURFACE
            if unit_category == 2 or unit_type == 2:
                return TargetDomain.SUBMARINE
            if unit_category in (3, 7) or unit_type in (3, 4, 5, 9, 10, 12):
                return TargetDomain.LAND

        if altitude_m < -1.0:
            return TargetDomain.SUBMARINE
        if altitude_m > 50.0:
            return TargetDomain.AIR
        if altitude_m >= -1.0:
            return TargetDomain.SURFACE
        return TargetDomain.UNKNOWN


PathLike = Union[str, Path]


def load_situation(path: PathLike) -> Dict[str, Any]:
    """从 UTF-8 JSON 文件读取完整态势响应。"""

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("态势文件顶层必须是 JSON 对象")
    return payload
