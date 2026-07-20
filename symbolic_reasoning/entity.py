"""对三维态势 SendMsg.UnitList 做字段校验、关联解析和数值编码。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np


class TargetDomain(IntEnum):
    """目标域，同时作为平台类型的统一编码。"""

    AIR = 0
    SURFACE = 1
    SUBMARINE = 2
    LAND = 3
    UNKNOWN = 4


OWN_SIDE_NAME = "红方"
ENEMY_SIDE_NAME = "蓝方"


# 字段来自 source/态势数据字段说明.md；可选库存/任务字段兼容现有运行数据。
FEATURE_NAMES: Tuple[str, ...] = (
    "is_own",
    "is_enemy",
    "is_contact",
    "is_weapon",
    "unit_type",
    "unit_category",
    "unit_specific_type",
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
    "weapon_impact",
    "weapon_target_distance_km",
    "weapon_count_air",
    "weapon_count_surface",
    "weapon_count_land",
    "weapon_count_submarine",
    "sonobuoy_count",
    "has_radar_sensor",
    "has_sonar_sensor",
    "fuel_percentage",
    "is_airborne",
    "is_parked",
    "has_strike_weapon_system",
    "strike_weapon_count",
    "communication_ok",
    "radar_jammed",
    "commandable",
    "has_patrol_mission",
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


def _contact_age_bucket(value: Any) -> str:
    """把持续变化的探测秒数压缩为质量等级，避免每秒解除攻击冷却。"""

    text = str(value or "").strip()
    if not text:
        return "unknown"
    digits = "".join(character for character in text if character.isdigit())
    amount = int(digits) if digits else -1
    if "秒" in text and amount >= 0:
        if amount <= 2:
            return "fresh"
        if amount <= 10:
            return "recent"
        return "stale"
    if "分" in text or "时" in text or "天" in text:
        return "stale"
    return text.casefold()


def _normalize_contact_quality(value: Any) -> Any:
    """抑制不确定区坐标的浮点噪声，只保留有意义的质量变化。"""

    if isinstance(value, Mapping):
        return {
            str(key): _normalize_contact_quality(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_contact_quality(item) for item in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def _contact_quality_signature(
    unit: Mapping[str, Any],
    entity_id: str,
    contact_id: Optional[str],
    is_contact: bool,
) -> str:
    """生成只随 Contact 身份或探测质量变化的稳定签名。"""

    if not is_contact:
        return ""
    quality = {
        "entity_id": entity_id,
        "contact_id": contact_id or "",
        "age_bucket": _contact_age_bucket(
            unit.get("ContactLastDetectTimeStr")
        ),
        "uncertainty": _normalize_contact_quality(
            unit.get("UncertainAreaList")
        ),
        "contact_type": unit.get("contactType", unit.get("ContactType")),
        "identify_status": unit.get(
            "IdentifyStatus", unit.get("IdentificationStatus")
        ),
    }
    serialized = json.dumps(
        quality,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _boolean(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("false", "0", "no", "off", "否"):
            return False
        if normalized in ("true", "1", "yes", "on", "是"):
            return True
    return bool(value)


def _canonical_side_name(unit: Mapping[str, Any]) -> str:
    for key in ("forceSide", "side", "SideName", "sideName"):
        value = str(unit.get(key) or "").strip()
        normalized = value.lower().replace(" ", "")
        if value == OWN_SIDE_NAME or normalized in ("red", "redside"):
            return OWN_SIDE_NAME
        if value == ENEMY_SIDE_NAME or normalized in ("blue", "blueside"):
            return ENEMY_SIDE_NAME
    return ""


def _first_number(mapping: Mapping[str, Any], names: Sequence[str]) -> int:
    for name in names:
        if name in mapping:
            return max(0, _integer(mapping.get(name), 0))
    return 0


def _has_sensor_capability(
    unit: Mapping[str, Any],
    explicit_names: Sequence[str],
    range_names: Sequence[str],
) -> bool:
    """Read an explicit capability flag, with a legacy range-list fallback."""

    for name in explicit_names:
        if name in unit:
            return _boolean(unit.get(name))

    # Historical text snapshots do not contain GetUnitData sensor inventory.
    # A non-empty sensor range list is the only available capability hint there.
    for name in range_names:
        value = unit.get(name)
        if isinstance(value, Mapping) and value:
            return True
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) > 0:
                return True
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized not in ("", "[]", "{}", "null", "none"):
                return True
        if isinstance(value, (int, float)) and _number(value, 0.0) > 0.0:
            return True
    return False


def _aircraft_operating_state(
    unit: Mapping[str, Any],
    is_aircraft: bool,
    altitude_m: float,
    speed: float,
) -> Tuple[bool, bool]:
    """Return ``(is_airborne, is_parked)`` without guessing unknown states."""

    if not is_aircraft:
        return False, False
    if "isAirborne" in unit or "is_airborne" in unit:
        airborne = _boolean(unit.get("isAirborne", unit.get("is_airborne")))
        parked = _boolean(unit.get("isParked", unit.get("is_parked")))
        return airborne, parked and not airborne

    status = " ".join(
        str(unit.get(name) or "")
        for name in ("airStatus", "AirStatus", "unitStatus", "UnitStatus")
    ).casefold()
    parked_markers = (
        "停放",
        "停机",
        "已降落",
        "着陆",
        "在基地",
        "parked",
        "on ground",
        "grounded",
    )
    airborne_markers = (
        "在空",
        "空中",
        "飞行",
        "返航",
        "airborne",
        "in flight",
        "on mission",
        "returning to base",
    )
    if any(marker in status for marker in parked_markers):
        return False, True
    if any(marker in status for marker in airborne_markers):
        return True, False

    # Status text is not present in historical snapshots.  Use conservative
    # kinematic boundaries and leave ambiguous aircraft in neither state.
    if altitude_m > 10.0 or speed >= 30.0:
        return True, False
    if -5.0 <= altitude_m <= 10.0 and 0.0 <= speed <= 5.0:
        return False, True
    return False, False


@dataclass(frozen=True)
class EncodedEntity:
    """保留符号推理所需的实体字段和固定顺序编码。"""

    entity_id: str
    contact_id: Optional[str]
    name: str
    unit_name: str
    side_id: str
    side_name: str
    is_own: bool
    is_enemy: bool
    is_contact: bool
    is_weapon: bool
    unit_type: int
    unit_category: int
    unit_specific_type: int
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
    weapon_impact: int
    weapon_target_distance_km: float
    weapon_target_name: str
    weapon_target_id: Optional[str]
    weapon_count_air: int
    weapon_count_surface: int
    weapon_count_land: int
    weapon_count_submarine: int
    sonobuoy_count: int
    has_radar_sensor: bool
    has_sonar_sensor: bool
    fuel_percentage: float
    is_airborne: bool
    is_parked: bool
    has_strike_weapon_system: bool
    strike_weapon_count: int
    mission_id: str
    has_patrol_mission: bool
    communication_ok: bool
    radar_jammed: bool
    commandable: bool
    icon_2d: str
    vector: np.ndarray
    contact_quality_signature: str = ""

    @property
    def command_id(self) -> str:
        """己方命令用 guid，接触目标命令优先使用 contactGuid。"""

        if self.is_contact and self.contact_id:
            return self.contact_id
        return self.entity_id

    @property
    def is_aircraft(self) -> bool:
        return self.domain is TargetDomain.AIR and not self.is_weapon

    @property
    def is_patrol_aircraft(self) -> bool:
        if not self.is_aircraft:
            return False
        text = " ".join((self.name, self.unit_name, self.icon_2d)).lower()
        return self.has_patrol_mission or "巡逻" in text or "patrol" in text

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

    def weapon_count_for(self, target: "EncodedEntity") -> int:
        if target.domain is TargetDomain.AIR:
            return self.weapon_count_air
        if target.domain is TargetDomain.SURFACE:
            return self.weapon_count_surface
        if target.domain is TargetDomain.SUBMARINE:
            return self.weapon_count_submarine
        if target.domain is TargetDomain.LAND:
            return self.weapon_count_land
        return 0

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
        ground_km = 2.0 * radius_km * math.asin(
            min(1.0, math.sqrt(max(0.0, value)))
        )
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
    deleted_entity_ids: Tuple[str, ...] = ()

    @property
    def own_entities(self) -> Tuple[EncodedEntity, ...]:
        return tuple(entity for entity in self.entities if entity.commandable)

    @property
    def targets(self) -> Tuple[EncodedEntity, ...]:
        return tuple(
            entity
            for entity in self.entities
            # 接触目标的 BloodAmount 可能因通信不足为 0，不能据此丢弃。
            if entity.is_enemy and (entity.is_contact or entity.health_pct != 0)
        )

    def find_entity(self, entity_id: str) -> Optional[EncodedEntity]:
        for entity in self.entities:
            if entity_id in (entity.entity_id, entity.command_id):
                return entity
        return None


class EntityEncoder:
    """编码我方视角完整态势，并解析武器目标链路。"""

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

        red_side_id = self._find_side_id(units, OWN_SIDE_NAME)
        if red_side_id:
            own_side_id = red_side_id
        elif not own_side_id:
            own_side_id = self._infer_own_side_id(units)
        if not own_side_id:
            raise ValueError("态势数据缺少 sideGuid，且无法从己方实体推断")

        target_links = self._weapon_target_links(send_msg, units)
        encoded_data = np.full(
            (self.max_entities, self.feature_dim), -1.0, dtype=np.float32
        )
        mask = np.zeros(self.max_entities, dtype=np.bool_)
        entities: List[EncodedEntity] = []

        for index, unit in enumerate(units[: self.max_entities]):
            if not isinstance(unit, Mapping):
                raise ValueError("UnitList[{}] 不是对象".format(index))
            entity = self._encode_unit(unit, own_side_id, target_links)
            entities.append(entity)
            encoded_data[index] = entity.vector
            mask[index] = True

        current_time = str(send_msg.get("CurrentTime") or "")
        if not current_time and units:
            current_time = str(units[0].get("CurrentScenarioTime") or "")

        return EncodedSituation(
            own_side_id=own_side_id,
            scene_name=str(send_msg.get("ScenName") or ""),
            current_time=current_time,
            entities=tuple(entities),
            encoded_data=encoded_data,
            mask=mask,
            deleted_entity_ids=self._deleted_entity_ids(send_msg),
        )

    @staticmethod
    def _extract_send_msg(
        payload: Mapping[str, Any]
    ) -> Tuple[Mapping[str, Any], str]:
        if not isinstance(payload, Mapping):
            raise ValueError("态势数据必须是 JSON 对象")

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

        if "UnitList" in payload:
            side_id = str(payload.get("SideGuid") or payload.get("sideGuid") or "")
            return payload, side_id

        raise ValueError("不支持的态势结构，期望完整响应 data.data.UnitList 或 SendMsg")

    @staticmethod
    def _find_side_id(
        units: Sequence[Mapping[str, Any]], side_name: str
    ) -> str:
        for unit in units:
            if _canonical_side_name(unit) != side_name:
                continue
            side_id = unit.get("SideId") or unit.get("sideId")
            return str(side_id) if side_id else side_name
        return ""

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
        self,
        unit: Mapping[str, Any],
        own_side_id: str,
        target_links: Mapping[str, str],
    ) -> EncodedEntity:
        entity_id = str(unit.get("guid") or unit.get("mdlID") or "").strip()
        if not entity_id:
            raise ValueError("UnitInfo 缺少必填字段 guid/mdlID")

        contact_value = unit.get("contactGuid")
        contact_id = str(contact_value).strip() if contact_value else None
        side_name = _canonical_side_name(unit)
        side_id = str(
            unit.get("SideId") or unit.get("sideId") or side_name or ""
        )
        is_contact = _boolean(unit.get("IsContact"))
        if side_name:
            is_own = side_name == OWN_SIDE_NAME and not is_contact
            is_enemy = side_name == ENEMY_SIDE_NAME
        elif side_id:
            is_own = side_id == own_side_id and not is_contact
            is_enemy = not is_own
        else:
            # 未知阵营不能被当作敌方自动攻击。
            is_own = False
            is_enemy = False
        is_weapon = _boolean(unit.get("IsWeapon"))
        unit_type = _integer(unit.get("unitType"))
        unit_category = _integer(unit.get("unitCategory"))
        unit_specific_type = _integer(unit.get("UnitSpecificType"))
        altitude_m = _number(unit.get("altitude"))
        icon_2d = str(unit.get("Icon2D") or "")
        domain = self._infer_domain(
            unit_category=unit_category,
            unit_type=unit_type,
            is_weapon=is_weapon,
            altitude_m=altitude_m,
            icon_2d=icon_2d,
            unit_model=str(unit.get("unitModel") or ""),
        )

        longitude = _number(unit.get("longitude"))
        latitude = _number(unit.get("latitude"))
        heading_deg = _number(unit.get("heading"), 0.0) % 360.0
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

        inventory = unit.get("weaponNumber")
        if not isinstance(inventory, Mapping):
            inventory = {}
        weapon_count_air = _first_number(
            inventory, ("airNum", "AirNum", "airCount", "air")
        )
        weapon_count_surface = _first_number(
            inventory, ("shipNum", "surfaceNum", "ShipNum", "ship")
        )
        weapon_count_land = _first_number(
            inventory, ("landNum", "LandNum", "landCount", "land")
        )
        weapon_count_submarine = _first_number(
            inventory, ("subNum", "submarineNum", "SubNum", "sub")
        )
        sonobuoy_count = _first_number(
            inventory,
            ("sonobuoyNum", "sonarBuoyNum", "buoyNum", "sonobuoy", "buoy"),
        )
        if sonobuoy_count == 0:
            sonobuoy_count = _first_number(
                unit,
                ("sonobuoyNum", "sonarBuoyNum", "buoyNum", "sonobuoyCount"),
            )

        has_radar_sensor = _has_sensor_capability(
            unit,
            ("hasRadarSensor", "has_radar_sensor", "radarAvailable"),
            ("rangeSensor_Air", "rangeSensor_Sea"),
        )
        has_sonar_sensor = _has_sensor_capability(
            unit,
            ("hasSonarSensor", "has_sonar_sensor", "sonarAvailable"),
            ("rangeSensor_UnderWater",),
        )
        fuel_percentage = _number(
            unit.get(
                "fuelPercentage",
                unit.get("fuelPct", unit.get("oil", -1.0)),
            ),
            -1.0,
        )
        is_aircraft = domain is TargetDomain.AIR and not is_weapon
        is_airborne, is_parked = _aircraft_operating_state(
            unit, is_aircraft, altitude_m, speed
        )
        explicit_strike_capability = unit.get("hasStrikeWeaponSystem")
        if explicit_strike_capability is None:
            has_strike_weapon_system = any(
                (
                    weapon_count_air > 0,
                    weapon_count_surface > 0,
                    weapon_count_land > 0,
                    weapon_count_submarine > 0,
                    range_strike_air_km > 0.0,
                    range_strike_surface_km > 0.0,
                    range_strike_land_km > 0.0,
                    range_strike_submarine_km > 0.0,
                )
            )
        else:
            has_strike_weapon_system = _boolean(explicit_strike_capability)
        strike_weapon_count = max(
            0,
            _integer(
                unit.get("strikeWeaponCount"),
                weapon_count_air
                + weapon_count_surface
                + weapon_count_land
                + weapon_count_submarine,
            ),
        )

        mission_id = str(unit.get("missionId") or unit.get("MissionId") or "")
        mission_text = " ".join(
            str(unit.get(key) or "")
            for key in ("missionType", "MissionType", "missionName", "MissionName")
        ).lower()
        has_patrol_mission = bool(mission_id) and (
            "巡逻" in mission_text or "patrol" in mission_text
        )

        comm_text = str(unit.get("CommText") or "")
        jam_text = str(unit.get("JammText") or "")
        communication_ok = "通信中断" not in comm_text
        radar_jammed = "被干扰" in jam_text
        # SituationDataw 的 proto3 bool 未填写时也会呈现 False。按项目口径，
        # 红方实体默认全部由符号智能体控制，不用该不可靠字段阻断命令。
        commandable = is_own and not is_weapon and health_pct > 0.0
        weapon_impact = max(0, _integer(unit.get("WeaponImpact"), 0))
        weapon_target_distance_km = max(
            0.0, _number(unit.get("WeaponTargetDistance"), 0.0)
        )

        vector = np.asarray(
            [
                float(is_own),
                float(is_enemy),
                float(is_contact),
                float(is_weapon),
                float(unit_type),
                float(unit_category),
                float(unit_specific_type),
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
                float(weapon_impact),
                weapon_target_distance_km,
                float(weapon_count_air),
                float(weapon_count_surface),
                float(weapon_count_land),
                float(weapon_count_submarine),
                float(sonobuoy_count),
                float(has_radar_sensor),
                float(has_sonar_sensor),
                fuel_percentage,
                float(is_airborne),
                float(is_parked),
                float(has_strike_weapon_system),
                float(strike_weapon_count),
                float(communication_ok),
                float(radar_jammed),
                float(commandable),
                float(has_patrol_mission),
            ],
            dtype=np.float32,
        )

        return EncodedEntity(
            entity_id=entity_id,
            contact_id=contact_id,
            name=str(unit.get("name") or unit.get("unitname") or entity_id),
            unit_name=str(unit.get("unitname") or ""),
            side_id=side_id,
            side_name=side_name,
            is_own=is_own,
            is_enemy=is_enemy,
            is_contact=is_contact,
            is_weapon=is_weapon,
            unit_type=unit_type,
            unit_category=unit_category,
            unit_specific_type=unit_specific_type,
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
            weapon_impact=weapon_impact,
            weapon_target_distance_km=weapon_target_distance_km,
            weapon_target_name=str(unit.get("WeaponTargetName") or ""),
            weapon_target_id=target_links.get(entity_id),
            weapon_count_air=weapon_count_air,
            weapon_count_surface=weapon_count_surface,
            weapon_count_land=weapon_count_land,
            weapon_count_submarine=weapon_count_submarine,
            sonobuoy_count=sonobuoy_count,
            has_radar_sensor=has_radar_sensor,
            has_sonar_sensor=has_sonar_sensor,
            fuel_percentage=fuel_percentage,
            is_airborne=is_airborne,
            is_parked=is_parked,
            has_strike_weapon_system=has_strike_weapon_system,
            strike_weapon_count=strike_weapon_count,
            mission_id=mission_id,
            has_patrol_mission=has_patrol_mission,
            communication_ok=communication_ok,
            radar_jammed=radar_jammed,
            commandable=commandable,
            icon_2d=icon_2d,
            vector=vector,
            contact_quality_signature=_contact_quality_signature(
                unit, entity_id, contact_id, is_contact
            ),
        )

    @staticmethod
    def _infer_domain(
        unit_category: int,
        unit_type: int,
        is_weapon: bool,
        altitude_m: float,
        icon_2d: str,
        unit_model: str = "",
    ) -> TargetDomain:
        icon = icon_2d.replace("\\", "/").lower()
        model_text = (icon + " " + unit_model).lower()
        if "torpedo" in model_text or "鱼雷" in model_text:
            return TargetDomain.SUBMARINE
        if "/aircraft/" in icon:
            return TargetDomain.AIR
        if "/ship/" in icon:
            return TargetDomain.SURFACE
        if "/submarine/" in icon:
            return TargetDomain.SUBMARINE
        if "/facility/" in icon or "/land/" in icon:
            return TargetDomain.LAND

        # 普通导弹均按 AIR 目标处理；水下鱼雷由图标/型号或负高度识别。
        if is_weapon:
            return (
                TargetDomain.SUBMARINE
                if altitude_m < -1.0
                else TargetDomain.AIR
            )

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

    @staticmethod
    def _weapon_target_links(
        send_msg: Mapping[str, Any], units: Sequence[Mapping[str, Any]]
    ) -> Dict[str, str]:
        links: Dict[str, str] = {}
        weapons = {
            str(unit.get("guid"))
            for unit in units
            if unit.get("guid") and bool(unit.get("IsWeapon"))
        }
        names: Dict[str, Set[str]] = {}
        for unit in units:
            unit_id = str(unit.get("guid") or "")
            for key in ("name", "unitname"):
                name = str(unit.get(key) or "").strip()
                if name and unit_id:
                    names.setdefault(name, set()).add(unit_id)

        radiation = send_msg.get("radiationAndDataLinkLine")
        if not isinstance(radiation, Mapping):
            radiation = send_msg.get("RadiationAndDataLinkLine")
        if isinstance(radiation, Mapping):
            raw_links = radiation.get("WeaponTarget") or []
            if isinstance(raw_links, Sequence) and not isinstance(
                raw_links, (str, bytes)
            ):
                for link in raw_links:
                    if not isinstance(link, Mapping):
                        continue
                    endpoints = link.get("Arr") or link.get("arr") or []
                    if not isinstance(endpoints, Sequence):
                        continue
                    endpoint_ids = [
                        str(endpoint.get("unitguid") or endpoint.get("unitGuid") or "")
                        for endpoint in endpoints
                        if isinstance(endpoint, Mapping)
                    ]
                    weapon_id = next(
                        (item for item in endpoint_ids if item in weapons), None
                    )
                    target_id = next(
                        (
                            item
                            for item in endpoint_ids
                            if item and item != weapon_id
                        ),
                        None,
                    )
                    if weapon_id and target_id:
                        links[weapon_id] = target_id

        # 当链路没有目标 ID 时，唯一匹配的 WeaponTargetName 可作为强关联。
        for unit in units:
            weapon_id = str(unit.get("guid") or "")
            if weapon_id not in weapons or weapon_id in links:
                continue
            target_name = str(unit.get("WeaponTargetName") or "").strip()
            candidates = names.get(target_name, set())
            if len(candidates) == 1:
                links[weapon_id] = next(iter(candidates))
        return links

    @staticmethod
    def _deleted_entity_ids(send_msg: Mapping[str, Any]) -> Tuple[str, ...]:
        result: List[str] = []
        deleted = send_msg.get("DeleteUnitIdList") or []
        if not isinstance(deleted, Sequence) or isinstance(deleted, (str, bytes)):
            return ()
        for item in deleted:
            if isinstance(item, Mapping):
                value = item.get("guid") or item.get("contactGuid")
            else:
                value = item
            if value:
                result.append(str(value))
        return tuple(result)


PathLike = Union[str, Path]


def load_situation(path: PathLike) -> Dict[str, Any]:
    """从 UTF-8 JSON 文件读取完整态势响应。"""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("态势文件顶层必须是 JSON 对象")
    return payload
