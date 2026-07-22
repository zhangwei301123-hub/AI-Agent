"""把符号推理实时态势转换为旧 MADDPG 的 38 维实体输入格式。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from symbolic_reasoning.entity import EntityEncoder as SymbolicEntityEncoder
from symbolic_reasoning.entity import TargetDomain


KM_PER_NAUTICAL_MILE = 1.852


def _unit_list(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("实时态势缺少 data")
    send_msg = data.get("data")
    if not isinstance(send_msg, Mapping):
        raise ValueError("实时态势缺少 data.data")
    units = send_msg.get("UnitList")
    if not isinstance(units, list):
        raise ValueError("实时态势缺少 data.data.UnitList")
    return [unit for unit in units if isinstance(unit, Mapping)]


def _legacy_mdl_type(entity: Any) -> str:
    if entity.is_weapon:
        return "WEAPON"
    return {
        TargetDomain.AIR: "AIRCRAFT",
        TargetDomain.SURFACE: "SHIP",
        TargetDomain.SUBMARINE: "SUBMARINE",
        TargetDomain.LAND: "FACILITY",
    }.get(entity.domain, "UNKNOWN")


def _range_nm(range_km: float) -> float:
    return max(0.0, float(range_km)) / KM_PER_NAUTICAL_MILE


def _numeric_report_time(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        # 新接口可能返回 ISO-8601；旧 38 维编码只接受数值且该列不参与控制。
        return 0.0


def legacy_entities_from_symbolic_payload(
    payload: Mapping[str, Any],
    situation: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """返回可直接交给根目录 ``EntityEncoder`` 的实体列表。

    红方平台使用真实 ``guid``；蓝方接触目标优先使用 ``contactGuid``，保证
    ``GetWeaponFiringInfo`` 和 ``AttackTarget`` 收到的是红方视角的目标 ID。
    """

    source_units = _unit_list(payload)
    source_by_id = {
        str(unit.get("guid") or unit.get("mdlID") or ""): unit
        for unit in source_units
    }
    if situation is None:
        situation = SymbolicEntityEncoder().encode(payload)
    result: List[Dict[str, Any]] = []

    for entity in situation.entities:
        source = source_by_id.get(entity.entity_id, {})
        command_id = entity.command_id
        inventory = {
            "airNum": int(entity.weapon_count_air),
            "shipNum": int(entity.weapon_count_surface),
            "landNum": int(entity.weapon_count_land),
            "subNum": int(entity.weapon_count_submarine),
            "sonobuoyNum": int(entity.sonobuoy_count),
        }
        offensive = sum(
            inventory[key] for key in ("airNum", "shipNum", "landNum", "subNum")
        )
        fuel_pct = float(entity.fuel_percentage)
        oil_ratio = fuel_pct / 100.0 if fuel_pct >= 0.0 else 0.0
        oil_status = "IS_BINGO" if 0.0 <= fuel_pct <= 20.0 else "NONE"
        air_status = str(source.get("airStatus") or source.get("AirStatus") or "")
        unit_status = str(source.get("unitStatus") or source.get("UnitStatus") or "")

        result.append(
            {
                "mdlID": command_id,
                "entityGuid": entity.entity_id,
                "contactGuid": entity.contact_id,
                "name": entity.name,
                "mdlName": entity.name,
                "unitName": entity.unit_name,
                "forceSide": "红方" if entity.is_own else "蓝方",
                "mdlType": _legacy_mdl_type(entity),
                "activeLvl": float(entity.health_pct),
                "attitude": {"pitch": 0.0, "roll": 0.0, "yaw": entity.heading_deg},
                "entitySpatialCoord": {
                    "altitude": entity.altitude_m,
                    "latitude": entity.latitude,
                    "longitude": entity.longitude,
                },
                "velocity": {"vx": entity.speed, "vy": 0.0, "vz": 0.0},
                "attrMap": {"AirBase": str(source.get("assignedHost") or "")},
                "loadMap": {
                    "offensive": offensive,
                    "offenseless": int(entity.sonobuoy_count),
                },
                "logisticStates": {"oil": oil_ratio, "oilStatus": oil_status},
                "innerstates": {"IsJamReaction": int(entity.radar_jammed)},
                "stateMap": {
                    "FuelBurnRate": -1.0,
                    "RemainDistance": -1.0,
                    "UnitStatus": unit_status,
                    "AirStatus": air_status,
                    "EcmStatus": 0,
                    "RadarStatus": int(entity.has_radar_sensor),
                    "SonarStatus": int(entity.has_sonar_sensor),
                    "IdentifyStatus": 1,
                    "IsUnderAttack": 0,
                },
                # 根 MADDPG 的规则层按海里读取 maxRange。
                "maxRange": {
                    "maxAir": _range_nm(entity.range_strike_air_km),
                    "maxSurface": _range_nm(entity.range_strike_surface_km),
                    "maxLand": _range_nm(entity.range_strike_land_km),
                    "maxSubsurface": _range_nm(entity.range_strike_submarine_km),
                },
                "unitCategory": int(entity.unit_category),
                "contactType": int(entity.unit_specific_type),
                "unitTarget": [],
                "weaponNumber": inventory,
                "missionId": entity.mission_id,
                "missionType": source.get("missionType", 0),
                "reportTime": _numeric_report_time(
                    source.get("CurrentScenarioTime", source.get("reportTime", 0))
                ),
                "isCanManaged": bool(entity.commandable),
                "IsContact": bool(entity.is_contact),
                "IsWeapon": bool(entity.is_weapon),
                # 保留符号推理已经解析好的武器生命周期字段，供 MADDPG
                # 跟踪“发射平台 -> 在途武器 -> 目标”的完整链路。
                "WeaponImpact": int(entity.weapon_impact),
                "WeaponTargetDistance": float(entity.weapon_target_distance_km),
                "WeaponTargetName": entity.weapon_target_name,
                "WeaponTargetId": entity.weapon_target_id,
            }
        )

    return result
