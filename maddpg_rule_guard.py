"""MADDPG 的确定性规则保护层。

神经网络仍负责产生动作概率；本模块只负责把攻击、规避和浮标动作约束到
``symbolic_reasoning`` 已验收的业务口径。这里不导入任何 protobuf 或 RPC 模块，
因此规则可以离线单元测试，也不会与根目录的旧版 protobuf 描述符冲突。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple


NAUTICAL_MILE_M = 1852.0
AIM_TOLERANCE_DEG = 30.0
MISSILE_EVADE_DISTANCE_M = 5_000.0
MISSILE_INTERCEPT_PRIORITY_DISTANCE_M = 50_000.0
ASW_TORPEDO_RELEASE_DISTANCE_M = 0.4 * NAUTICAL_MILE_M
MAX_ATTACKERS_PER_TARGET = 3
MAX_INTERCEPTORS_PER_MISSILE = 4
DEFAULT_ATTACK_QUANTITY = 2

DOMAIN_AIR = "air"
DOMAIN_SURFACE = "surface"
DOMAIN_SUBMARINE = "submarine"
DOMAIN_UNKNOWN = "unknown"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _target_domain(
    entity_type: int, contact_type: int, altitude_m: float
) -> str:
    if entity_type == 0:
        return DOMAIN_AIR
    if entity_type == 1:
        return DOMAIN_SURFACE
    if entity_type == 2:
        return DOMAIN_SUBMARINE
    if entity_type == 4:
        # 水下鱼雷/水下诱饵按水下目标处理；其他武器均按空中导弹处理。
        if contact_type in (3, 9, 16) or altitude_m < 0.0:
            return DOMAIN_SUBMARINE
        return DOMAIN_AIR
    return DOMAIN_UNKNOWN


def _weapon_count(own_raw: Mapping[str, Any], domain: str) -> int:
    inventory = own_raw.get("weaponNumber")
    if not isinstance(inventory, Mapping):
        inventory = {}
    key = {
        DOMAIN_AIR: "airNum",
        DOMAIN_SURFACE: "shipNum",
        DOMAIN_SUBMARINE: "subNum",
    }.get(domain)
    return max(0, _integer(inventory.get(key), 0)) if key else 0


def _maximum_range_m(own_raw: Mapping[str, Any], domain: str) -> float:
    ranges = own_raw.get("maxRange")
    if not isinstance(ranges, Mapping):
        ranges = {}
    key = {
        DOMAIN_AIR: "maxAir",
        DOMAIN_SURFACE: "maxSurface",
        DOMAIN_SUBMARINE: "maxSubsurface",
    }.get(domain)
    # 旧 MADDPG 态势接口的 maxRange 单位为海里。
    return max(0.0, _number(ranges.get(key), 0.0) * NAUTICAL_MILE_M) if key else 0.0


def _is_asw_aircraft(own_raw: Mapping[str, Any], own_type: int) -> bool:
    if own_type != 0:
        return False
    category = own_raw.get("unitCategory")
    category_id = _integer(category)
    text = " ".join(
        str(own_raw.get(key) or "")
        for key in ("unitCategory", "unitName", "name", "mdlName")
    ).casefold()
    explicit = category_id in (13, 14) or any(
        marker in text
        for marker in ("aircraft_asw", "aircraft_mpa", "反潜", "asw", "mpa", "直-8j", "直8j", "直-9c", "直9c")
    )
    return explicit and _weapon_count(own_raw, DOMAIN_SUBMARINE) > 0


def _target_type_allowed(
    own_raw: Mapping[str, Any], contact_type: int, domain: str
) -> bool:
    configured = own_raw.get("unitTarget")
    if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
        return True
    normalized = {_integer(value) for value in configured}
    if not normalized or contact_type in normalized:
        return True
    # 兼容旧接口的 0/1/2 域编号和 ContactType 的 0/1/2/3 编号。
    domain_aliases = {
        DOMAIN_AIR: {0, 1, 13},
        DOMAIN_SURFACE: {1, 2, 14},
        DOMAIN_SUBMARINE: {2, 3, 9, 16},
    }
    return bool(normalized & domain_aliases.get(domain, set()))


def _weapon_rule_id(domain: str) -> str:
    return {
        DOMAIN_AIR: "R-WPN-001",
        DOMAIN_SURFACE: "R-WPN-002",
        DOMAIN_SUBMARINE: "R-WPN-003",
    }.get(domain, "R-WPN-000")


def _bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(
        math.radians, (lon1, lat1, lon2, lat2)
    )
    delta_lon = lon2_rad - lon1_rad
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    )
    return math.degrees(math.atan2(x, y)) % 360.0


def geo_distance_m(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    altitude_difference_m: float = 0.0,
) -> float:
    radius_m = 6_371_000.0
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(
        math.radians, (lon1, lat1, lon2, lat2)
    )
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2.0) ** 2
    )
    horizontal_m = radius_m * 2.0 * math.asin(math.sqrt(haversine))
    return math.hypot(horizontal_m, altitude_difference_m)


@dataclass(frozen=True)
class AttackTargetInput:
    index: int
    target_id: str
    raw: Mapping[str, Any]
    entity_type: int
    altitude_m: float
    longitude: float
    latitude: float


@dataclass(frozen=True)
class AttackCandidate:
    index: int
    target_id: str
    domain: str
    contact_type: int
    distance_m: float
    maximum_range_m: float
    heading_difference_deg: float
    aimed_at_target: bool
    target_type_allowed: bool
    weapon_count: int
    within_attack_range: bool
    is_missile: bool
    is_asw_release: bool
    longitude: float
    latitude: float
    altitude_m: float

    @property
    def eligible(self) -> bool:
        return (
            self.domain != DOMAIN_UNKNOWN
            and self.target_type_allowed
            and self.weapon_count > 0
            and self.maximum_range_m > 0.0
        )


@dataclass(frozen=True)
class AttackRuleDecision:
    conclusion: str
    rule_id: str
    reason: str
    candidate: Optional[AttackCandidate] = None
    quantity: int = 0


def _evaluate_candidate(
    own_raw: Mapping[str, Any],
    own_type: int,
    own_altitude_m: float,
    own_heading_deg: float,
    own_longitude: float,
    own_latitude: float,
    target: AttackTargetInput,
) -> AttackCandidate:
    contact_type = _integer(target.raw.get("contactType"))
    domain = _target_domain(target.entity_type, contact_type, target.altitude_m)
    maximum_range_m = _maximum_range_m(own_raw, domain)
    is_asw_release = _is_asw_aircraft(own_raw, own_type) and domain == DOMAIN_SUBMARINE
    if is_asw_release and maximum_range_m > 0.0:
        maximum_range_m = min(maximum_range_m, ASW_TORPEDO_RELEASE_DISTANCE_M)

    distance_m = geo_distance_m(
        own_longitude,
        own_latitude,
        target.longitude,
        target.latitude,
        abs(own_altitude_m - target.altitude_m),
    )
    bearing = _bearing_deg(
        own_longitude, own_latitude, target.longitude, target.latitude
    )
    heading_difference = abs(
        (float(own_heading_deg) - bearing + 180.0) % 360.0 - 180.0
    )
    aimed = own_type == 1 or heading_difference < AIM_TOLERANCE_DEG
    is_missile = (
        target.entity_type == 4
        and domain == DOMAIN_AIR
        and contact_type in (-1, 0, 1, 11, 13)
    )
    return AttackCandidate(
        index=target.index,
        target_id=target.target_id,
        domain=domain,
        contact_type=contact_type,
        distance_m=distance_m,
        maximum_range_m=maximum_range_m,
        heading_difference_deg=heading_difference,
        aimed_at_target=aimed,
        target_type_allowed=_target_type_allowed(own_raw, contact_type, domain),
        weapon_count=_weapon_count(own_raw, domain),
        within_attack_range=(
            maximum_range_m > 0.0 and distance_m <= maximum_range_m
        ),
        is_missile=is_missile,
        is_asw_release=is_asw_release,
        longitude=target.longitude,
        latitude=target.latitude,
        altitude_m=target.altitude_m,
    )


def decide_attack(
    own_raw: Mapping[str, Any],
    own_type: int,
    own_altitude_m: float,
    own_heading_deg: float,
    own_longitude: float,
    own_latitude: float,
    targets: Iterable[AttackTargetInput],
) -> AttackRuleDecision:
    """按符号推理同口径选择目标并给出可验证结论。"""

    candidates = [
        _evaluate_candidate(
            own_raw,
            own_type,
            own_altitude_m,
            own_heading_deg,
            own_longitude,
            own_latitude,
            target,
        )
        for target in targets
    ]
    candidates.sort(key=lambda item: (item.distance_m, item.target_id))
    eligible = [item for item in candidates if item.eligible]

    priority_missiles = [
        item
        for item in eligible
        if item.is_missile
        and item.within_attack_range
        and item.distance_m <= MISSILE_INTERCEPT_PRIORITY_DISTANCE_M
    ]
    immediate = [item for item in eligible if item.within_attack_range]
    pursuit = [item for item in eligible if not item.within_attack_range]

    selected: Optional[AttackCandidate]
    if priority_missiles:
        selected = priority_missiles[0]
    elif immediate:
        selected = immediate[0]
    elif own_type == 0 and pursuit:
        selected = pursuit[0]
    elif eligible:
        selected = eligible[0]
    else:
        selected = candidates[0] if candidates else None

    if selected is None:
        return AttackRuleDecision("HOLD", "R-TGT-001", "没有可评估的敌方目标")
    if not selected.eligible:
        return AttackRuleDecision(
            "HOLD",
            _weapon_rule_id(selected.domain),
            "目标类型、射程能力或对应弹药不满足",
            selected,
        )
    if not selected.within_attack_range:
        if own_type == 0:
            return AttackRuleDecision(
                "CHASE_TO_RANGE",
                "R-ASW-001" if selected.is_asw_release else "R-RNG-004",
                "飞机位于允许发射距离之外，先追击并在进入射程后重新判断",
                selected,
            )
        return AttackRuleDecision(
            "HOLD", "R-RNG-003", "非飞机平台超距时禁止攻击和追击", selected
        )
    if not selected.aimed_at_target:
        if own_type == 0:
            return AttackRuleDecision(
                "CHASE_AND_ALIGN",
                "R-AIM-002",
                "飞机未满足严格小于30度的朝向条件，先转向目标",
                selected,
            )
        return AttackRuleDecision(
            "HOLD", "R-AIM-002", "平台未满足严格小于30度的朝向条件", selected
        )
    return AttackRuleDecision(
        "REQUEST_ATTACK",
        _weapon_rule_id(selected.domain),
        "目标、射程、朝向和弹药条件均满足",
        selected,
        quantity=DEFAULT_ATTACK_QUANTITY,
    )


class AttackThrottle:
    """跨帧攻击槽位和导弹累计拦截弹数量管理。"""

    def __init__(
        self,
        cooldown_frames: int = 600,
        max_parallel: int = MAX_ATTACKERS_PER_TARGET,
        max_interceptors: int = MAX_INTERCEPTORS_PER_MISSILE,
    ) -> None:
        if cooldown_frames <= 0 or max_parallel <= 0 or max_interceptors <= 0:
            raise ValueError("攻击节流参数必须为正数")
        self.cooldown_frames = int(cooldown_frames)
        self.max_parallel = int(max_parallel)
        self.max_interceptors = int(max_interceptors)
        self.table: Dict[str, Dict[str, int]] = {}
        self.interceptor_totals: Dict[str, int] = {}
        self._reservations: Dict[Tuple[str, str], Tuple[bool, int]] = {}

    def reset(self) -> None:
        self.table.clear()
        self.interceptor_totals.clear()
        self._reservations.clear()

    def tick(self, amount: int = 1) -> None:
        decrement = max(1, int(amount))
        for target_id in list(self.table):
            for attacker_id in list(self.table[target_id]):
                self.table[target_id][attacker_id] -= decrement
                if self.table[target_id][attacker_id] <= 0:
                    del self.table[target_id][attacker_id]
            if not self.table[target_id]:
                del self.table[target_id]

    def begin_frame(self, valid_target_ids: Iterable[str]) -> None:
        valid = {str(target_id) for target_id in valid_target_ids}
        for target_id in list(self.table):
            if target_id not in valid:
                del self.table[target_id]
        for target_id in list(self.interceptor_totals):
            if target_id not in valid:
                del self.interceptor_totals[target_id]
        self._reservations.clear()

    def can_attack(
        self,
        attacker_id: str,
        target_id: str,
        is_missile: bool = False,
        quantity: int = DEFAULT_ATTACK_QUANTITY,
    ) -> bool:
        attacker_id, target_id = str(attacker_id), str(target_id)
        if attacker_id in self.table.get(target_id, {}):
            return False
        if (attacker_id, target_id) in self._reservations:
            return False
        active = set(self.table.get(target_id, {}))
        reserved = {
            attacker
            for attacker, target in self._reservations
            if target == target_id
        }
        if len(active | reserved) >= self.max_parallel:
            return False
        if is_missile:
            planned = sum(
                planned_quantity
                for (attacker, target), (missile, planned_quantity) in self._reservations.items()
                if target == target_id and missile
            )
            remaining = self.max_interceptors - self.interceptor_totals.get(target_id, 0) - planned
            if remaining <= 0:
                return False
        return True

    def reserve(
        self,
        attacker_id: str,
        target_id: str,
        is_missile: bool = False,
        quantity: int = DEFAULT_ATTACK_QUANTITY,
    ) -> int:
        if not self.can_attack(attacker_id, target_id, is_missile, quantity):
            return 0
        accepted_quantity = max(1, int(quantity))
        if is_missile:
            planned = sum(
                planned_quantity
                for (attacker, target), (missile, planned_quantity) in self._reservations.items()
                if target == str(target_id) and missile
            )
            remaining = self.max_interceptors - self.interceptor_totals.get(str(target_id), 0) - planned
            accepted_quantity = min(accepted_quantity, remaining)
        self._reservations[(str(attacker_id), str(target_id))] = (
            bool(is_missile),
            accepted_quantity,
        )
        return accepted_quantity

    def commit(self, attacker_id: str, target_id: str, success: bool) -> None:
        key = (str(attacker_id), str(target_id))
        reservation = self._reservations.pop(key, None)
        if not success or reservation is None:
            return
        is_missile, quantity = reservation
        self.table.setdefault(key[1], {})[key[0]] = self.cooldown_frames
        if is_missile:
            self.interceptor_totals[key[1]] = min(
                self.max_interceptors,
                self.interceptor_totals.get(key[1], 0) + quantity,
            )


def missile_points_at_entity(
    missile_raw: Mapping[str, Any],
    own_id: str,
    missile_longitude: float,
    missile_latitude: float,
    own_longitude: float,
    own_latitude: float,
) -> bool:
    """确认导弹目标，缺少目标ID时以航向差严格小于30度推断。"""

    for key in ("weaponTargetId", "WeaponTargetId", "targetId", "targetGuid"):
        target_id = str(missile_raw.get(key) or "").strip()
        if target_id:
            return target_id == str(own_id)
    attitude = missile_raw.get("attitude")
    if not isinstance(attitude, Mapping) or "yaw" not in attitude:
        return False
    heading = _number(attitude.get("yaw"), float("nan"))
    if not math.isfinite(heading):
        return False
    bearing = _bearing_deg(
        missile_longitude,
        missile_latitude,
        own_longitude,
        own_latitude,
    )
    difference = abs((heading - bearing + 180.0) % 360.0 - 180.0)
    return difference < AIM_TOLERANCE_DEG


def point_in_polygon_with_boundary(
    point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]
) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        cross = (x - previous_x) * (current_y - previous_y) - (
            y - previous_y
        ) * (current_x - previous_x)
        if abs(cross) <= 1e-12 and min(previous_x, current_x) - 1e-12 <= x <= max(
            previous_x, current_x
        ) + 1e-12 and min(previous_y, current_y) - 1e-12 <= y <= max(
            previous_y, current_y
        ) + 1e-12:
            return True
        intersects = (current_y > y) != (previous_y > y)
        if intersects:
            x_intersection = (
                (previous_x - current_x)
                * (y - current_y)
                / (previous_y - current_y)
                + current_x
            )
            if x <= x_intersection:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def sonobuoy_deployment_allowed(
    own_raw: Mapping[str, Any],
    longitude: float,
    latitude: float,
    altitude_m: float,
    mission_area: Sequence[Tuple[float, float]],
) -> bool:
    inventory = own_raw.get("weaponNumber")
    if not isinstance(inventory, Mapping):
        inventory = {}
    buoy_count = max(
        0,
        _integer(
            inventory.get("buoyNum", inventory.get("sonobuoyNum", 0)),
            0,
        ),
    )
    category_text = str(own_raw.get("unitCategory") or "").casefold()
    is_patrol_aircraft = (
        "aircraft" in str(own_raw.get("mdlType") or "").casefold()
        and (
            _integer(own_raw.get("unitCategory")) in (13, 14)
            or "asw" in category_text
            or "mpa" in category_text
            or "反潜" in category_text
        )
    )
    return (
        is_patrol_aircraft
        and buoy_count > 0
        and 0.0 <= float(altitude_m) <= 500.0
        and point_in_polygon_with_boundary(
            (float(longitude), float(latitude)), mission_area
        )
    )
