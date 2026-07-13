"""符号规则所需的跨步状态：并发攻击槽位和累计拦截弹。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

from .entity import EncodedSituation


ATTACK_SLOT_TIMEOUT_SECONDS = 10.0 * 60.0
MAX_ATTACKERS_PER_TARGET = 3
MAX_INTERCEPTORS_PER_MISSILE = 4


@dataclass(frozen=True)
class AttackSlot:
    attacker_id: str
    target_id: str
    started_at: float
    target_is_missile: bool


class EngagementState:
    """保存一次运行期间不能只靠单帧态势得到的规则状态。"""

    def __init__(self, timeout_seconds: float = ATTACK_SLOT_TIMEOUT_SECONDS) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        self.timeout_seconds = float(timeout_seconds)
        self._slots: Dict[str, Dict[str, AttackSlot]] = {}
        self._interceptors_launched: Dict[str, int] = {}
        self._missile_last_seen: Dict[str, float] = {}
        self._target_aliases: Dict[str, str] = {}
        self._last_time: Optional[float] = None

    def reset(self) -> None:
        self._slots.clear()
        self._interceptors_launched.clear()
        self._missile_last_seen.clear()
        self._target_aliases.clear()
        self._last_time = None

    def update_from_situation(
        self, situation: EncodedSituation, current_time: float
    ) -> None:
        """处理想定重置、10 分钟超时、命中反馈和导弹目标消失。"""

        now = float(current_time)
        if self._last_time is not None and now + 1.0 < self._last_time:
            # 想定时间倒退视为重新开始，避免旧状态污染新想定。
            self.reset()
        self._last_time = now

        for target_id, slots in list(self._slots.items()):
            for attacker_id, slot in list(slots.items()):
                if now - slot.started_at >= self.timeout_seconds:
                    del slots[attacker_id]
            if not slots:
                del self._slots[target_id]

        target_ids: Set[str] = set()
        hit_target_ids: Set[str] = set(situation.deleted_entity_ids)
        for target in situation.targets:
            target_ids.add(target.command_id)
            target_ids.add(target.entity_id)
            if target.is_weapon:
                self._missile_last_seen[target.command_id] = now

        # 只有“我方武器实体发生动能命中并关联目标”才视为我方导弹命中。
        # 目标自身 WeaponImpact=2 无法说明攻击来源，不能据此提前释放槽位。
        for weapon in situation.entities:
            if (
                weapon.is_own
                and weapon.is_weapon
                and weapon.weapon_impact == 2
                and weapon.weapon_target_id
            ):
                hit_target_ids.add(weapon.weapon_target_id)
                target = situation.find_entity(weapon.weapon_target_id)
                if target is not None:
                    hit_target_ids.add(target.command_id)
                    hit_target_ids.add(target.entity_id)

        # 我方武器命中或目标被系统删除时，释放该目标的攻击槽位。
        for target_id in hit_target_ids:
            canonical_id = self._target_aliases.get(target_id, target_id)
            self.release_target(canonical_id)
            self._interceptors_launched.pop(canonical_id, None)
            self._missile_last_seen.pop(canonical_id, None)

        # “永久丢失”缺少单独字段，采用 10 分钟未再次出现作为保守清理条件。
        for target_id, last_seen in list(self._missile_last_seen.items()):
            if target_id not in target_ids and now - last_seen >= self.timeout_seconds:
                self._missile_last_seen.pop(target_id, None)
                self._interceptors_launched.pop(target_id, None)
                self.release_target(target_id)

    def active_attackers(self, target_id: str) -> int:
        return len(self._slots.get(target_id, {}))

    def attacker_ids(self, target_id: str) -> Set[str]:
        return set(self._slots.get(target_id, {}))

    def is_attacking(self, attacker_id: str, target_id: str) -> bool:
        return attacker_id in self._slots.get(target_id, {})

    def slot_available(
        self,
        attacker_id: str,
        target_id: str,
        planned_attackers: Iterable[str] = (),
    ) -> bool:
        active = self.attacker_ids(target_id)
        active.update(planned_attackers)
        if attacker_id in active:
            return False
        return len(active) < MAX_ATTACKERS_PER_TARGET

    def interceptors_launched(self, target_id: str) -> int:
        return self._interceptors_launched.get(target_id, 0)

    def interceptor_available(self, target_id: str, planned_count: int = 0) -> bool:
        return (
            self.interceptors_launched(target_id) + max(0, int(planned_count))
            < MAX_INTERCEPTORS_PER_MISSILE
        )

    def record_successful_attack(
        self,
        attacker_id: str,
        target_id: str,
        started_at: float,
        target_is_missile: bool,
        target_aliases: Sequence[Optional[str]] = (),
    ) -> None:
        slots = self._slots.setdefault(target_id, {})
        slots[attacker_id] = AttackSlot(
            attacker_id=attacker_id,
            target_id=target_id,
            started_at=float(started_at),
            target_is_missile=bool(target_is_missile),
        )
        self._target_aliases[target_id] = target_id
        for alias in target_aliases:
            if alias:
                self._target_aliases[str(alias)] = target_id
        if target_is_missile:
            current = self._interceptors_launched.get(target_id, 0)
            self._interceptors_launched[target_id] = min(
                MAX_INTERCEPTORS_PER_MISSILE, current + 1
            )
            self._missile_last_seen[target_id] = float(started_at)

    def release_target(self, target_id: str) -> None:
        canonical_id = self._target_aliases.get(target_id, target_id)
        self._slots.pop(canonical_id, None)
        aliases = [
            alias
            for alias, canonical in self._target_aliases.items()
            if canonical == canonical_id
        ]
        for alias in aliases:
            self._target_aliases.pop(alias, None)

    @property
    def slots(self) -> Mapping[str, Mapping[str, AttackSlot]]:
        return self._slots
