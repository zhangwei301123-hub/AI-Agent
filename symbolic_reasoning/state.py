"""符号规则所需的跨步状态：并发攻击槽位和累计拦截弹。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from .entity import EncodedSituation


ATTACK_SLOT_TIMEOUT_FRAMES = 600
ATTACK_WEAPON_APPEARANCE_GRACE_FRAMES = 10
MAX_ATTACKERS_PER_TARGET = 3
MAX_INTERCEPTORS_PER_MISSILE = 4
LIFECYCLE_REQUEST_RETRY_FRAMES = 10
TARGET_CONTACT_LOSS_GRACE_FRAMES = 3
FIRE_CONTROL_REJECTION_COOLDOWN_FRAMES = 10


@dataclass(frozen=True)
class AttackSlot:
    attacker_id: str
    target_id: str
    started_frame: int
    target_is_missile: bool


@dataclass(frozen=True)
class FireControlRejection:
    """一次武器可发射性拒绝及其重试条件。"""

    attacker_id: str
    target_entity_id: str
    reason_key: str
    reason: str
    contact_quality_signature: str
    rejected_frame: int


class EngagementState:
    """保存一次运行期间不能只靠单帧态势得到的规则状态。"""

    def __init__(
        self,
        timeout_frames: int = ATTACK_SLOT_TIMEOUT_FRAMES,
        weapon_appearance_grace_frames: int = (
            ATTACK_WEAPON_APPEARANCE_GRACE_FRAMES
        ),
        target_contact_loss_grace_frames: int = (
            TARGET_CONTACT_LOSS_GRACE_FRAMES
        ),
        fire_control_rejection_cooldown_frames: int = (
            FIRE_CONTROL_REJECTION_COOLDOWN_FRAMES
        ),
    ) -> None:
        if timeout_frames <= 0:
            raise ValueError("timeout_frames 必须大于 0")
        if weapon_appearance_grace_frames <= 0:
            raise ValueError("weapon_appearance_grace_frames 必须大于 0")
        if target_contact_loss_grace_frames <= 0:
            raise ValueError("target_contact_loss_grace_frames 必须大于 0")
        if fire_control_rejection_cooldown_frames <= 0:
            raise ValueError(
                "fire_control_rejection_cooldown_frames 必须大于 0"
            )
        self.timeout_frames = int(timeout_frames)
        self.weapon_appearance_grace_frames = int(
            weapon_appearance_grace_frames
        )
        self.target_contact_loss_grace_frames = int(
            target_contact_loss_grace_frames
        )
        self.fire_control_rejection_cooldown_frames = int(
            fire_control_rejection_cooldown_frames
        )
        self._slots: Dict[str, Dict[str, AttackSlot]] = {}
        self._interceptors_launched: Dict[str, int] = {}
        self._missile_last_seen: Dict[str, int] = {}
        self._target_aliases: Dict[str, str] = {}
        self._targets_with_seen_weapon: Set[str] = set()
        self._takeoff_requests: Dict[str, int] = {}
        self._return_requests: Dict[str, int] = {}
        self._target_missing_frames: Dict[Tuple[str, str], int] = {}
        self._fire_control_rejections: Dict[
            Tuple[str, str], FireControlRejection
        ] = {}
        self._last_frame: Optional[int] = None

    def reset(self) -> None:
        self._slots.clear()
        self._interceptors_launched.clear()
        self._missile_last_seen.clear()
        self._target_aliases.clear()
        self._targets_with_seen_weapon.clear()
        self._takeoff_requests.clear()
        self._return_requests.clear()
        self._target_missing_frames.clear()
        self._fire_control_rejections.clear()
        self._last_frame = None

    def update_from_situation(
        self, situation: EncodedSituation, current_frame: int
    ) -> None:
        """按倍速折算后的冷却帧处理超时、命中和目标消失。"""

        frame = int(current_frame)
        if frame < 0:
            raise ValueError("current_frame 不能小于 0")
        if self._last_frame is not None and frame < self._last_frame:
            # 数据帧序号倒退视为重新开始，避免旧状态污染新想定。
            self.reset()
        self._last_frame = frame

        for requests in (self._takeoff_requests, self._return_requests):
            for entity_id, requested_frame in list(requests.items()):
                if frame - requested_frame >= LIFECYCLE_REQUEST_RETRY_FRAMES:
                    requests.pop(entity_id, None)

        for entity in situation.own_entities:
            if entity.is_airborne:
                self._takeoff_requests.pop(entity.command_id, None)
            if entity.is_parked:
                self._return_requests.pop(entity.command_id, None)

        for target_id, slots in list(self._slots.items()):
            for attacker_id, slot in list(slots.items()):
                if frame - slot.started_frame >= self.timeout_frames:
                    del slots[attacker_id]
            if not slots:
                self.release_target(target_id)

        target_ids: Set[str] = set()
        hit_target_ids: Set[str] = set(situation.deleted_entity_ids)
        for target in situation.targets:
            target_ids.add(target.command_id)
            target_ids.add(target.entity_id)
            # Contact GUID 在重新捕获后可能变化。只要稳定实体 GUID 已经被某个
            # 攻击槽位登记，就把新的 Contact GUID 也绑定到同一个规范目标。
            canonical_id = self._target_aliases.get(target.entity_id)
            if canonical_id:
                self._target_aliases[target.command_id] = canonical_id
            if target.is_weapon:
                self._missile_last_seen[target.command_id] = frame

        # 目标必须连续丢失若干帧才会使攻击条件失效；单帧雷达闪烁不会触发
        # cancelAttackw。稳定实体 GUID 可跨越 Contact GUID 的变化完成关联。
        visible_targets = tuple(situation.targets)
        for target_id, slots in self._slots.items():
            for attacker_id in slots:
                key = (attacker_id, target_id)
                visible = any(
                    self.same_target(
                        target_id, target.command_id, target.entity_id
                    )
                    for target in visible_targets
                )
                if visible:
                    self._target_missing_frames[key] = 0
                else:
                    self._target_missing_frames[key] = (
                        self._target_missing_frames.get(key, 0) + 1
                    )

        in_flight_target_ids: Set[str] = set()
        # 只有“我方武器实体发生动能命中并关联目标”才视为我方导弹命中。
        # 目标自身 WeaponImpact=2 无法说明攻击来源，不能据此提前释放槽位。
        for weapon in situation.entities:
            if (
                weapon.is_own
                and weapon.is_weapon
                and weapon.weapon_target_id
            ):
                weapon_target_id = weapon.weapon_target_id
                canonical_id = self._target_aliases.get(
                    weapon_target_id, weapon_target_id
                )
                if weapon.weapon_impact == 2:
                    hit_target_ids.add(weapon_target_id)
                    target = situation.find_entity(weapon_target_id)
                    if target is not None:
                        hit_target_ids.add(target.command_id)
                        hit_target_ids.add(target.entity_id)
                else:
                    in_flight_target_ids.add(canonical_id)

        # 在途武器一旦出现就开始跟踪；之后从态势中消失，视为已命中、被拦截
        # 或失效，立即释放并发槽位，允许再次攻击。累计拦截弹数量仍保留。
        self._targets_with_seen_weapon.update(in_flight_target_ids)
        for target_id, slots in list(self._slots.items()):
            canonical_id = self._target_aliases.get(target_id, target_id)
            if canonical_id in in_flight_target_ids:
                continue
            if canonical_id in self._targets_with_seen_weapon:
                self.release_target(canonical_id)
                continue

            # AttackTarget 成功后允许若干帧等待武器实体形成；超过宽限仍未
            # 观察到在途武器，认为发射未形成或已立即失效，释放对应攻击者。
            for attacker_id, slot in list(slots.items()):
                if (
                    frame - slot.started_frame
                    >= self.weapon_appearance_grace_frames
                ):
                    del slots[attacker_id]
            if not slots:
                self.release_target(target_id)

        # 我方武器命中或目标被系统删除时，释放该目标的攻击槽位。
        for target_id in hit_target_ids:
            canonical_id = self._target_aliases.get(target_id, target_id)
            self.release_target(canonical_id)
            self._interceptors_launched.pop(canonical_id, None)
            self._missile_last_seen.pop(canonical_id, None)
            self._targets_with_seen_weapon.discard(canonical_id)

        # “永久丢失”缺少单独字段，采用 600 帧未再次出现作为保守清理条件。
        for target_id, last_seen in list(self._missile_last_seen.items()):
            if target_id not in target_ids and frame - last_seen >= self.timeout_frames:
                self._missile_last_seen.pop(target_id, None)
                self._interceptors_launched.pop(target_id, None)
                self.release_target(target_id)

        # 长时间没有再次使用的拒绝记录只占内存、不再参与决策，保守清理。
        for key, rejection in list(self._fire_control_rejections.items()):
            if frame - rejection.rejected_frame >= self.timeout_frames:
                self._fire_control_rejections.pop(key, None)

    def canonical_target_id(self, target_id: str) -> str:
        return self._target_aliases.get(str(target_id), str(target_id))

    def same_target(self, target_id: str, *candidate_ids: Optional[str]) -> bool:
        canonical_id = self.canonical_target_id(target_id)
        return any(
            candidate is not None
            and self.canonical_target_id(str(candidate)) == canonical_id
            for candidate in candidate_ids
        )

    def target_missing_frames(self, attacker_id: str, target_id: str) -> int:
        canonical_id = self.canonical_target_id(target_id)
        return self._target_missing_frames.get(
            (str(attacker_id), canonical_id), 0
        )

    def fire_control_rejection(
        self,
        attacker_id: str,
        target_entity_id: str,
        contact_quality_signature: str,
        current_frame: int,
    ) -> Optional[FireControlRejection]:
        """返回仍在冷却中的同质量拒绝；质量变化会立即解除冷却。"""

        key = (str(attacker_id), str(target_entity_id))
        rejection = self._fire_control_rejections.get(key)
        if rejection is None:
            return None
        if rejection.contact_quality_signature != contact_quality_signature:
            self._fire_control_rejections.pop(key, None)
            return None
        if (
            int(current_frame) - rejection.rejected_frame
            < self.fire_control_rejection_cooldown_frames
        ):
            return rejection
        return None

    def record_fire_control_rejection(
        self,
        attacker_id: str,
        target_entity_id: str,
        reason_key: str,
        reason: str,
        contact_quality_signature: str,
        current_frame: int,
    ) -> None:
        key = (str(attacker_id), str(target_entity_id))
        self._fire_control_rejections[key] = FireControlRejection(
            attacker_id=key[0],
            target_entity_id=key[1],
            reason_key=str(reason_key),
            reason=str(reason),
            contact_quality_signature=str(contact_quality_signature),
            rejected_frame=int(current_frame),
        )

    def clear_fire_control_rejection(
        self, attacker_id: str, target_entity_id: str
    ) -> None:
        self._fire_control_rejections.pop(
            (str(attacker_id), str(target_entity_id)), None
        )

    def active_attackers(self, target_id: str) -> int:
        return len(self._slots.get(self.canonical_target_id(target_id), {}))

    def attacker_ids(self, target_id: str) -> Set[str]:
        return set(self._slots.get(self.canonical_target_id(target_id), {}))

    def is_attacking(self, attacker_id: str, target_id: str) -> bool:
        return attacker_id in self._slots.get(
            self.canonical_target_id(target_id), {}
        )

    def attack_target_for(self, attacker_id: str) -> Optional[str]:
        for target_id, slots in self._slots.items():
            if attacker_id in slots:
                return target_id
        return None

    def release_attacker(self, attacker_id: str) -> None:
        for target_id, slots in list(self._slots.items()):
            if attacker_id in slots:
                del slots[attacker_id]
            if not slots:
                self.release_target(target_id)
        for key in list(self._target_missing_frames):
            if key[0] == attacker_id:
                self._target_missing_frames.pop(key, None)

    def takeoff_pending(self, entity_id: str) -> bool:
        return entity_id in self._takeoff_requests

    def return_pending(self, entity_id: str) -> bool:
        return entity_id in self._return_requests

    def record_takeoff_request(self, entity_id: str, current_frame: int) -> None:
        self._takeoff_requests[entity_id] = int(current_frame)

    def record_return_request(self, entity_id: str, current_frame: int) -> None:
        self._return_requests[entity_id] = int(current_frame)

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

    def target_attack_status(
        self,
        attacker_id: str,
        target_id: str,
        planned_attackers: Iterable[str] = (),
    ) -> Tuple[int, bool, bool, int]:
        """一次字典查找返回目标评估所需的全部攻击状态。"""

        if (
            not self._slots
            and not self._target_aliases
            and not self._interceptors_launched
        ):
            if not planned_attackers:
                return 0, True, False, 0
            if isinstance(planned_attackers, (set, frozenset)):
                planned = planned_attackers
            else:
                planned = set(planned_attackers)
            return (
                0,
                attacker_id not in planned
                and len(planned) < MAX_ATTACKERS_PER_TARGET,
                False,
                0,
            )

        canonical_id = self.canonical_target_id(target_id)
        active_slots = self._slots.get(canonical_id, {})
        if not planned_attackers:
            planned = ()
        elif isinstance(planned_attackers, (set, frozenset)):
            planned = planned_attackers
        else:
            planned = set(planned_attackers)
        already_attacking = attacker_id in active_slots
        if not active_slots:
            slot_available = (
                attacker_id not in planned
                and len(planned) < MAX_ATTACKERS_PER_TARGET
            )
        elif not planned:
            slot_available = (
                attacker_id not in active_slots
                and len(active_slots) < MAX_ATTACKERS_PER_TARGET
            )
        else:
            occupied = set(active_slots)
            occupied.update(planned)
            slot_available = (
                attacker_id not in occupied
                and len(occupied) < MAX_ATTACKERS_PER_TARGET
            )
        return (
            len(active_slots),
            slot_available,
            already_attacking,
            self._interceptors_launched.get(canonical_id, 0),
        )

    def interceptors_launched(self, target_id: str) -> int:
        return self._interceptors_launched.get(
            self.canonical_target_id(target_id), 0
        )

    def interceptor_available(self, target_id: str, planned_count: int = 0) -> bool:
        return (
            self.interceptors_launched(target_id) + max(0, int(planned_count))
            < MAX_INTERCEPTORS_PER_MISSILE
        )

    def record_successful_attack(
        self,
        attacker_id: str,
        target_id: str,
        started_frame: int,
        target_is_missile: bool,
        target_aliases: Sequence[Optional[str]] = (),
        interceptor_count: int = 1,
    ) -> None:
        if (
            not isinstance(interceptor_count, int)
            or isinstance(interceptor_count, bool)
            or interceptor_count <= 0
        ):
            raise ValueError("interceptor_count 必须是正整数")
        target_id = self.canonical_target_id(target_id)
        self._targets_with_seen_weapon.discard(target_id)
        slots = self._slots.setdefault(target_id, {})
        slots[attacker_id] = AttackSlot(
            attacker_id=attacker_id,
            target_id=target_id,
            started_frame=int(started_frame),
            target_is_missile=bool(target_is_missile),
        )
        self._target_aliases[target_id] = target_id
        for alias in target_aliases:
            if alias:
                self._target_aliases[str(alias)] = target_id
        self._target_missing_frames[(attacker_id, target_id)] = 0
        if target_is_missile:
            current = self._interceptors_launched.get(target_id, 0)
            self._interceptors_launched[target_id] = min(
                MAX_INTERCEPTORS_PER_MISSILE,
                current + interceptor_count,
            )
            self._missile_last_seen[target_id] = int(started_frame)

    def release_target(self, target_id: str) -> None:
        canonical_id = self._target_aliases.get(target_id, target_id)
        self._slots.pop(canonical_id, None)
        self._targets_with_seen_weapon.discard(canonical_id)
        for key in list(self._target_missing_frames):
            if key[1] == canonical_id:
                self._target_missing_frames.pop(key, None)
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
