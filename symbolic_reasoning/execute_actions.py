"""符号推理包的动作执行层和 ``AttackTarget`` 攻击流水线。"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


ACTOR_COUNT = 8
ACTION_SIZE = 5
ATTACK_ACTOR = 4
DEFAULT_ATTACK_QUANTITY = 2
DEFAULT_ATTACK_RPC_TARGET = "10.2.0.106:50051"

ActionsDict = Dict[str, List[List[Any]]]
ExecuteBackend = Callable[[ActionsDict, Sequence[str], float, Any], Tuple[Any, Any]]


@dataclass(frozen=True)
class AttackPipelineResult:
    """一次“查询武器 → 选择武器 → 提交攻击”的执行结果。"""

    success: bool
    attacker_id: str
    target_id: str
    weapon_db_id: Optional[int] = None
    weapon_name: str = ""
    requested_quantity: int = 0
    submitted_quantity: int = 0
    mode: str = "manual"
    reason: str = ""
    candidate_evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class WeaponFirePrecheck:
    """供推理层使用的 GetWeaponFiringInfo 标准化结果。"""

    can_fire: bool
    attacker_id: str
    target_id: str
    weapon_db_id: Optional[int] = None
    weapon_name: str = ""
    ready_quantity: int = 0
    reason: str = ""
    reason_key: str = ""
    candidate_evidence: Tuple[str, ...] = ()


AttackPipelineBackend = Callable[..., AttackPipelineResult]
WeaponFirePrecheckBackend = Callable[..., WeaponFirePrecheck]

# 旧态势可能给出全局实体 GUID 而不是红方 Contact ID。相同失败只提示一次，
# 避免持续推演时每帧刷出完整 gRPC NOT_FOUND 堆栈。
_MISSING_CONTACT_WARNED = set()
_LAST_SENSOR_ACTIONS: Dict[str, Tuple[bool, bool, bool]] = {}


class ActionValidationError(ValueError):
    """动作字典或攻击流水线参数不符合接口约定。"""


def _log(logger: Any, level: str, message: str, *args: Any) -> None:
    if logger is None:
        return
    method = getattr(logger, level, None) or getattr(logger, "info", None)
    if method is not None:
        method(message, *args)


def validate_actions_dict(actions_dict: Mapping[str, Sequence[Sequence[Any]]]) -> None:
    """检查 ``entity_id -> 8×5 动作矩阵``。"""

    if not isinstance(actions_dict, Mapping):
        raise ActionValidationError("actions_dict 必须是字典")

    for entity_id, actions in actions_dict.items():
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ActionValidationError("实体 ID 必须是非空字符串")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise ActionValidationError("实体 {} 的动作必须是列表".format(entity_id))
        if len(actions) != ACTOR_COUNT:
            raise ActionValidationError(
                "实体 {} 应包含 {} 个 Actor，实际为 {}".format(
                    entity_id, ACTOR_COUNT, len(actions)
                )
            )

        for actor_index, action in enumerate(actions):
            if not isinstance(action, Sequence) or isinstance(action, (str, bytes)):
                raise ActionValidationError(
                    "实体 {} 的 Actor {} 参数必须是列表".format(
                        entity_id, actor_index
                    )
                )
            if len(action) != ACTION_SIZE:
                raise ActionValidationError(
                    "实体 {} 的 Actor {} 应包含 {} 个参数，实际为 {}".format(
                        entity_id, actor_index, ACTION_SIZE, len(action)
                    )
                )
            probability = action[0]
            if not isinstance(probability, (int, float)):
                raise ActionValidationError(
                    "实体 {} 的 Actor {} 概率必须是数值".format(
                        entity_id, actor_index
                    )
                )
            probability = float(probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ActionValidationError(
                    "实体 {} 的 Actor {} 概率必须位于 [0, 1]".format(
                        entity_id, actor_index
                    )
                )


def _validate_attack_request(
    attacker_id: str,
    target_id: str,
    quantity: int,
    mode: str,
) -> None:
    if not isinstance(attacker_id, str) or not attacker_id.strip():
        raise ActionValidationError("attacker_id 必须是非空字符串")
    if not isinstance(target_id, str) or not target_id.strip():
        raise ActionValidationError("target_id 必须是非空字符串")
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
        raise ActionValidationError("quantity 必须是正整数")
    if mode not in ("manual", "salvo", "all"):
        raise ActionValidationError("mode 必须是 manual、salvo 或 all")


def _is_missing_target_contact(error: Exception) -> bool:
    try:
        details = str(error.details())
        code = str(error.code())
    except (AttributeError, TypeError):
        return False
    return "NOT_FOUND" in code and "Contact" in details and "未找到" in details


def _ready_quantity(weapon: Any) -> int:
    """返回当前挂架上能够立即发射的数量。"""

    evaluations = tuple(getattr(weapon, "fire_evaluations", ()) or ())
    if not evaluations:
        # 实时模式要求明确的 can_fire 证据；只有库存但没有评估时安全拒绝。
        return 0
    return sum(
        max(0, int(getattr(evaluation, "quantity", 0)))
        for evaluation in evaluations
        if bool(getattr(evaluation, "can_fire", False))
    )


def _is_guided_strike_weapon(name: str) -> bool:
    """让导弹/鱼雷优先于舰炮等同样被列为 suitable 的武器。"""

    normalized = name.casefold()
    keywords = (
        "导弹",
        "鱼雷",
        "missile",
        "torpedo",
        "sam",
        "aam",
        "asm",
        "yj-",
        "yj ",
        "hhq-",
        "hq-",
        "pl-",
        "cm-",
    )
    return any(keyword in normalized for keyword in keywords)


def _choose_weapon(
    suitable_weapons: Sequence[Any],
    preferred_weapon_name: Optional[str],
) -> Tuple[Optional[Any], Tuple[str, ...]]:
    candidates: List[Tuple[int, int, int, int, Any]] = []
    evidence: List[str] = []
    preferred = (preferred_weapon_name or "").strip().casefold()

    for weapon in suitable_weapons:
        weapon_db_id = int(getattr(weapon, "weapon_db_id", 0))
        weapon_name = str(getattr(weapon, "weapon_name", "") or "")
        total = max(0, int(getattr(weapon, "total_quantity", 0)))
        ready = _ready_quantity(weapon)
        evaluations = tuple(getattr(weapon, "fire_evaluations", ()) or ())
        blocked = [
            str(getattr(item, "evaluation", "") or "")
            for item in evaluations
            if not bool(getattr(item, "can_fire", False))
        ]
        usable = weapon_db_id > 0 and total > 0 and ready > 0
        evidence.append(
            "weapon_db_id={} name={} total={} ready={} usable={} denied={}".format(
                weapon_db_id,
                weapon_name or "<unnamed>",
                total,
                ready,
                usable,
                " | ".join(item for item in blocked if item) or "none",
            )
        )
        if not usable:
            continue

        preferred_match = int(bool(preferred and preferred in weapon_name.casefold()))
        guided_priority = int(_is_guided_strike_weapon(weapon_name))
        # max() 依次选择：显式名称偏好、导弹/鱼雷、立即可发数量、库存。
        candidates.append(
            (
                preferred_match,
                guided_priority,
                ready,
                total,
                weapon,
            )
        )

    if not candidates:
        return None, tuple(evidence)
    selected = max(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            item[3],
            -int(getattr(item[4], "weapon_db_id", 0)),
        ),
    )[4]
    return selected, tuple(evidence)


def _weapon_denial_details(suitable_weapons: Sequence[Any]) -> Tuple[str, ...]:
    details: List[str] = []
    for weapon in suitable_weapons:
        auto_denied = str(
            getattr(weapon, "auto_fire_denied_reason", "") or ""
        ).strip()
        if auto_denied and auto_denied not in details:
            details.append(auto_denied)
        evaluations = tuple(
            getattr(weapon, "fire_evaluations", ()) or ()
        )
        if not evaluations and "fire_evaluations=empty" not in details:
            details.append("fire_evaluations=empty")
        for evaluation in evaluations:
            if bool(getattr(evaluation, "can_fire", False)):
                continue
            denied = str(
                getattr(evaluation, "evaluation", "") or ""
            ).strip()
            if denied and denied not in details:
                details.append(denied)
    if not suitable_weapons:
        details.append("suitable_weapons=empty")
    return tuple(details)


def _rpc_error_key(error: Exception) -> str:
    try:
        code = str(error.code())
    except (AttributeError, TypeError):
        code = type(error).__name__
    return "GET_WEAPON_RPC_ERROR:{}".format(code)


def precheck_weapon_fire(
    attacker_id: str,
    target_id: str,
    quantity: int = DEFAULT_ATTACK_QUANTITY,
    mode: str = "manual",
    preferred_weapon_name: Optional[str] = None,
    rpc_target: Optional[str] = None,
    timeout: float = 10.0,
    logger: Any = None,
    stub: Any = None,
) -> WeaponFirePrecheck:
    """只查询并评估武器，不提交攻击；失败结果可供跨帧冷却。"""

    _validate_attack_request(attacker_id, target_id, quantity, mode)
    if not isinstance(timeout, (int, float)) or float(timeout) <= 0.0:
        raise ActionValidationError("timeout 必须大于 0")
    endpoint = (
        rpc_target
        or os.environ.get("SYMBOLIC_REASONING_RPC_TARGET")
        or DEFAULT_ATTACK_RPC_TARGET
    )
    channel = None
    try:
        from . import engine_pb2, engine_pb2_grpc

        if stub is None:
            import grpc

            channel = grpc.insecure_channel(endpoint)
            grpc.channel_ready_future(channel).result(timeout=float(timeout))
            stub = engine_pb2_grpc.SimulationServiceStub(channel)
        _log(
            logger,
            "info",
            "[攻击预检] GetWeaponFiringInfo attacker_id=%s target_id=%s rpc=%s",
            attacker_id,
            target_id,
            endpoint,
        )
        response = stub.GetWeaponFiringInfo(
            engine_pb2.GetWeaponFiringInfoRequest(
                attacker_unit_id=attacker_id,
                target_unit_id=target_id,
            ),
            timeout=float(timeout),
        )
        _MISSING_CONTACT_WARNED.discard((attacker_id, target_id))
        suitable_weapons = tuple(response.suitable_weapons)
        selected, evidence = _choose_weapon(
            suitable_weapons,
            preferred_weapon_name=preferred_weapon_name,
        )
        for item in evidence:
            _log(logger, "info", "[攻击预检] 候选依据 %s", item)
        if selected is not None:
            return WeaponFirePrecheck(
                can_fire=True,
                attacker_id=attacker_id,
                target_id=target_id,
                weapon_db_id=int(selected.weapon_db_id),
                weapon_name=str(selected.weapon_name or ""),
                ready_quantity=_ready_quantity(selected),
                reason="GetWeaponFiringInfo 存在 can_fire=true 的可立即发射武器",
                reason_key="CAN_FIRE",
                candidate_evidence=evidence,
            )

        denial_details = _weapon_denial_details(suitable_weapons)
        reason = "GetWeaponFiringInfo 无可立即发射武器"
        if denial_details:
            reason += "：{}".format(" | ".join(denial_details))
        return WeaponFirePrecheck(
            can_fire=False,
            attacker_id=attacker_id,
            target_id=target_id,
            reason=reason,
            reason_key="NO_READY_WEAPON:{}".format(
                "|".join(denial_details) or "unspecified"
            ),
            candidate_evidence=evidence,
        )
    except Exception as error:
        reason = "GetWeaponFiringInfo RPC 失败：{}".format(error)
        if _is_missing_target_contact(error):
            warning_key = (attacker_id, target_id)
            if warning_key not in _MISSING_CONTACT_WARNED:
                _MISSING_CONTACT_WARNED.add(warning_key)
                _log(
                    logger,
                    "info",
                    "[攻击预检] 跳过非红方有效Contact target_id=%s；"
                    "相同目标进入冷却",
                    target_id,
                )
        else:
            _log(logger, "warning", "[攻击预检] %s", reason)
        return WeaponFirePrecheck(
            can_fire=False,
            attacker_id=attacker_id,
            target_id=target_id,
            reason=reason,
            reason_key=_rpc_error_key(error),
        )
    finally:
        if channel is not None:
            channel.close()


def execute_attack_pipeline(
    attacker_id: str,
    target_id: str,
    quantity: int = DEFAULT_ATTACK_QUANTITY,
    mode: str = "manual",
    preferred_weapon_name: Optional[str] = None,
    rpc_target: Optional[str] = None,
    timeout: float = 10.0,
    logger: Any = None,
    stub: Any = None,
) -> AttackPipelineResult:
    """执行 ``GetWeaponFiringInfo → 选择武器 → AttackTarget``。

    ``stub`` 用于单元测试或复用已有连接；未传入时连接 ``rpc_target``。
    ``GetWeaponFiringInfo`` 已按目标过滤武器，本函数再优先选择可立即发射的
    导弹/鱼雷。显式传入 ``preferred_weapon_name="YJ-18"`` 可优先选 YJ-18。
    """

    _validate_attack_request(attacker_id, target_id, quantity, mode)
    if not isinstance(timeout, (int, float)) or float(timeout) <= 0.0:
        raise ActionValidationError("timeout 必须大于 0")

    endpoint = (
        rpc_target
        or os.environ.get("SYMBOLIC_REASONING_RPC_TARGET")
        or DEFAULT_ATTACK_RPC_TARGET
    )
    channel = None
    precheck: Optional[WeaponFirePrecheck] = None

    try:
        from . import engine_pb2, engine_pb2_grpc

        if stub is None:
            import grpc

            channel = grpc.insecure_channel(endpoint)
            grpc.channel_ready_future(channel).result(timeout=float(timeout))
            stub = engine_pb2_grpc.SimulationServiceStub(channel)

        precheck = precheck_weapon_fire(
            attacker_id=attacker_id,
            target_id=target_id,
            quantity=quantity,
            mode=mode,
            preferred_weapon_name=preferred_weapon_name,
            rpc_target=endpoint,
            timeout=timeout,
            logger=logger,
            stub=stub,
        )
        if not precheck.can_fire or precheck.weapon_db_id is None:
            reason = precheck.reason
            _log(logger, "warning", "[攻击流水线] 拒绝攻击：%s", reason)
            return AttackPipelineResult(
                success=False,
                attacker_id=attacker_id,
                target_id=target_id,
                requested_quantity=quantity,
                mode=mode,
                reason=reason,
                candidate_evidence=precheck.candidate_evidence,
            )

        weapon_db_id = precheck.weapon_db_id
        weapon_name = precheck.weapon_name
        ready = precheck.ready_quantity
        submitted_quantity = quantity if mode in ("salvo", "all") else min(
            quantity, ready
        )
        _log(
            logger,
            "info",
            "[攻击流水线] 选择武器 name=%s weapon_db_id=%s "
            "requested=%s submitted=%s mode=%s",
            weapon_name,
            weapon_db_id,
            quantity,
            submitted_quantity,
            mode,
        )
        stub.AttackTarget(
            engine_pb2.AttackTargetRequest(
                attacker_unit_id=attacker_id,
                target_unit_id=target_id,
                weapon_db_id=weapon_db_id,
                quantity=submitted_quantity,
                mode=mode,
            ),
            timeout=float(timeout),
        )
        reason = "AttackTarget RPC 已接受攻击请求"
        _log(
            logger,
            "info",
            "[攻击流水线] 执行成功 attacker_id=%s target_id=%s "
            "weapon=%s quantity=%s",
            attacker_id,
            target_id,
            weapon_name,
            submitted_quantity,
        )
        return AttackPipelineResult(
            success=True,
            attacker_id=attacker_id,
            target_id=target_id,
            weapon_db_id=weapon_db_id,
            weapon_name=weapon_name,
            requested_quantity=quantity,
            submitted_quantity=submitted_quantity,
            mode=mode,
            reason=reason,
            candidate_evidence=precheck.candidate_evidence,
        )
    except Exception as error:
        reason = "攻击流水线 RPC 失败：{}".format(error)
        if _is_missing_target_contact(error):
            warning_key = (attacker_id, target_id)
            if warning_key not in _MISSING_CONTACT_WARNED:
                _MISSING_CONTACT_WARNED.add(warning_key)
                _log(
                    logger,
                    "info",
                    "[攻击流水线] 跳过非红方有效Contact target_id=%s；"
                    "相同目标后续不再重复提示",
                    target_id,
                )
        else:
            _log(logger, "warning", "[攻击流水线] %s", reason)
        return AttackPipelineResult(
            success=False,
            attacker_id=attacker_id,
            target_id=target_id,
            weapon_db_id=(
                precheck.weapon_db_id if precheck is not None else None
            ),
            weapon_name=(precheck.weapon_name if precheck is not None else ""),
            requested_quantity=quantity,
            mode=mode,
            reason=reason,
            candidate_evidence=(
                precheck.candidate_evidence if precheck is not None else ()
            ),
        )
    finally:
        if channel is not None:
            channel.close()


def _as_route_values(value: Any, size: int) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
        if len(result) != size:
            raise ActionValidationError("航路经纬度、高度和速度数量必须一致")
        return result
    return [value] * size


def _rpc_success(response: Any) -> bool:
    return int(getattr(response, "code", 1)) == 0


def _execute_symbolic_rpc_actions(
    actions_dict: ActionsDict,
    enemy_ids: Sequence[str],
    probability: float,
    logger: Any,
    rpc_target: Optional[str] = None,
    timeout: float = 10.0,
    stub: Any = None,
) -> Tuple[Dict[str, List[bool]], List[float]]:
    """执行非攻击 Actor，保持根项目的 8×5 接口但只加载本包 proto。

    根目录 ``execute.py`` 使用另一份同为 ``package proto`` 的生成文件。它与
    ``symbolic_reasoning.engine_pb2`` 进入同一进程时会在全局描述符池重名，
    因此这里只复用动作契约和 RPC 方法，不再导入根目录模块。
    """

    from . import engine_pb2, engine_pb2_grpc

    endpoint = (
        rpc_target
        or os.environ.get("SYMBOLIC_REASONING_RPC_TARGET")
        or DEFAULT_ATTACK_RPC_TARGET
    )
    channel = None
    if stub is None:
        import grpc

        channel = grpc.insecure_channel(endpoint)
        grpc.channel_ready_future(channel).result(timeout=float(timeout))
        stub = engine_pb2_grpc.SimulationServiceStub(channel)

    rewards = [0.0] * ACTOR_COUNT
    execute_results: Dict[str, List[bool]] = {}
    try:
        for entity_id, source_actions in actions_dict.items():
            actions = copy.deepcopy(source_actions)
            result: List[bool] = []
            for actor_index, action in enumerate(actions):
                enabled = float(action[0]) >= float(probability)
                if not enabled or actor_index == ATTACK_ACTOR:
                    result.append(False)
                    continue

                try:
                    if actor_index == 0:
                        response = stub.aircraftTakeOffSinglew(
                            engine_pb2.IdRequestw(mdlID=entity_id),
                            timeout=float(timeout),
                        )
                    elif actor_index == 1:
                        response = stub.aircraftReturnToBasew(
                            engine_pb2.IdRequestw(mdlID=entity_id),
                            timeout=float(timeout),
                        )
                    elif actor_index == 2:
                        longitudes = (
                            list(action[1])
                            if isinstance(action[1], Sequence)
                            and not isinstance(action[1], (str, bytes))
                            else [action[1]]
                        )
                        if not longitudes:
                            raise ActionValidationError("航路至少需要一个航点")
                        latitudes = _as_route_values(action[2], len(longitudes))
                        altitudes = _as_route_values(action[3], len(longitudes))
                        velocities = _as_route_values(action[4], len(longitudes))
                        route = [
                            engine_pb2.WayPointw(
                                longitude=float(lon),
                                latitude=float(lat),
                                altitude=int(altitude),
                                velocity=int(velocity),
                            )
                            for lon, lat, altitude, velocity in zip(
                                longitudes, latitudes, altitudes, velocities
                            )
                        ]
                        response = stub.setUnitRoutew(
                            engine_pb2.UnitRoutew(mdlID=entity_id, Route=route),
                            timeout=float(timeout),
                        )
                    elif actor_index == 3:
                        response = stub.adjustUnitAltitudeAndSpeed(
                            engine_pb2.UnitAltitudeAndSpeedw(
                                mdlID=entity_id,
                                velocity=int(action[1]),
                                altitude=int(action[2]),
                            ),
                            timeout=float(timeout),
                        )
                    elif actor_index == 5:
                        sensor_action = (
                            float(action[1]) > 0.5,
                            float(action[2]) > 0.5,
                            float(action[3]) > 0.5,
                        )
                        if _LAST_SENSOR_ACTIONS.get(entity_id) == sensor_action:
                            # 传感器是状态设置而非一次性动作；避免每帧重复 RPC。
                            result.append(True)
                            continue
                        response = stub.controlUnitSensorw(
                            engine_pb2.SensorControlRequestw(
                                id=entity_id,
                                radar=sensor_action[0],
                                sonar=sensor_action[1],
                                ecm=sensor_action[2],
                            ),
                            timeout=float(timeout),
                        )
                    elif actor_index == 6:
                        response = stub.delpoySonobuoyw(
                            engine_pb2.SonobuoyDelpoyRequestw(
                                id=entity_id,
                                passiveOrActive=float(action[1]) > 0.5,
                                shallowOrDeep=float(action[2]) > 0.5,
                            ),
                            timeout=float(timeout),
                        )
                    elif actor_index == 7:
                        response = stub.cancelAttackw(
                            engine_pb2.IdRequestw(mdlID=entity_id),
                            timeout=float(timeout),
                        )
                    else:
                        result.append(False)
                        continue

                    success = _rpc_success(response)
                    result.append(success)
                    if success:
                        rewards[actor_index] += 1.0
                        if actor_index == 5:
                            _LAST_SENSOR_ACTIONS[entity_id] = sensor_action
                        if actor_index == 6:
                            rewards[actor_index] -= 2.0
                    else:
                        _log(
                            logger,
                            "warning",
                            "[动作执行] entity=%s actor=%s RPC拒绝 code=%s error=%s",
                            entity_id,
                            actor_index,
                            getattr(response, "code", None),
                            getattr(response, "error_message", ""),
                        )

                    # 维持原 execute 接口的互斥语义。
                    if actor_index == 0 and success:
                        actions[1][0] = 0.0
                    if actor_index == 2 and success:
                        actions[3][0] = 0.0
                except Exception as error:
                    result.append(False)
                    _log(
                        logger,
                        "warning",
                        "[动作执行] entity=%s actor=%s RPC异常=%s",
                        entity_id,
                        actor_index,
                        error,
                    )
            while len(result) < ACTOR_COUNT:
                result.append(False)
            execute_results[entity_id] = result
        return execute_results, rewards
    finally:
        if channel is not None:
            channel.close()


def _load_symbolic_backend(rpc_target: Optional[str]) -> ExecuteBackend:
    return partial(_execute_symbolic_rpc_actions, rpc_target=rpc_target)


def _attack_requested(action: Sequence[Any], probability: float) -> bool:
    # 保持公共 execute.py 对攻击 Actor 使用 0.4 阈值的兼容行为。
    return float(action[0]) >= min(float(probability), 0.4)


def _attack_quantity(action: Sequence[Any]) -> int:
    value = action[4]
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return DEFAULT_ATTACK_QUANTITY


def _set_actor_result(execute_results: Any, entity_id: str, success: bool) -> None:
    if not isinstance(execute_results, dict):
        return
    result = execute_results.get(entity_id)
    if not isinstance(result, list):
        result = []
        execute_results[entity_id] = result
    while len(result) < ACTOR_COUNT:
        result.append(False)
    result[ATTACK_ACTOR] = bool(success)


def execute_actions(
    actions_dict: ActionsDict,
    enemy_ids: Sequence[str],
    probablity: float = 0.7,
    logger: Any = None,
    backend: Optional[ExecuteBackend] = None,
    attack_backend: Optional[AttackPipelineBackend] = None,
    rpc_target: Optional[str] = None,
) -> Tuple[Any, Any]:
    """执行符号动作；攻击 Actor 默认使用新的 ``AttackTarget`` 流水线。

    非攻击 Actor 复用项目的 8×5 动作和 RPC 接口约定，但只加载本包生成的
    protobuf，避免与根目录生成文件发生描述符重名。为避免同一目标被下发两次，
    传给兼容执行层的副本会关闭攻击 Actor，随后由本模块查询武器并调用
    ``AttackTarget``。仅注入 ``backend`` 时保留原有测试/扩展行为；同时注入
    ``attack_backend`` 时会测试完整分流流程。
    """

    validate_actions_dict(actions_dict)
    if not isinstance(enemy_ids, Sequence) or isinstance(enemy_ids, (str, bytes)):
        raise ActionValidationError("enemy_ids 必须是目标 ID 列表")
    if not isinstance(probablity, (int, float)):
        raise ActionValidationError("probablity 必须是数值")
    threshold = float(probablity)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ActionValidationError("probablity 必须位于 [0, 1]")

    # 保持既有的 backend 注入契约，不在用户自定义执行器外再发真实 RPC。
    if backend is not None and attack_backend is None:
        return backend(actions_dict, list(enemy_ids), threshold, logger)

    selected_backend = backend or _load_symbolic_backend(rpc_target)
    selected_attack_backend = attack_backend or execute_attack_pipeline
    delegated_actions = copy.deepcopy(actions_dict)
    requests: List[Tuple[str, str, int]] = []

    for entity_id, actions in delegated_actions.items():
        attack_action = actions[ATTACK_ACTOR]
        if not _attack_requested(attack_action, threshold):
            continue
        target_id = attack_action[1]
        if target_id is None or not str(target_id).strip():
            _log(
                logger,
                "warning",
                "[攻击流水线] %s 的攻击动作缺少 target_id，安全拒绝",
                entity_id,
            )
            continue
        requests.append(
            (entity_id, str(target_id), _attack_quantity(attack_action))
        )
        # 公共执行层仍执行高度等其他动作，但不再调用 attackContactw。
        attack_action[0] = 0.0

    execute_results, rewards = selected_backend(
        delegated_actions, list(enemy_ids), threshold, logger
    )
    for attacker_id, target_id, quantity in requests:
        result = selected_attack_backend(
            attacker_id=attacker_id,
            target_id=target_id,
            quantity=quantity,
            mode="manual",
            rpc_target=rpc_target,
            logger=logger,
        )
        _set_actor_result(execute_results, attacker_id, result.success)
        if result.success:
            # 与公共 execute.py 中成功攻击的净奖励（-2 + 1）保持一致。
            try:
                rewards[ATTACK_ACTOR] -= 1
            except (IndexError, KeyError, TypeError):
                pass

    return execute_results, rewards
