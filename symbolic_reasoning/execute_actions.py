"""符号推理包的动作执行层。"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


ACTOR_COUNT = 8
ACTION_SIZE = 5

ActionsDict = Dict[str, List[List[Any]]]
ExecuteBackend = Callable[[ActionsDict, Sequence[str], float, Any], Tuple[Any, Any]]


class ActionValidationError(ValueError):
    """动作字典不符合现有 execute_actions 接口约定。"""


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


def _load_project_backend() -> ExecuteBackend:
    # 根目录 execute.py 会初始化 gRPC，因此只在真正下发动作时导入。
    from execute import execute_actions as project_execute_actions

    return project_execute_actions


def execute_actions(
    actions_dict: ActionsDict,
    enemy_ids: Sequence[str],
    probablity: float = 0.7,
    logger: Any = None,
    backend: Optional[ExecuteBackend] = None,
) -> Tuple[Any, Any]:
    """校验动作并复用根目录 ``execute.execute_actions`` 执行。

    参数名 ``probablity`` 保留了原接口的拼写，保证调用方式兼容。测试时可通过
    ``backend`` 注入假执行器，不连接仿真服务。
    """

    validate_actions_dict(actions_dict)
    if not isinstance(enemy_ids, Sequence) or isinstance(enemy_ids, (str, bytes)):
        raise ActionValidationError("enemy_ids 必须是目标 ID 列表")
    if not isinstance(probablity, (int, float)):
        raise ActionValidationError("probablity 必须是数值")
    threshold = float(probablity)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ActionValidationError("probablity 必须位于 [0, 1]")

    selected_backend = backend or _load_project_backend()
    return selected_backend(actions_dict, list(enemy_ids), threshold, logger)

