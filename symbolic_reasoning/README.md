# 符号推理智能算法

本目录实现 `rule.md` 中已经确认的业务规则：

- 50 km 内来袭导弹优先拦截；进入 5 km 后触发固定右转 90° 紧急规避；
- 按目标域选择射程、库存校验、攻击高度和期望武器类型；
- 距离等于最大射程时允许提交攻击请求；
- 只有飞机允许超距追击，进入射程后重新推理；
- 飞机和潜艇需要严格小于 30°，水面舰艇免除朝向限制；
- 50 km 内的合法导弹目标优先于其他目标，否则选择攻击范围内最近合法目标；
- 同一目标最多 3 个攻击者；在途武器消失时立即释放槽位，发射后 10 帧未形成武器实体时释放，600 帧为最终兜底；
- 同一导弹目标生命周期内累计最多 4 发拦截弹；
- 健康可控且处于停放状态的飞机主动起飞，并防止逐帧重复下发；
- 在空飞机燃油不高于 20%，或已安装的打击武器耗尽时返航；
- 已有攻击的目标/授权/安全/后勤条件消失时主动调用 `cancelAttackw`；
- 巡逻机在含边界的巡逻区内、距海面 0～500 m 时部署浮标；
- 默认通过 `getMissionList` 获取巡逻区域，为巡逻飞机下达多点巡逻坐标。

## 文件

- `entity.py`：读取 `data.data.UnitList`，编码油量、停放/在空状态、库存、任务、武器目标链路和删除反馈；
- `agent.py`：统一 `TargetEvaluation` 攻击约束，匹配规则、输出推理路径并生成 8×5 Actor 动作矩阵；
- `state.py`：维护并发攻击槽位、起飞/返航防重复窗口、在途武器释放和累计拦截弹数量；
- `control.py`：监听前端开始、暂停和停止状态，并门控符号推理循环；
- `live.py`：读取红方视角态势，并通过 `GetUnitData` 获取红方真实武器、传感器、油量和运行状态；
- `mission.py`：复用 `getMissionList`，保留任务名称、类型和区域坐标；
- `execute_actions.py`：校验动作；非攻击动作复用公共 8×5 接口和 RPC 约定，攻击动作执行
  `GetWeaponFiringInfo → 武器选择 → AttackTarget` 流水线；
- `symbolic_reasoning4test.py`：类似 `maddpg4test.py` 的运行入口，默认实际执行；
- `acceptance.py`：自动执行正确性、覆盖性、可解释性和性能测试。

## 默认推理并实际执行

连接当前推演服务时，推荐使用实时模式：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test `
  --live `
  --steps 0 `
  --attack-rpc-target 10.2.0.106:50051
```

`--steps 0` 表示持续逐帧推理。没有 `--dry-run` 时会实际执行；前端暂停后程序
自动停止取帧和下发命令，前端继续后自动恢复。

导弹优先拦截距离默认是 50 km，可通过 `--missile-intercept-distance-km` 调整，例如：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test `
  --live `
  --steps 0 `
  --missile-intercept-distance-km 80 `
  --attack-rpc-target 10.2.0.106:50051
```

这个参数只控制“多远开始优先拦截导弹”；5 km 紧急规避阈值保持不变。

下面的命令是文件回放模式，主要用于离线复现和测试：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test --steps 1
```

执行链为：

```text
态势编码 → 事实校验 → 导弹优先/最近合法目标 → 规则匹配 → 推理路径
        → 8×5 动作矩阵 → symbolic_reasoning.execute_actions
        → 非攻击动作使用本目录的 protobuf 调用既有 RPC
        → 攻击动作查询武器并调用 AttackTarget → 系统执行反馈
```

时间口径：UI 的轮询间隔使用现实秒，`0～9` 表示 UI 的推演倍速。算法通过
`GetEngineStatus.time_compress` 读取当前倍率；每成功处理一份新态势，冷却时钟按
当前倍率推进，即 `cooldown_frame += multiplier`。例如 50 倍速下，600 帧超时的
剩余量按 `600 → 550 → 500` 递减。前端暂停期间不处理新态势，冷却时钟不推进。
目标 Contact 连续丢失 3 帧属于探测防抖，仍按实际收到的连续态势份数计数，不乘
倍速。Turbo 没有固定数值倍率，当前按 60 倍推进冷却时钟。

> 注意：符号推理执行层保持与项目一致的 8×5 动作格式，但不再导入根目录
> `execute.py`。两套生成代码都使用 protobuf 的 `package proto`，放在同一 Python
> 进程会产生描述符重名。符号执行层现在直接使用动作中的高度/速度等级 0/1/3/5。

如果只检查推理结果、不向仿真系统下发命令，必须显式使用：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test --steps 1 --dry-run
```

## 跟随前端自动暂停

默认启动前端状态监听。实时模式通过同一个远程服务的 `GetEngineStatus` 查询状态；
文件回放模式也通过本目录 protobuf 的 `GetEngineStatus` 查询状态。收到 `pause` 后停止进入下一轮推理和命令下发，收到 `start` 或
`running` 后自动恢复，收到 `stop` 后退出循环。暂停时间不计入 `--steps`。
主线程等待 UI 继续时采用 0.1 秒有界等待，因此在前端暂停状态下按 `Ctrl+C` 也会
直接停止，不需要先从前端发送继续或停止信号。

如果只在没有前端/仿真服务的环境中离线检查文件，可以显式关闭 UI 控制：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test `
  --steps 1 `
  --dry-run `
  --ignore-ui-control
```

控制信号读取失败时默认保持暂停，防止在无法确认前端状态时继续下发命令。

## 查看可解释性推理路径

默认采用安静日志：`INFO` 输出启动阶段的实时连接、任务加载、规则参数、UI 监听和
UI 状态切换，以及已经被 RPC 接受的正向操作，例如打开/关闭传感器、设置航路、
起飞、返航、部署浮标和成功提交武器发射。普通规则未命中、候选排除、武器预检
拒绝以及重复态势不会显示为 `INFO`；相同 RPC 故障只首次显示 `WARNING`。

需要执行可解释性检查时，增加 `--verbose-reasoning`：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test `
  --live `
  --steps 0 `
  --verbose-reasoning
```

此时 `DEBUG` 会输出各实体的规则编号、命中/未命中结果、事实依据、最终结论和执行状态。
日志会同时显示在控制台，并保存在：

```text
logs/symbolic_reasoning_YYYYMMDD_HHMMSS.log
```

使用 `--dry-run --verbose-reasoning` 可只检查完整推理路径而不下发命令；不使用
`--dry-run` 时默认实际执行。

## 可控性和武器能力判定

红方存活的非武器实体默认全部可控。旧 `getSituation` 中的 proto3
`isCanManaged=false` 可能只是服务端未填写该字段，因此不再用它阻断红方命令。

实时模式也不再相信旧 `getSituation` 中经常为 0 的 `weaponNumber.airNum`、
`shipNum`、`subNum`。每帧会针对红方平台调用 `GetUnitData(unit_id)`，根据
`unit_weapons` 中每种武器的实际剩余数量、武器类型和对空/对海/对潜射程，生成
符号规则需要的“是否具有对应域武器”和“最大有效射程”事实。舰炮弹药不会被误算成
防空导弹。

文件回放模式仍可读取态势文件中的 `weaponNumber`，用于离线测试：

```json
{
  "airNum": 4,
  "shipNum": 2,
  "subNum": 1,
  "landNum": 0,
  "buoyNum": 6
}
```

实时推理在生成 `REQUEST_ATTACK` 前，会针对最终的攻击者和目标调用
`GetWeaponFiringInfo(attacker_id, target_id)`，并把 `can_fire`、拒绝原因和冷却状态
写入 `ReasoningFacts`。没有可立即发射武器时直接输出 `HOLD`，不会再进入
`AttackTarget`。执行层会在真正提交前再做一次防御性复核，并优先选择
当前可发射的导弹/鱼雷（显式指定名称时优先匹配，例如 `YJ-18`），随后把
`weapon_db_id` 和数量传给 `AttackTarget`。普通目标默认提交 2 发；导弹目标会
依据累计最多 4 发规则缩减本次数量。
因此实时攻击的判定链为：

```text
GetUnitData：平台确有对应域导弹和射程
    → 符号规则：距离、朝向、并发和拦截弹累计限制均通过
    → GetWeaponFiringInfo：该目标当前存在 can_fire=true 的合适武器
    → AttackTarget
```

`GetUnitData` 查询失败时该平台本帧按无可确认武器处理，不使用最大射程冒充弹药数量。

相同“攻击方 + 稳定目标实体 GUID + Contact 质量 + 拒绝原因”默认冷却 10 帧，
冷却内不重复调用 `GetWeaponFiringInfo`；冷却到期后重试。Contact GUID、探测
新鲜度等级、不确定区、目标类别或识别状态发生变化时，视为 Contact 质量变化，
立即解除旧冷却并重新预检。可用 `--fire-rejection-cooldown-frames` 调整窗口。

成功攻击后按稳定目标实体 GUID 关联可能变化的 Contact GUID。目标短暂消失的
前 2 帧保持现有攻击，第 3 个连续丢失帧才生成 `CANCEL_ATTACK`；重新捕获会把
计数清零。可用 `--target-loss-grace-frames` 调整阈值。Contact 的
`BloodAmount=0` 在红方视角表示未知，不单独作为取消依据。

目标必须出现在红方视角 `GetThreeSituation` 中，且具有红方 `contactGuid`，才会
进入攻击候选。全局/上帝视角可以看见但红方尚未形成 Contact 的蓝方实体不会自动
开火，因为 `GetWeaponFiringInfo` 明确拒绝全局实体 GUID。旧态势产生的重复
`NOT_FOUND` 只提示一次，不再每帧输出完整 warning。

攻击 RPC 默认连接 `10.2.0.106:50051`，可在运行时修改：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test `
  --attack-rpc-target 10.2.0.106:50051 `
  --steps 1
```

也可以直接调用独立流水线函数：

```python
from symbolic_reasoning import execute_attack_pipeline

result = execute_attack_pipeline(
    attacker_id="我方单位ID",
    target_id="敌方目标或Contact ID",
    quantity=2,
    preferred_weapon_name="YJ-18",
    rpc_target="10.2.0.106:50051",
    logger=logger,
)
```

`result.success` 表示 `AttackTarget` RPC 是否成功接受请求；实际是否发射、命中
仍以后续态势中的武器实体和攻击报告为准。默认只把成功提交的发射记录为 `INFO`；
候选武器、库存、可发射评估、最终选择和请求数量在 `--verbose-reasoning` 下作为
`DEBUG` 可解释依据输出。

## 巡逻任务区域和坐标执行

默认运行时会通过本目录的 RPC stub 调用 `getMissionList()`。算法使用实体的 `missionId`
匹配任务，保留 `missionName`、`missionType` 和 `areaPoints`。巡逻机没有紧急
规避、攻击或浮标部署动作时，输出 `PATROL` 结论：

```text
任务区域顶点 → 向中心收缩 20% → 从距离飞机最近的点开始排序
             → 多点航路 Actor → setUnitRoutew RPC
```

巡逻和普通搜索只开启平台实际装备的雷达/声呐。实时模式通过
`GetUnitData.unir_sensor_params` 读取装备能力，不把当前 `active_status` 当成能力；
没有雷达和声呐的平台不发送 `controlUnitSensorw`。高度保持在飞机当前高度对应的
离散层，速度使用等级 3。区域向内收缩可以避免浮点误差导致航点落到任务区外。

`--mission-areas` 仍可加载本地 JSON，并覆盖相同 `missionId` 的 RPC 数据：

```json
{
  "mission-patrol-1": {
    "is_patrol": true,
    "area_points": [
      [120.0, 30.0],
      [121.0, 30.0],
      [121.0, 31.0],
      [120.0, 31.0]
    ]
  }
}
```

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test `
  --mission-areas source/mission_areas.json `
  --steps 1
```

多边形边界视为巡逻区域内，高度直接采用系统反馈的实体 `altitude`。
离线运行且没有推演服务时，增加 `--ignore-rpc-missions`。

## 测试

```powershell
python -m unittest discover -s tests -v
```

甲方四项自动验收：

```powershell
python -m symbolic_reasoning.acceptance
```

验收内容包括：

- 21 个业务正确性与边界用例；
- 15 个标准化布尔事实的 `2^15 = 32768` 种组合全覆盖；
- 32768 条结论的规则编号、证据和推理路径检查；
- 默认 10000 次核心推理的耗时和 Python 峰值分配内存测试；
- 默认 10000 次“核心推理 + 完整 `Decision.explanation` 格式化”测试；
- 默认 5 次 700 实体最坏端到端测试：350 个我方平台对 350 个目标，
  每帧核验 122500 次目标评估、122500 次来袭候选扫描和 350 份完整解释。

默认核心推理和完整解释的 P95 门槛均为 5 ms，峰值分配内存不超过
16 MiB；700 实体最坏端到端 P95 门槛为 5000 ms、峰值分配内存不超过
128 MiB。可通过 `--max-p95-ms`、`--max-explanation-p95-ms`、
`--max-memory-mib`、`--worst-case-iterations`、
`--max-worst-case-p95-ms` 和 `--max-worst-case-memory-mib` 调整。
