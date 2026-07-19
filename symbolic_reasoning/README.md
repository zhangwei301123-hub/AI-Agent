# 符号推理智能算法

本目录实现 `rule.md` 中已经确认的业务规则：

- 5 km 来袭导弹识别和固定右转 90° 规避；
- 按目标域选择射程、库存校验、攻击高度和期望武器类型；
- 距离等于最大射程时允许提交攻击请求；
- 只有飞机允许超距追击，进入射程后重新推理；
- 飞机和潜艇需要严格小于 30°，水面舰艇免除朝向限制；
- 攻击范围内最近合法目标优先；
- 同一目标最多 3 个攻击者，命中或成功发射后 600 个后台数据帧释放；
- 同一导弹目标生命周期内累计最多 4 发拦截弹；
- 巡逻机在含边界的巡逻区内、距海面 0～500 m 时部署浮标。

## 文件

- `entity.py`：读取 `data.data.UnitList`，编码字段、库存、任务、武器目标链路和删除反馈；
- `agent.py`：统一 `TargetEvaluation` 攻击约束，匹配规则、输出推理路径并生成 8×5 Actor 动作矩阵；
- `state.py`：维护并发攻击槽位、600 帧超时和累计拦截弹数量；
- `control.py`：监听前端开始、暂停和停止状态，并门控符号推理循环；
- `execute_actions.py`：校验动作并复用根目录 `execute.execute_actions`；
- `symbolic_reasoning4test.py`：类似 `maddpg4test.py` 的运行入口，默认实际执行；
- `acceptance.py`：自动执行正确性、覆盖性、可解释性和性能测试。

## 默认推理并实际执行

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test --steps 1
```

执行链为：

```text
态势编码 → 事实校验 → 最近合法目标 → 规则匹配 → 推理路径
        → 8×5 动作矩阵 → symbolic_reasoning.execute_actions
        → execute.execute_actions → 系统执行反馈
```

时间口径：UI 的轮询间隔使用现实秒，`0～9` 表示 UI 的推演倍速；算法每成功
取得一份后台态势数据计为 1 帧。规则冷却、并发槽位超时等只按数据帧累计，
不乘 1x、5x、10x、Turbo 等时间压缩倍率。前端暂停期间不会取得下一帧，帧计数
也不会增加。

> 注意：符号推理动作矩阵会正确写入高度/速度等级，但当前根目录
> `execute.py` 的航路和高度速度执行分支仍将两者固定为等级 4。若仿真系统必须
> 实际采用等级 0/1/3/5，需要另行修改公共执行层；本目录未擅自改变神经算法共用接口。

如果只检查推理结果、不向仿真系统下发命令，必须显式使用：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test --steps 1 --dry-run
```

## 跟随前端自动暂停

默认启动前端状态监听。程序每个现实秒通过根目录 `execute.get_control_signal()` 查询
推演状态：收到 `pause` 后停止进入下一轮推理和命令下发，收到 `start` 或
`running` 后自动恢复，收到 `stop` 后退出循环。暂停时间不计入 `--steps`。

如果只在没有前端/仿真服务的环境中离线检查文件，可以显式关闭 UI 控制：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test `
  --steps 1 `
  --dry-run `
  --ignore-ui-control
```

控制信号读取失败时默认保持暂停，防止在无法确认前端状态时继续下发命令。

## 查看可解释性推理路径

无需额外参数。程序与 `maddpg4test.py` 一样使用项目的 `logger.info(...)`，
每一步都会输出各实体的规则编号、匹配结果、事实依据、最终结论和执行状态。
日志会同时显示在控制台，并保存在：

```text
logs/symbolic_reasoning_YYYYMMDD_HHMMSS.log
```

使用 `--dry-run` 时执行状态为 `DRY_RUN`；不使用该参数时默认实际执行，
日志会记录实际执行状态以及相同的完整推理路径。

## 弹药字段

算法读取运行态势中可选的 `weaponNumber`：

```json
{
  "airNum": 4,
  "shipNum": 2,
  "subNum": 1,
  "landNum": 0,
  "buoyNum": 6
}
```

只使用数量判断是否允许发射，不能指定具体武器；真正的武器选择交由系统。
如果态势缺少对应库存数量，算法按 0 处理并安全拒绝攻击，不使用最大射程冒充弹药数量。

## 巡逻任务区域

浮标规则需要任务类型和巡逻区域。可以在实体中提供 `missionId`、`missionType`，
并通过 `--mission-areas` 加载任务区域：

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
