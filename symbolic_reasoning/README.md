# 符号推理智能算法

本目录实现 `rule.md` 中已经确认的业务规则：

- 5 km 来袭导弹识别和固定右转 90° 规避；
- 按目标域选择射程、库存校验、攻击高度和期望武器类型；
- 距离等于最大射程时允许提交攻击请求；
- 只有飞机允许超距追击，进入射程后重新推理；
- 飞机和潜艇需要严格小于 30°，水面舰艇免除朝向限制；
- 攻击范围内最近合法目标优先；
- 同一目标最多 3 个攻击者，命中或成功发射后 10 分钟释放；
- 同一导弹目标生命周期内累计最多 4 发拦截弹；
- 巡逻机在含边界的巡逻区内、距海面 0～500 m 时部署浮标。

## 文件

- `entity.py`：读取 `data.data.UnitList`，编码字段、库存、任务、武器目标链路和删除反馈；
- `agent.py`：匹配规则、输出结论和逐条推理路径，并生成 8×5 Actor 动作矩阵；
- `state.py`：维护并发攻击槽位、10 分钟超时和累计拦截弹数量；
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

> 注意：符号推理动作矩阵会正确写入高度/速度等级，但当前根目录
> `execute.py` 的航路和高度速度执行分支仍将两者固定为等级 4。若仿真系统必须
> 实际采用等级 0/1/3/5，需要另行修改公共执行层；本目录未擅自改变神经算法共用接口。

如果只检查推理结果、不向仿真系统下发命令，必须显式使用：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test --steps 1 --dry-run
```

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
- 默认 10000 次推理的耗时和 Python 峰值分配内存测试。

默认性能门槛为 P95 不超过 5 ms、峰值分配内存不超过 16 MiB，可通过
`--max-p95-ms` 和 `--max-memory-mib` 调整。
