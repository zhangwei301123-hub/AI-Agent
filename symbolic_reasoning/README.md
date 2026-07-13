# 简单符号推理智能体

目录包含三个核心文件：

- `entity.py`：读取新接口的 `data.data.UnitList`，编码文档规定的实体字段；
- `agent.py`：按规则推出返航、攻击、追击、搜索或保持；
- `execute_actions.py`：校验 8×5 动作矩阵并调用根目录执行接口；
- `symbolic_reasoning4test.py`：类似 `maddpg4test.py` 的运行环境和循环。

```python
from symbolic_reasoning import ReasoningFacts, SymbolicReasoningAgent

agent = SymbolicReasoningAgent()
facts = ReasoningFacts(
    entity_id="our-aircraft-0001",
    target_id="enemy-aircraft-0001",
    attack_authorized=True,
    target_type_allowed=True,
    weapon_available=True,
    within_attack_range=True,
    aimed_at_target=True,
    safety_clearance=True,
    target_lon=120.1,
    target_lat=30.2,
)

decision = agent.reason(facts)
print(decision.conclusion)   # ATTACK
print(decision.explanation)  # 规则编号、结论和依据

# 真正连接仿真并复用 execute.execute_actions：
result = agent.run([facts], enemy_ids=[facts.target_id])
```

使用 `source/我方视角下态势完整响应.txt` 跑完整链路。默认会完成推理并实际
调用执行接口：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test --steps 1
```

执行链为 `agent → symbolic_reasoning.execute_actions → execute.execute_actions`。
如果只需要检查推理结果而不下发命令，必须显式使用：

```powershell
python -m symbolic_reasoning.symbolic_reasoning4test --steps 1 --dry-run
```

编码向量共有 21 个字段，顺序由 `entity.FEATURE_NAMES` 固定，包括阵营、接触目标、
武器标志、实体枚举、目标域、经纬高、航向、速度、血量、探测/打击范围、通信、
干扰和可操纵标志。打击范围采用接口文档规定的公里单位。

测试：

```powershell
python -m unittest discover -s tests -v
```

自动执行甲方要求的四项测试：

```powershell
python -m symbolic_reasoning.acceptance
```

该命令一次完成：9 个正确性用例、512 种输入事实组合覆盖、512 种组合的
推理路径校验，以及默认 10000 次推理的耗时和内存测试。默认门槛为
P95 不超过 5 ms、Python 峰值分配内存不超过 16 MiB，可用命令行参数调整。
