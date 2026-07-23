# 符号推理规则（当前代码实现版）

> 同步日期：2026-07-21；状态：已与当前 `symbolic_reasoning` 实现核对。
>
> 本文件以 `agent.py`、`symbolic_reasoning4test.py`、`state.py`、`live.py` 和
> `execute_actions.py` 的当前实现为准。当前代码会从目标适用候选中选择具体
> `weapon_db_id`，并把武器数据库 ID、发射数量和攻击模式明确提交给
> `AttackTarget`。

## 1. 总体推理流程

```text
态势字段
  ↓
生成事实（平台、目标、距离、航向、射程、区域、并发状态）
  ↓
事实合法性校验
  ↓
按优先级匹配规则
  ↓
形成候选目标和候选动作
  ↓
执行安全约束（射程、朝向、并发数、弹药数）
  ↓
GetWeaponFiringInfo 查询当前目标的可发射候选武器
  ↓
选择具体 weapon_db_id 和发射数量
  ↓
输出结论（规避/攻击/追击/起飞/返航/取消攻击/部署浮标/巡逻/搜索/保持）
  ↓
生成逐条规则推理路径和 8×5 动作
  ↓
AttackTarget 明确提交 attacker_id、target_id、weapon_db_id、quantity、mode
```

## 2. 术语与事实定义

### 2.1 运行环境固定约定

#### 时间压缩编码

| 编码 | 推演速度 |
|---:|---:|
| 0 | 1 倍速 |
| 1 | 2 倍速 |
| 2 | 5 倍速 |
| 3 | 10 倍速 |
| 4 | 15 倍速 |
| 5 | 30 倍速 |
| 6 | 40 倍速 |
| 7 | 50 倍速 |
| 8 | 60 倍速 |
| 9 | Turbo（极速） |

当前通过 `GetEngineStatus.time_compress` 读取编码。每处理一份新态势，冷却时钟按
映射后的倍率推进：

```text
cooldown_frame = previous_cooldown_frame + time_compression_multiplier

示例（50 倍速）：
600 - 1×50 - 1×50 = 500
```

该折算用于攻击槽位 600 帧硬超时、武器实体出现 10 帧等待、火控拒绝 10 帧冷却
以及起飞/返航 10 帧重试窗口。倍速中途变化时不重置已有状态，从下一份态势开始
使用新倍率。暂停期间没有新态势进入推理，冷却时钟不推进。目标 Contact 连续丢失
3 帧是探测防抖，仍按实际连续态势份数计数，不乘倍速。协议没有给 Turbo 固定
倍率，当前确定性地按 60 倍兜底。

#### 阵营与控制关系

- 我方阵营为红方，只控制红方单位；
- 对方阵营为蓝方；
- 红方单位以蓝方单位作为攻击、防御和威胁判断对象。

### 2.2 平台类型

| 编码 | 类型 | 来源字段 |
|---:|---|---|
| 0 | 飞机 | `unitCategory` / `unitType` / `Icon2D` |
| 1 | 水面舰艇 | `unitCategory` / `unitType` / `Icon2D` |
| 2 | 潜艇 | `unitCategory` / `unitType` / `Icon2D` |
| 3 | 地面目标/设施 | `unitCategory` / `unitType` / `Icon2D` |
| 6 | 武器 | `unitCategory`、`IsWeapon`、`Icon2D` |

**当前实现：** 我方视角样例中，部分接触目标的 `unitCategory` 和 `unitType` 统一为 0，因此目标类型识别顺序采用：

```text
Icon2D 路径 → IsWeapon → unitCategory/unitType → altitude 高度兜底
```

### 2.3 目标域

| 目标域 | 典型目标 | 对应射程字段 | 对应武器类型 |
|---|---|---|---|
| AIR | 飞机、空中导弹 | `rangeStrike_Air` | 防空导弹/空空导弹 |
| SURFACE | 水面舰艇、海平面目标 | `rangeStrike_Surface` | 反舰导弹 |
| SUBMARINE | 潜艇、水下鱼雷 | `rangeStrike_Submarine` | 反潜武器/鱼雷 |
| LAND | 地面设施、车辆 | `rangeStrike_Land` | 对地武器 |

> 原始需求中的“鱼类”按“鱼雷”处理。

### 2.4 关键事实

```text
own_id                         我方实体 ID
own_platform_type              我方平台类型
own_position                   我方经度、纬度、高度
own_heading_deg                我方当前航向角
own_altitude_above_sea_m       系统反馈的我方实体距海面高度
attack_authorized              我方实体是否可操纵；当前为红方存活非武器实体的 commandable
safety_clearance               通信是否正常且未被雷达干扰
target_id                      命令目标 ID；实时红方视角优先且实际攻击应使用 contactGuid
target_entity_id               目标稳定实体 GUID；用于 Contact 重捕获、冷却和别名关联
target_domain                  AIR / SURFACE / SUBMARINE / LAND
target_position                目标经度、纬度、高度
distance_km                    我方到目标的三维距离
target_bearing_deg             我方位置指向目标的方位角
heading_difference_deg         航向与目标方位的最小夹角
max_attack_range_km            针对目标域选择的最大攻击距离
expected_weapon_type           目标域对应的业务武器类别；用于域能力、库存校验和解释
compatible_weapon_count        GetUnitData 中对应目标域武器的当前剩余数量
selected_weapon_db_id          GetWeaponFiringInfo 候选中由执行层选定的具体武器数据库 ID
selected_weapon_name           选定武器名称；用于日志和可解释证据
attack_quantity                规则请求数量；普通目标默认 2，导弹受累计 4 发限制约束
attack_mode                    当前自动执行固定为 manual
incoming_missile               是否存在来袭导弹
missile_target_id              导弹正在追踪的目标 ID
missile_distance_km            导弹与被追踪我方实体的距离
missile_heading_deg            导弹当前航向角
missile_pointing_difference_deg 导弹航向与“导弹至我方”方位的最小夹角
active_attackers_on_target      当前正在攻击该目标的不同攻击者数量
interceptors_launched           针对该导弹目标已经发射的拦截弹数量
inside_patrol_area              实体是否位于巡逻区域内或边界上
has_patrol_mission              实体是否承担巡逻任务
sonobuoy_available              是否携带可部署浮标
is_airborne                    飞机是否处于在空/飞行/返航状态
is_parked                      飞机是否处于停放/停机/已降落状态
fuel_percentage                GetUnitData 返回的燃油百分比；未知为 -1
fuel_low                       燃油信息已知且 fuel_percentage <= 20
has_strike_weapon_system       平台是否确实安装过打击武器
strike_weapon_count            当前剩余打击武器总数
ammunition_low                 已安装打击武器且当前剩余数量为 0
currently_attacking            跨帧攻击状态中是否有该实体的活动攻击槽位
attack_target_missing_frames   原攻击目标连续未出现在红方态势中的帧数
attack_target_loss_grace_frames 连续丢失多少帧后取消，默认 3
attack_conditions_valid        原目标可见或仍在丢失宽限内，且授权、安全、油料条件有效
fire_control_checked           本帧是否已执行或命中缓存的最终武器预检
fire_control_available         GetWeaponFiringInfo 是否存在 can_fire=true 武器
fire_control_cooldown          是否因同目标同质量的相同拒绝处于冷却
fire_control_reason            可发射性结论或服务器拒绝原因
target_quality_signature       Contact GUID、探测新鲜度、不确定区和识别状态的摘要
takeoff_pending                起飞 RPC 已成功接受且仍在 10 帧防重复窗口
return_pending                 返航 RPC 已成功接受且仍在 10 帧防重复窗口
```

### 2.5 当前 RPC 数据与执行链

| 阶段 | 当前接口 | 实际用途 |
|---|---|---|
| 红方态势 | `GetThreeSituation(is_god_view=False)` | 获取红方单位和红方可见 Contact |
| 平台能力 | `GetUnitData(unit_id)` | 获取真实武器库存、目标域射程、传感器、油量和运行状态 |
| 武器预检 | `GetWeaponFiringInfo(attacker_id, target_id)` | 返回目标适用武器、`weapon_db_id`、库存和逐项 `can_fire` 结果 |
| 明确攻击 | `AttackTarget(...)` | 提交攻击方、Contact、具体武器数据库 ID、数量和模式 |

实时攻击目标使用红方视角的 `contactGuid`。稳定实体 GUID 仍保留用于跨帧关联，但
不得在红方尚未形成 Contact 时替代 Contact 强行发射。离线文件回放可以使用实体
GUID 进行规则测试，但真实服务最终是否接受仍以 `GetWeaponFiringInfo` 为准。

### 2.6 当前可输出结论

```text
EVADE_MISSILE      紧急规避来袭导弹
REQUEST_ATTACK     生成攻击请求并进入武器查询、选择和提交流水线
CHASE_TO_RANGE     飞机超距追击，进入射程后重新推理
CHASE_AND_ALIGN    飞机在射程内转向对准，完成后重新推理
DEPLOY_SONOBUOY    部署浮标
SEARCH             开启实际装备的传感器搜索
TAKEOFF            停放飞机起飞
CANCEL_ATTACK      取消已失效攻击
RETURN_TO_BASE     低油或弹药耗尽返航
HOLD               当前不生成危险动作
```

`ATTACK`、`CHASE` 仅是代码中的旧调用兼容别名，分别等价于 `REQUEST_ATTACK`、
`CHASE_TO_RANGE`，新推理路径不使用旧名称。代码没有 `RESELECT_TARGET`、
`HOLD_INTERCEPT_FIRE` 或 `HOLD_SONOBUOY` 这类独立结论。

## 3. 规则优先级

| 优先级 | 规则组 | 说明 |
|---:|---|---|
| P0 | 事实非法/缺失 | 关键事实缺失时禁止危险动作 |
| P1 | 来袭导弹规避 | 直接威胁时覆盖攻击、追击等普通动作 |
| P2 | 放弃失效攻击 | 原攻击条件消失时先取消攻击，避免与返航等动作冲突 |
| P3 | 后勤返航 | 在空飞机低油或已耗尽安装的打击武器时返航 |
| P4 | 停放飞机起飞 | 健康可控且停放的飞机起飞；已接受请求时等待 |
| P5 | 目标选择 | 先得到合法候选目标，再判断攻击 |
| P6 | 并发、射程、朝向与武器 | 检查全部攻击约束，执行可发射性预检并确定攻击参数 |
| P7 | 浮标部署 | 独立任务动作；不得与更高优先级动作冲突 |
| P8 | 巡逻/搜索 | 无更高优先级动作时巡逻或按装备能力搜索 |

同一实体每一帧只输出一个主结论；高优先级规则命中后，不再生成与其冲突的低优先级
动作。来袭导弹规避始终优先于攻击、追击、浮标部署和普通巡逻。

## 4. 详细规则

## R-MSL-001 来袭导弹识别

**目的：** 只有“确实指向该实体”的敌方导弹才触发该实体规避。

**输入事实：**

```text
target.is_weapon
target_domain == AIR
missile_target_id
own_id
missile_distance_km
MISSILE_EVADE_DISTANCE_KM
```

**规则：**

```text
MISSILE_EVADE_DISTANCE_KM = 5

IF target.is_weapon == True
AND target_domain == AIR
AND missile_distance_km <= 5
AND (
    missile_target_id == own_id
    OR (
        missile_target_id is missing
        AND missile_pointing_difference_deg < 30°
    )
)
THEN incoming_missile = True
ELSE incoming_missile = False
```

**已明确：**

1. 来袭导弹规避距离阈值固定为 **5 km**，边界 `5 km` 也触发规避；
2. 优先从 `radiationAndDataLinkLine.WeaponTarget` 等系统字段获取导弹目标 ID；
3. 没有导弹目标 ID 时，允许使用“导弹航向指向我方 + 距离不超过 5 km”推断来袭目标；
4. 航向推断默认采用夹角 `< 30°`，即导弹航向与“导弹位置指向我方位置”的方位差小于 30°。

> 第 4 项的 30° 是当前代码的可计算默认值；如系统提供明确制导目标 ID，始终优先使用系统目标 ID。

## R-MSL-002 横向 90° 紧急规避

**规则：**

```text
IF incoming_missile == True
THEN
    evade_heading = normalize(missile_heading + 90°)
    movement_speed_level = 4
    movement_altitude_level = 5
    attack_enabled = False
    sonobuoy_deploy_enabled = False
    conclusion = EVADE_MISSILE
```

**已明确：** 固定向导弹航向右侧转 90°，即按当前航向角定义计算
`normalize(missile_heading + 90°)`，不进行随机左右选择。

**推理路径示例：**

```text
输入：导弹 M1；被跟踪目标=A01；A01 与 M1 距离=4 km；阈值=5 km
1. R-MSL-001：M1 是武器且属于空中威胁                         → 通过
2. R-MSL-001：missile_target_id(A01) == own_id(A01)           → 通过
3. R-MSL-001：4 km <= 5 km                                    → 通过
4. R-MSL-002：计算 M1 航向右侧 90° 规避方向                    → 命中
5. R-MSL-002：速度等级=4，高度等级=5，关闭攻击和浮标部署       → 输出
结论：EVADE_MISSILE
依据：R-MSL-001、R-MSL-002
```

---

## R-CANCEL-001～002 攻击条件失效与放弃打击

```text
IF currently_attacking == True
AND attack_conditions_valid == False
THEN conclusion = CANCEL_ATTACK
AND actor[7] = enabled
AND rpc = cancelAttackw(own_id)                         [R-CANCEL-001]

IF currently_attacking == True
AND attack_conditions_valid == True
THEN conclusion = HOLD
AND do_not_generate_conflicting_action = True           [R-CANCEL-002]
```

`attack_conditions_valid` 通过稳定实体 GUID 关联重新捕获后变化的 Contact GUID。
目标消失时累计 `attack_target_missing_frames`：默认第 1～2 个连续丢失帧仍保持
攻击，第 3 帧才失效；任一帧重新发现即清零。因此单帧探测闪烁不会触发
`cancelAttackw`。红方 Contact 的 `BloodAmount=0` 视为健康未知，不据此取消；
明确的删除/命中反馈仍会释放目标。实体还必须可控、通信安全且未触发低油。
取消 RPC 成功后立即释放该攻击者占用的攻击槽位；
如果同时低油，则下一帧进入返航规则。弹药耗尽不取消已经形成的在途攻击，待该
攻击结束、槽位释放后再进入返航规则。

---

## R-RTB-001～002 油量/弹药不足返航

```text
fuel_low = fuel_percentage 已知 AND fuel_percentage <= 20
ammunition_low = has_strike_weapon_system AND strike_weapon_count == 0

IF is_aircraft AND is_airborne
AND (fuel_low OR ammunition_low)
AND return_pending == False
THEN conclusion = RETURN_TO_BASE
AND actor[1] = enabled
AND rpc = aircraftReturnToBasew(own_id)                 [R-RTB-001]

IF return_pending == True
THEN conclusion = HOLD                                  [R-RTB-002]
```

非武装任务飞机的 `has_strike_weapon_system=False`，不会因为武器数量为 0 被误判
返航。成功接受的返航请求在 10 帧内不重复下发；飞机进入停放状态后立即清除等待
状态，10 帧仍无状态变化则允许重试。

---

## R-TAKEOFF-001～002 停放飞机起飞

```text
IF is_aircraft AND is_parked AND attack_authorized
AND takeoff_pending == False
THEN conclusion = TAKEOFF
AND actor[0] = enabled
AND rpc = aircraftTakeOffSinglew(own_id)                [R-TAKEOFF-001]

IF takeoff_pending == True
THEN conclusion = HOLD                                  [R-TAKEOFF-002]
```

状态优先使用 `GetUnitData.unit_current_status.text_block_unit_status`；旧态势使用
`stateMap.AirStatus/UnitStatus`。缺少状态文字时，仅将高度大于 10 m 或速度不低于
30 判为在空，将高度 -5～10 m 且速度 0～5 判为停放，其余模糊状态不触发起飞。
成功接受的起飞请求在 10 帧内不重复下发，观测到飞机进入在空状态后立即清除。

---

## R-VAL-001 攻击授权与通信安全校验

```text
attack_authorized = own.commandable
safety_clearance = own.communication_ok AND NOT own.radar_jammed

IF target_id exists
AND (attack_authorized == False OR safety_clearance == False)
THEN conclusion = HOLD
AND do_not_generate_attack_or_chase = True
```

**当前实现：**

1. 实时红方态势中，存活的红方非武器实体默认 `commandable=True`，不会再因原始
   `isCanManaged` 字段缺失或恒为 `false` 而阻止攻击；
2. 实体被识别为通信中断，或干扰状态文本表明雷达被干扰时，
   `safety_clearance=False`；
3. 本规则在并发、射程、朝向和武器检查之前执行，未通过时直接输出 `HOLD`；
4. 可解释路径记录 `attack_authorized` 和 `safety_clearance` 两项事实。

**推理路径示例：**

```text
输入：我方 A01；目标 T01；attack_authorized=True；safety_clearance=False
1. R-TGT-001：存在已选择目标 T01                              → 通过
2. R-VAL-001：实体可操纵且通信安全状态允许攻击                 → 不通过
3. 禁止生成攻击或追击动作                                      → 输出
结论：HOLD
依据：R-VAL-001；事实：attack_authorized=True，safety_clearance=False
```

---

## R-ALT-001 攻击高度参数派生

`R-ALT-001` 是动作参数的业务计算口径，不是 `agent.py` 当前单独输出的主结论或
独立 `InferenceStep`。环境在构造 `ReasoningFacts` 时计算 `attack_altitude_level`；
攻击或追击命中后，该值进入 8×5 动作，可通过事实和动作参数核对。

### 高度等级定义

高度使用系统离散等级，不在符号推理层强制换算为固定米数：

| 平台 | 等级 | 业务含义 |
|---|---:|---|
| 水面舰艇 | 0 | 海面 |
| 飞机等非潜艇平台 | 1 | 最低高度层 |
| 飞机等非潜艇平台 | 3 | 中高空层 |
| 飞机等非潜艇平台 | 5 | 最高高度层 |
| 潜艇 | 1 | 浅层（潜艇专用水下层级） |
| 潜艇 | 3 | 中层（潜艇专用水下层级） |
| 潜艇 | 5 | 高层（潜艇专用水下层级） |

> 潜艇的 1、3、5 是水下层级，不得按飞机高度解释，也不得转换为海拔正高度。

### 攻击高度量化表

| 目标 | 我方平台 | 攻击高度等级 | 说明 | 状态 |
|---|---|---:|---|---|
| 水面舰艇/海平面目标 | 飞机 | 1 | 最低高度层攻击 | 已确认 |
| 水面舰艇/海平面目标 | 水面舰艇 | 0 | 保持海面 | 已确认 |
| 飞机/空中导弹，目标高度 ≤ 5000 m | 飞机 | 3 | 中高空攻击 | 已确认 |
| 飞机/空中导弹，目标高度 > 5000 m | 飞机 | 5 | 最高高度层攻击 | 已确认 |
| 飞机/空中导弹 | 水面舰艇 | 0 | 舰艇保持海面 | 已确认 |
| 潜艇/水下鱼雷 | 飞机 | 1 | 最低高度层反潜 | 已确认 |
| 潜艇/水下鱼雷 | 水面舰艇 | 0 | 舰艇保持海面 | 已确认 |

### 补充平台组合

| 目标 | 我方平台 | 当前实现等级 |
|---|---|---:|
| 水面舰艇 | 潜艇 | 1 |
| 飞机/空中导弹 | 潜艇 | 1 |
| 潜艇/水下鱼雷 | 潜艇 | 3 |

**规则：**

```text
IF target_domain == SURFACE AND own_platform_type == AIRCRAFT
THEN attack_altitude_level = 1

IF target_domain == SURFACE AND own_platform_type == SHIP
THEN attack_altitude_level = 0

IF target_domain == AIR AND own_platform_type == SHIP
THEN attack_altitude_level = 0

IF target_domain == AIR AND own_platform_type == AIRCRAFT AND target_altitude <= 5000m
THEN attack_altitude_level = 3

IF target_domain == AIR AND own_platform_type == AIRCRAFT AND target_altitude > 5000m
THEN attack_altitude_level = 5

IF target_domain == SUBMARINE AND own_platform_type == AIRCRAFT
THEN attack_altitude_level = 1

IF target_domain == SUBMARINE AND own_platform_type == SHIP
THEN attack_altitude_level = 0
```

**参数派生示例：**

```text
输入：我方=A01(飞机)；目标=S01(水面舰艇)；目标高度=0 m
1. 目标分类：Icon2D/高度判定 target_domain=SURFACE             → 通过
2. R-ALT-001：SURFACE + AIRCRAFT                               → 参数匹配
3. 设置 attack_altitude_level=1                                → 写入事实
4. 若最终结论为 REQUEST_ATTACK，则 Mobility Actor 使用高度等级 1 → 写入动作
参数结果：attack_altitude_level=1
参数依据：R-ALT-001
```

**已明确：** 5000 m 继续作为空中目标高度分界；等级对应的实际控制高度由系统负责
映射。当前应从 `ReasoningFacts.attack_altitude_level` 和 Mobility/Waypoint Actor
参数核对“等级 + 业务含义”，不能把它误写成独立的决策结论。

---

## R-RNG-001、R-RNG-003、R-RNG-004 攻击距离约束

**距离公式：** 使用经纬度水平距离和高度差计算三维距离，统一换算为 km。

```text
AIR        → max_attack_range_km = rangeStrike_Air
SURFACE    → max_attack_range_km = rangeStrike_Surface
SUBMARINE  → max_attack_range_km = rangeStrike_Submarine
LAND       → max_attack_range_km = rangeStrike_Land
```

**规则：**

```text
IF max_attack_range_km is missing OR max_attack_range_km <= 0
THEN attack_allowed = False
AND reason = NO_VALID_RANGE
AND conclusion = HOLD                                      [R-RNG-001]

IF distance_km > max_attack_range_km
AND own_platform_type == AIRCRAFT
THEN immediate_attack_allowed = False
AND pursuit_allowed = True
AND conclusion = CHASE_TO_RANGE                            [R-RNG-004]

IF distance_km > max_attack_range_km
AND own_platform_type != AIRCRAFT
THEN immediate_attack_allowed = False
AND pursuit_allowed = False
AND reason = OUT_OF_RANGE
AND conclusion = HOLD                                      [R-RNG-003]

IF distance_km <= max_attack_range_km
THEN within_attack_range = True
AND attack_request_allowed = True
```

**已明确：**

- 飞机或空中导弹使用最大对空射程；
- 水面舰艇使用最大对海射程；
- 潜艇或水下鱼雷使用最大对潜射程；
- 超出对应射程禁止立即攻击；
- 只有飞机允许在射程外追击目标，舰艇、潜艇等其他平台不因超距生成追击动作；
- 飞机追击进入射程后，必须重新执行导弹威胁、目标有效性、并发、射程、朝向和弹药数量检查，全部通过后才允许发射。

**已明确：** 距离恰好等于最大射程时允许提交攻击请求，即使用 `<=`。该结论只表示符号规则允许请求攻击；实际是否成功以系统执行反馈为准。

**推理路径示例（超距）：**

```text
输入：我方舰艇 D01；目标飞机 A02；distance=155 km；rangeStrike_Air=148.16 km
1. 目标分类：A02 属于 AIR                                      → 通过
2. R-RNG-001：选择 rangeStrike_Air=148.16 km                    → 通过
3. R-RNG-004：155 > 148.16 且我方不是飞机，不能追击             → 未命中
4. R-RNG-003：非飞机平台超出射程                               → 命中
5. 禁止生成攻击动作                                             → 输出
结论：若存在其他合法目标则继续对该目标推理；否则输出 HOLD；不新增单独的改选结论
依据：R-RNG-001、R-RNG-004、R-RNG-003
```

**推理路径示例（飞机超距追击）：**

```text
输入：我方飞机 A01；目标 S01；distance=120 km；rangeStrike_Surface=100 km
1. 目标分类：S01 属于 SURFACE                                  → 通过
2. R-RNG-001：选择 rangeStrike_Surface=100 km                  → 通过
3. R-RNG-004：120 > 100 且我方平台为 AIRCRAFT                   → 命中
4. 禁止立即攻击并允许追击                                      → 输出
5. 生成追击动作；不生成攻击动作                                 → 输出
6. 进入 100 km 范围后重新执行全部攻击约束                       → 等待系统新态势
结论：CHASE_TO_RANGE
依据：R-RNG-001、R-RNG-004
```

当前代码没有 `R-RNG-002`。距离等于最大射程时允许攻击是
`within_attack_range = distance_km <= max_attack_range_km` 的边界条件，进入后续
朝向、火控和武器规则，不单独产生距离规则结论。

---

## R-ASW-001 反潜飞机鱼雷释放距离

反潜飞机通过 `unitCategory=Aircraft_ASW(13)`、`UnitSpecificType=7401`、名称中的
“反潜/ASW/直-8J”或有效对潜武器能力识别。0.4 海里统一换算为 0.7408 km，
距离沿用本项目经纬度和高度差计算的三维距离。

```text
IF is_asw_aircraft == True
AND target_domain == SUBMARINE
AND distance_km > 0.7408
THEN immediate_attack_allowed = False
AND conclusion = CHASE_TO_RANGE
AND waypoint = target_position                              [R-ASW-001]

IF is_asw_aircraft == True
AND target_domain == SUBMARINE
AND distance_km <= 0.7408
THEN torpedo_release_distance_valid = True                  [R-ASW-001]
AND continue_with_heading_fire_control_and_weapon_checks = True
```

距离等于 0.4 海里时视为进入释放范围。超距追踪动作使用目标坐标和对潜最低高度层，
不生成攻击动作；进入范围后仍须满足严格小于 30° 的朝向、并发、弹药和
`GetWeaponFiringInfo` 火控检查，才会通过 `R-WPN-003` 发射鱼雷。

---

## R-WPN-000～003 武器适配、具体武器选择与发射请求

| 目标域 | 选择武器 | 规则 ID |
|---|---|---|
| AIR（飞机、空中导弹） | 防空导弹；飞机平台可使用空空导弹 | R-WPN-001 |
| SURFACE（水面舰艇） | 反舰导弹 | R-WPN-002 |
| SUBMARINE（潜艇、水下鱼雷） | 反潜武器/鱼雷 | R-WPN-003 |
| LAND（地面设施、车辆） | 对地武器 | R-WPN-000 |

**规则：**

```text
IF target_domain == AIR
THEN expected_weapon_type = AIR_DEFENCE_OR_AIR_TO_AIR_MISSILE

IF target_domain == SURFACE
THEN expected_weapon_type = ANTI_SHIP_MISSILE

IF target_domain == SUBMARINE
THEN expected_weapon_type = ANTI_SUBMARINE_WEAPON_OR_TORPEDO

IF target_domain == LAND
THEN expected_weapon_type = LAND_ATTACK_WEAPON

IF compatible_weapon_count <= 0
THEN attack_allowed = False
AND reason = NO_MATCHING_WEAPON

IF compatible_weapon_count > 0
AND all_other_attack_constraints_passed == True
THEN query GetWeaponFiringInfo(attacker_id, contact_id)

IF fire_control_checked == True
AND fire_control_available == False
THEN conclusion = HOLD                                      [R-FIRE-001]
AND do_not_call_AttackTarget = True

IF fire_control_available == True
THEN selected_weapon = rank_usable_suitable_weapons(
       preferred_name_match,
       guided_missile_or_torpedo,
       ready_quantity,
       total_quantity,
       lower_weapon_db_id_as_stable_tiebreaker)
AND selected_weapon_db_id = selected_weapon.weapon_db_id
AND attack_mode = manual
AND submitted_quantity = min(attack_quantity, selected_weapon.ready_quantity)
AND call AttackTarget(attacker_id, contact_id,
                      selected_weapon_db_id,
                      submitted_quantity,
                      attack_mode)
```

**当前实现：**

1. 实时库存与目标域能力来自 `GetUnitData.unit_weapons`，不使用旧态势中可能恒为
   0 的 `weaponNumber.airNum/shipNum/subNum` 决定是否攻击；
2. 8×5 攻击 Actor 提供目标 ID、目标坐标和规则请求数量；随后执行层查询
   `GetWeaponFiringInfo`，从 `suitable_weapons` 中选择具体武器；
3. 候选必须同时满足 `weapon_db_id > 0`、总库存大于 0，并至少存在一条
   `fire_evaluations.can_fire=true` 且数量大于 0 的评估；
4. 候选排序依次优先：显式 `preferred_weapon_name` 名称匹配、导弹/鱼雷等制导武器、
   当前可立即发射数量、总库存；完全相同时选择较小 `weapon_db_id` 保证稳定结果；
5. 默认符号推理主流程不固定某个武器名称，而是按上述确定性规则选择具体
   `weapon_db_id`；独立调用攻击流水线时可传入 `preferred_weapon_name="YJ-18"`
   等名称偏好；
6. 当前自动攻击使用 `mode="manual"`。普通目标规则请求数量默认 2；导弹目标请求
   数量为 `min(2, 4-interceptors_launched)`；最终提交量还会取规则请求量与候选
   `ready_quantity` 的较小值；
7. `AttackTargetRequest` 明确包含 `attacker_unit_id`、`target_unit_id`、
   `weapon_db_id`、`quantity` 和 `mode`，因此具体发射武器不是由系统隐式决定；
8. `expected_weapon_type` 是目标域对应的业务类别和解释事实，不等同于最终的
   `weapon_db_id`；最终武器以执行层选出的候选为准；
9. `GetWeaponFiringInfo` 的 `can_fire` 属于最终攻击事实，不再等到执行层失败后才处理；
10. 无可立即发射武器时输出 `HOLD`，保留服务器原因并进入默认 10 帧冷却；
11. 冷却键为攻击方、稳定目标 GUID、Contact 质量和拒绝原因。Contact GUID、探测
   新鲜度等级、不确定区、目标类别或识别状态变化时立即重试；否则冷却到期再重试；
12. 实时推理在形成 `REQUEST_ATTACK` 前执行一次预检；执行层在 `AttackTarget` 前
    再次查询并选择武器，防止推理与提交之间库存或火控状态变化。

**推理路径示例：**

```text
输入：目标 S01 属于 SURFACE；我方平台具备反舰能力；规则请求数量=2
1. R-WPN-002：target_domain=SURFACE                            → 命中
2. expected_weapon_type=ANTI_SHIP_MISSILE                      → 业务类别
3. GetUnitData：对应目标域武器数量 2 > 0                        → 通过
4. R-FIRE-001：GetWeaponFiringInfo 返回 YJ-18，weapon_db_id=2001，
   total=2，ready=2，can_fire=true                              → 可用候选
5. 确定性选择 selected_weapon_name=YJ-18，weapon_db_id=2001     → 选中
6. mode=manual；submitted_quantity=min(2,2)=2                   → 数量确定
7. AttackTarget(attacker=A01,target=S01,weapon_db_id=2001,
   quantity=2,mode=manual)                                     → 提交
结论：REQUEST_ATTACK；expected_weapon_type=ANTI_SHIP_MISSILE
执行依据：R-WPN-002、R-FIRE-001、selected_weapon_db_id=2001
```

---

## R-AIM-001～002 攻击朝向对准

**计算：**

```text
target_bearing_deg = bearing(own_position, target_position)
heading_difference_deg = abs(normalize(own_heading_deg - target_bearing_deg))
结果范围：0°～180°
```

**规则：**

```text
IF own_platform_type == SHIP
THEN aimed_at_target = True

IF own_platform_type != SHIP
AND heading_difference_deg < 30°
THEN aimed_at_target = True

IF own_platform_type != SHIP
AND heading_difference_deg >= 30°
THEN aimed_at_target = False
AND immediate_attack_allowed = False
```

**已明确：** 只有水面舰艇不考虑朝向；飞机和潜艇都必须满足航向与目标方位差严格小于 30°。夹角恰好为 30° 时不允许立即攻击。

**推理路径示例（未对准）：**

```text
输入：我方飞机航向=10°；目标方位=45°；夹角=35°
1. R-AIM-001：我方不是舰艇                                     → 不适用豁免
2. R-AIM-002：35° < 30°                                       → 不通过
3. 我方是飞机，允许生成朝向目标的机动动作                       → 输出
结论：CHASE_AND_ALIGN
依据：R-AIM-002
```

---

## R-TGT-001 最近候选目标选择

**射程内候选目标的前置条件：**

```text
目标有效且仍存活/仍被探测
目标域能够识别
平台具备该目标域的有效射程和剩余武器
distance_km <= max_attack_range_km
并发和拦截弹限制未超限
```

朝向、攻击授权/通信安全和 `GetWeaponFiringInfo` 最终可发射性不参与这一阶段的
候选排序，而是在目标选定后依次检查。因此“进入射程内候选集合”不等于最终允许
发射。

**规则：**

```text
attack_candidates = 所有满足立即攻击约束的敌方目标
pursuit_candidates = []

IF attack_candidates is not empty
THEN selected_target = min(attack_candidates, key=(distance_km, target_id))
AND selection_state = EVALUATE_IMMEDIATE_ATTACK

IF attack_candidates is empty
AND own_platform_type == AIRCRAFT
THEN pursuit_candidates = 所有目标有效、具备打击能力、并发未满但超出射程的敌方目标

IF attack_candidates is empty
AND pursuit_candidates is not empty
THEN selected_target = min(pursuit_candidates, key=(distance_km, target_id))
AND selection_state = EVALUATE_PURSUIT

IF attack_candidates is empty
AND pursuit_candidates is empty
AND detected_targets is not empty
THEN selected_target = min(detected_targets, key=(distance_km, target_id))
AND selection_state = DIAGNOSTIC_ONLY

IF detected_targets is empty
THEN selected_target = None
AND continue_to_sonobuoy_patrol_search_or_hold = True
```

`EVALUATE_IMMEDIATE_ATTACK`、`EVALUATE_PURSUIT` 和 `DIAGNOSTIC_ONLY` 是本文描述
目标筛选阶段使用的内部状态，不是 `Conclusion` 枚举。诊断目标用于让后续规则明确
说明是授权、并发、射程、武器或拦截数量中的哪一项阻止了攻击。

**已明确：** 目标选择分为以下优先级：

1. 对射程内、满足上述目标选择前置条件且距离不超过 **50 km** 的导弹目标建立
   “优先拦截集合”，优先选择其中最近的导弹；
2. 没有上述导弹时，在其他射程内合法目标中选择最近目标；
3. 只有飞机在没有射程内合法目标时，才可选择最近的射程外目标进行追击；
4. 距离相同时按 `target_id` 排序，保证同一输入得到稳定结论。

50 km 是默认的导弹优先拦截距离，可通过运行参数 `--missile-intercept-distance-km` 调整。它与 R-MSL-001 的 5 km 紧急规避距离相互独立：导弹进入 50 km 后即可优先拦截，进入 5 km 后才触发横向 90° 规避。

**推理路径示例：**

```text
输入：T1=5 km（水面舰艇、合法）；M1=38 km（导弹、合法）；T3=4 km（超并发上限）
1. T1 通过类型、射程、并发检查                                  → 普通候选
2. M1 通过类型、射程、并发检查，且 38 km <= 50 km                → 优先拦截候选
3. T3 已有 3 个攻击者                                            → 排除
4. R-TGT-001：优先拦截集合非空，选择其中最近导弹 M1              → 命中
结论：selected_target=M1
依据：R-TGT-001、R-CON-001
```

---

## R-CON-001 同目标并发攻击限制

**规则：**

```text
IF active_attackers_on_target < 3
THEN concurrency_slot_available = True

IF active_attackers_on_target >= 3
THEN concurrency_slot_available = False
AND current_target_attack_allowed = False
AND try_next_nearest_target = True
```

**已明确：** 同一目标最多允许 3 个攻击者并发攻击。

**已明确：**

1. 在选定目标后、发出攻击请求前原子预留槽位；预留槽位也计入上限 3，避免多个实体同帧超限；
2. `AttackTarget` RPC 被服务端接受、执行状态为 `SUCCESS` 后，将本帧预留转为正式
   占用并记录 `slot_started_frame`；这只表示攻击请求已接受，不等同于已确认形成武器实体；
3. 态势中出现指向该目标的我方武器实体时，以武器 GUID 持续跟踪；由于态势没有
   直接提供发射平台字段，武器首次出现时按空间距离和本次齐射数量归属到具体攻击者；
4. 某一攻击者已确认在途的最后一发武器随后从态势中消失时，下一帧只释放该攻击者
   的并发槽位。该消失可能表示命中、被拦截或失效；其他攻击者仍有远距离在途武器
   时，不再锁住已经空出的槽位，允许新舰立即补位；
5. `AttackTarget` RPC 返回成功但连续 10 个后台数据帧仍未出现归属于该攻击者的
   武器实体时，释放该攻击者槽位，避免请求已接受但武器未形成或链路缺失造成长期锁定；
6. 系统明确反馈目标命中或目标删除时立即释放槽位；
7. 自 `slot_started_frame` 起满 600 个后台数据帧仍未被上述条件释放时，按硬超时兜底释放；
8. 攻击请求失败、尚未成功发射时，立即回滚预留槽位。

```text
IF system_feedback == MISSILE_HIT_TARGET
THEN release_concurrency_slot = True

IF own_weapon_was_seen_in_flight == True
AND own_weapon_is_present_now == False
THEN release_concurrency_slot = True
AND release_reason = IN_FLIGHT_WEAPON_DISAPPEARED

IF own_weapon_was_seen_in_flight == False
AND current_frame - slot_started_frame >= 10 frames
THEN release_concurrency_slot = True
AND release_reason = WEAPON_ENTITY_NOT_OBSERVED

IF current_frame - slot_started_frame >= 600 frames
THEN release_concurrency_slot = True
AND release_reason = ATTACK_SLOT_HARD_TIMEOUT
```

以上槽位按攻击者分别释放，只影响“本实体已有针对该目标的在途攻击”和同目标并发数判断。对于导弹目标，R-INT-001 的累计最多 4 发限制仍按目标生命周期累计，已发射数量不会因我方拦截弹消失而回减。

**推理路径示例：**

```text
输入：目标 T1 当前攻击者={A1,A2,A3}；新攻击者=A4
1. R-CON-001：active_attackers_on_target=3                      → 达到上限
2. 不为 A4 预留 T1 槽位                                         → 输出
3. A4 尝试选择下一个最近合法目标                                 → 输出
结论：优先改选其他合法目标；没有其他合法目标时输出 HOLD
依据：R-CON-001
```

---

## R-INT-001 拦截弹最多发射 4 发

**适用目标：** `target.is_weapon == True` 且目标被识别为空中导弹。

**规则：**

```text
IF target_is_missile == True
AND interceptors_launched < 4
THEN interceptor_launch_allowed = True

IF target_is_missile == True
AND interceptors_launched >= 4
THEN interceptor_launch_allowed = False
AND reason = INTERCEPTOR_LIMIT_REACHED
```

**已明确：** 针对一个导弹目标最多发射 4 发拦截弹。

**当前实现：** 普通攻击默认请求 2 发。攻击导弹时，本次请求数量为
`min(2, 4-interceptors_launched)`；同一帧后续实体会把已计划数量计入
`interceptors_launched`，避免同帧累计超过 4。只有 `AttackTarget` 执行状态为
`SUCCESS` 才写入跨帧累计状态，执行失败不记入累计数量。

**已明确：** “最多 4 发”是该导弹目标生命周期内的累计限制，不是同时在途限制。已经发射的数量不因拦截弹命中、失效或离开态势而回减；导弹目标被摧毁或永久丢失后清理该目标的计数状态。

**推理路径示例：**

```text
输入：目标 M1 是导弹；已发射拦截弹=4
1. 目标分类：M1 is_weapon=True 且属于 AIR                       → 通过
2. R-INT-001：interceptors_launched(4) < 4                      → 不通过
3. 禁止继续生成拦截弹攻击动作                                   → 输出
结论：HOLD
依据：R-INT-001
```

---

## R-BUOY-001 浮标部署

**输入事实：**

```text
inside_patrol_area
own_platform_role
has_patrol_mission
own_altitude_above_sea_m
sonobuoy_available
sonobuoy_deployment_due
sonobuoy_track_distance_km
sonobuoy_track_spacing_km
incoming_missile
```

**规则：**

```text
IF own_platform_role == PATROL_AIRCRAFT
AND has_patrol_mission == True
AND inside_patrol_area == True
AND 0m <= own_altitude_above_sea_m <= 500m
AND sonobuoy_available == True
AND (first_successful_deployment == True
     OR sonobuoy_track_distance_km >= sonobuoy_track_spacing_km)
AND incoming_missile == False
THEN deploy_sonobuoy_allowed = True
AND passiveOrActive = True
AND shallowOrDeep = True
ELSE deploy_sonobuoy_allowed = False
```

**已明确：**

1. 只有承担巡逻任务的巡逻机允许部署浮标；
2. 高度范围为距海面 `0～500 m`，0 m 和 500 m 两个边界均允许；
3. 高度取系统反馈的飞机距海面高度，不使用海拔高度自行推算；
4. 巡逻区域边界视为区域内，几何判断应采用包含边界的判断；
5. 紧急导弹规避时禁止部署浮标；
6. 本规则替代旧代码中的 `altitude > 150英尺 × 0.3048` 条件。
7. 只部署浅层主动声呐浮标：`passiveOrActive=True`、`shallowOrDeep=True`；
8. 不读取、不使用附近已有浮标及其有效期阻断部署；
9. 首次满足任务、高度和库存条件时立即部署；此后仅在该飞机自上次成功部署后沿航迹累计达到 `14.816 km` 时再次部署；
10. `14.816 km` 为原 `7.408 km` 间距的 2 倍；RPC 部署失败时不清零累计航程；
11. 航迹采用相邻态势点的分段距离累计，因此飞机绕航返回旧位置时，仍可按累计航程再次部署。

**推理路径示例（允许部署）：**

```text
输入：巡逻机 H1；承担巡逻任务；位于巡逻区边界；距海面高度=480m；浮标数量=3；首次部署；无导弹威胁
1. R-BUOY-001：PATROL_AIRCRAFT 且 has_patrol_mission=True        → 通过
2. R-BUOY-001：inside_patrol_area=True（边界计入）               → 通过
3. R-BUOY-001：0 <= 480 <= 500                                  → 通过
4. R-BUOY-001：sonobuoy_available=True                           → 通过
5. R-BUOY-001：first_successful_deployment=True                  → 通过
6. R-BUOY-001：incoming_missile=False                            → 通过
结论：DEPLOY_SONOBUOY，浅层主动（True, True）
依据：R-BUOY-001
```

**推理路径示例（拒绝部署）：**

```text
输入：实体 H1；位于巡逻区；高度=650m；浮标数量=3
1. R-BUOY-001：inside_patrol_area=True                           → 通过
2. R-BUOY-001：0 <= 650 <= 500                                  → 不通过
结论：不生成 DEPLOY_SONOBUOY；巡逻由系统管理，符号推理不下发航路
依据：R-BUOY-001；未满足条件=ALTITUDE_TOO_HIGH
```

---

## 巡逻路径执行边界（不进入推理路径）

**输入事实：**

```text
mission_id
has_patrol_mission
mission_area_points
own_position
```

**规则：**

```text
IF unit_mission_id is empty
AND count(valid_patrol_missions) == 1
AND own_platform_role == ASW_AIRCRAFT
THEN mission_id = the_only_patrol_mission_id

IF has_patrol_mission == True
THEN inside_patrol_area = point_in_polygon_including_boundary(
    own_position, mission_area_points
)
AND generated_patrol_route = NONE
AND setUnitRoutew_allowed = False
```

**推理路径示例：**

```text
输入：反潜巡逻机 P1；态势 missionId 为空；getMissionList 只有一个巡逻任务 M001
1. getMissionList：有效巡逻任务数量=1                         → 通过
2. 平台识别：P1 是反潜巡逻机                                  → 通过
3. 自动使用 M001 的区域顶点判断 P1 是否在区内                 → 生成区域事实
4. 不生成航点，不调用 setUnitRoutew                           → 通过
结论：只继续判断 R-BUOY-001，不干预系统巡逻路径
说明：这是系统与符号算法的职责边界，不记录为规则命中，也不显示在推理路径中。
```

**执行约定：** 系统现有巡逻任务负责航路、速度和高度。符号推理只读取区域并在
满足 R-BUOY-001 时调用浮标部署 RPC，不为巡逻任务调用 `setUnitRoutew`。

---

## R-SEARCH-001 / R-SEARCH-002 普通搜索

```text
IF detected_target_count == 0
AND (has_radar_sensor == True OR has_sonar_sensor == True)
THEN enable_radar = has_radar_sensor
AND enable_sonar = has_sonar_sensor
AND conclusion = SEARCH                         [R-SEARCH-001]

IF detected_target_count == 0
AND has_radar_sensor == False
AND has_sonar_sensor == False
THEN do_not_send_sensor_control = True
AND conclusion = HOLD                           [R-SEARCH-002]
```

传感器能力来自 `GetUnitData.unir_sensor_params` 的装备清单；`active_status` 仅是
当前开关状态，不用于判断平台是否具备该传感器。

---

## 5. 综合攻击推理路径

### 5.1 允许攻击水面舰艇

```text
输入：
  我方 A01=飞机，航向 80°
  目标 S01=水面舰艇，方位 90°，距离 60km，高度 0m
  rangeStrike_Surface=100km
  反舰导弹数量=2
  S01 当前攻击者数量=1

1. 威胁检查：A01 无直接来袭导弹                              → 不触发规避
2. R-TGT-001：选择候选目标 S01                                 → 命中
3. R-VAL-001：可操纵且通信安全                                 → 未触发阻断
4. R-CON-001：当前攻击者 1 < 3                                 → 未触发阻断
5. R-INT-001：S01 不是导弹目标                                 → 未触发限制
6. R-RNG-001：rangeStrike_Surface=100km > 0                    → 未触发阻断
7. R-WPN-002：反舰武器数量 2 > 0                               → 未触发阻断
8. 射程边界：60km <= 100km                                     → 射程内
9. R-AIM-002：|80°-90°|=10° < 30°                              → 未触发阻断
10. R-FIRE-001：首次 GetWeaponFiringInfo 存在可立即发射候选     → 未触发阻断
11. R-WPN-002：全部约束满足                                    → 命中 REQUEST_ATTACK
12. R-ALT-001：派生 attack_altitude_level=1                    → 写入动作参数
13. 执行层再次查询并选定具体 weapon_db_id                       → 防止状态变化
14. AttackTarget 明确提交 weapon_db_id、quantity=2、mode=manual → 输出
15. RPC 接受后跟踪我方在途武器；消失即释放，10/600 帧作为两级兜底 → 等待反馈
结论：REQUEST_ATTACK S01
推理路径依据：R-MSL-001、R-TGT-001、R-VAL-001、R-CON-001、R-INT-001、
R-RNG-001、R-WPN-002、R-AIM-002、R-FIRE-001
动作参数依据：R-ALT-001
```

### 5.2 射程内但未对准

```text
输入：目标在射程内；我方飞机航向与目标方位差=42°

1. 射程检查                                                      → 通过
2. R-AIM-002：42° < 30°                                         → 不通过
3. 禁止立即攻击                                                  → 输出
4. 因我方是飞机，生成对准/追击航路，高度使用 R-ALT-001           → 输出
结论：CHASE_AND_ALIGN
依据：R-AIM-002
```

### 5.3 多约束同时失败时按优先级短路

```text
输入：我方舰艇；仅有目标 T01；T01 距离超射程且已有 3 个攻击者

1. 目标筛选：没有合法攻击/追击候选，保留最近的 T01 作为诊断目标    → 诊断
2. R-TGT-001：选择 T01                                          → 命中
3. R-VAL-001：授权和安全条件正常                                 → 未触发阻断
4. R-CON-001：并发数达到 3                                      → 命中
5. 输出 HOLD；本帧到此结束                                      → 短路
6. R-RNG-003/R-RNG-004                                          → 本帧不再评估
结论：HOLD
依据：R-CON-001；原因：同目标并发攻击槽位已满
```

如果存在另一个符合前置条件的合法目标，目标筛选阶段会优先选择该合法目标，而不是
选择 T01 后再输出“改选目标”。如果所有目标都不合法，则保留最近目标做一次诊断，并
按 `R-VAL → R-CON → R-INT → R-RNG → R-WPN` 顺序在首个阻断点返回。

### 5.4 飞机在射程外追击后攻击

```text
输入：我方 A01=飞机；目标 S01=水面舰艇；距离 120km；最大对海射程 100km

1. R-TGT-001：当前没有射程内合法目标                            → 通过
2. R-RNG-004：120km > 100km 且 own_platform_type=AIRCRAFT      → 命中
3. 禁止立即发射并允许追击                                       → 输出
4. 输出 CHASE_TO_RANGE，不输出发射请求                          → 输出
5. 系统反馈新距离=95km                                          → 重新开始完整推理
6. 导弹威胁、目标有效性、并发、射程、朝向和弹药检查全部通过      → 通过
7. 输出 REQUEST_ATTACK；执行层选定具体 weapon_db_id 和数量后调用 AttackTarget → 输出
结论：先 CHASE_TO_RANGE，进入射程并复检通过后 REQUEST_ATTACK
依据：R-TGT-001、R-RNG-001、R-RNG-004、R-AIM-002、R-CON-001、R-WPN-002
```

### 5.5 最大射程边界与系统反馈

```text
输入：飞机 A01 到目标距离=100km；最大射程=100km；其他约束均通过

1. R-RNG-001：平台具备目标域有效射程                             → 通过
2. 射程边界：100km <= 100km                                    → 允许进入后续攻击检查
3. 输出 REQUEST_ATTACK                                          → 输出
4. AttackTarget RPC 被接受                                      → execution_status=SUCCESS
   或预检/提交失败                                                → execution_status=FAILED
5. 后续态势出现关联武器实体才表示确实形成在途武器                 → 持续跟踪
结论：规则允许和 RPC 接受均不等于最终命中，最终结果以后续态势反馈为准
依据：R-RNG-001、射程边界条件、系统执行反馈
```

## 6. 推理路径统一输出格式

当前代码的解释对象为 `Decision`，字段包括：

```text
conclusion             最终 Conclusion 枚举
rule_id                决定最终结论的规则编号
reason                 最终原因
matched_facts          决定性事实
inference_path         按执行顺序记录的 InferenceStep 列表
actions                8×5 动作矩阵
target_id              目标 Contact ID；无目标时为 None
expected_weapon_type   业务武器类别；不是具体 weapon_db_id
```

每条 `InferenceStep` 包含 `rule_id、rule、matched、evidence`。实际
`Decision.explanation` 为可读文本，例如：

```text
推理路径：
1. R-MSL-001 [未命中] 5 km 内确认或推断为直接指向本实体的敌方导弹属于来袭威胁；
   事实：incoming_missile=False，distance_km=60.000，threshold_km=5
2. R-TGT-001 [命中] 从合法候选目标中选择最近目标；
   事实：target_id=S01，detected_target_count=2，distance_km=60.000
3. R-CON-001 [未命中] 同一目标最多允许 3 个不同攻击者占用或预留并发槽位；
   事实：active_attackers=1，limit=3，slot_available=True
4. R-FIRE-001 [未命中] 最终攻击前必须通过 GetWeaponFiringInfo 可立即发射性预检；
   事实：fire_control_checked=True，fire_control_available=True
5. R-WPN-002 [命中] 射程、朝向、并发和弹药条件均满足，向系统提交发射请求；
   事实：compatible_weapon_count=2，attack_quantity=2
结论：REQUEST_ATTACK；决定规则：R-WPN-002；依据：within_attack_range=True，...
```

执行反馈不写入 `Decision` 本身，而是在 `SymbolicStepResult.execution_status` 中单独
记录：未下发为 `DRY_RUN`，无动作请求为 `NOT_REQUESTED`，执行结果为 `SUCCESS`、
`FAILED` 或 `UNKNOWN`。具体武器选择保存在 `WeaponFirePrecheck` 和
`AttackPipelineResult` 中，包括 `weapon_db_id、weapon_name、ready_quantity、
requested_quantity、submitted_quantity、mode、candidate_evidence`。

默认日志只保留启动状态和实际成功操作的 `INFO`；完整命中/未命中路径使用
`--verbose-reasoning` 后以 `DEBUG` 输出。可解释性数据始终存在于 `Decision`，日志
级别只影响是否显示，不影响推理路径生成。

## 7. 人工确认结果

- [x] 来袭导弹规避距离阈值：5 km，5 km 边界触发。
- [x] 导弹优先拦截距离：默认 50 km，可通过 `--missile-intercept-distance-km` 调整；不改变 5 km 紧急规避阈值。
- [x] 时间压缩冷却：每处理一份态势按当前 UI 倍率推进冷却时钟；50 倍速每帧
  扣减 50，暂停不扣减，Turbo 暂按 60 倍；Contact 连续丢失防抖仍按实际态势帧。
- [x] 无导弹目标 ID 时：允许采用航向和距离推断是否指向我方。
- [x] 规避方向：固定向右转 90°。
- [x] 普通高度等级：0=海面，1=最低高度层，3=中高空层，5=最高高度层。
- [x] 潜艇高度等级例外：1=浅层、3=中层、5=高层，均为潜艇专用水下层级，具体深度由系统映射。
- [x] 空中目标高度分界：继续使用 5000 m。
- [x] 最大射程边界：距离等于最大射程时允许提交攻击请求，最终成功与否以系统反馈为准。
- [x] 潜艇攻击朝向：必须满足与目标方位差严格小于 30°。
- [x] 超距追击：只有飞机允许；进入射程后重新检查并发、射程、朝向和弹药等条件。
- [x] 可控性接口：红方存活的非武器实体默认可控，不再使用可能未填写的
  `isCanManaged=false` 阻断命令。
- [x] 武器接口：实时模式不使用不可靠的 `weaponNumber.airNum/shipNum/subNum`
  判断能否攻击，而是通过 `GetUnitData.unit_weapons` 获得真实剩余数量、武器类型
  和对应域射程；推理最终决策前通过 `GetWeaponFiringInfo` 获得当前目标的候选
  武器和 `can_fire` 评估，无可发射武器时 `HOLD`，通过后才将选中的
  `weapon_db_id`、数量和 `mode=manual` 提交给 `AttackTarget`。
- [x] 具体武器选择：默认按名称偏好（若显式提供）、制导武器优先、立即可发数量、
  总库存和稳定 ID 顺序选择；当前自动流程会明确指定具体 `weapon_db_id`，不是只
  选择“是否发射”。
- [x] 发射数量：普通目标默认请求 2；导弹目标按累计最多 4 发计算本次数量；
  `manual` 模式最终提交量不超过候选武器的当前可发数量。
- [x] 攻击拒绝冷却：相同攻击方、稳定目标、Contact 质量和拒绝原因默认冷却
  10 帧；Contact 质量变化立即重试，否则到期重试。
- [x] Contact 丢失防抖：通过稳定实体 GUID 关联变化的 Contact GUID，默认连续
  丢失 3 帧才取消攻击，重新发现立即清零。
- [x] 同目标并发槽位：按武器 GUID 和发射平台分别跟踪；某平台最后一发在途武器
  消失时只释放该平台槽位，其他平台仍在途不阻止补位；`AttackTarget` RPC 被接受
  后 10 帧仍未观察到归属武器时释放；命中/目标删除时释放；600 帧仅作最终兜底。
- [x] 拦截弹数量：同一导弹目标生命周期内累计最多 4 发。
- [x] 浮标高度：距海面 0～500 m，0 m 和 500 m 边界均允许，采用系统反馈高度。
- [x] 巡逻区域：边界视为区域内；只有承担巡逻任务的巡逻机允许部署浮标。
- [x] 可解释性输出：完整推理路径始终保存在 `Decision.inference_path`；默认日志
  保持安静，使用 `--verbose-reasoning` 显示逐条命中、未命中和事实依据。
