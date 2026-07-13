# 符号推理业务规则（已确认稿）

> 状态：第一版业务口径已于 2026-07-13 完成人工确认，可据此编写算法和测试。
>
> 标记说明：
>
> - **已明确**：来自当前需求，可直接作为规则口径。
> - **建议口径**：为了让规则可执行、可测试而补充、后续仍可调整的实现细节。
> - **待确认**：尚未签认的新增参数或业务歧义。

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
输出结论（规避/攻击/追击/部署浮标/保持）
  ↓
生成逐条规则推理路径
```

## 2. 术语与事实定义

### 2.1 平台类型

| 编码 | 类型 | 来源字段 |
|---:|---|---|
| 0 | 飞机 | `unitCategory` / `unitType` / `Icon2D` |
| 1 | 水面舰艇 | `unitCategory` / `unitType` / `Icon2D` |
| 2 | 潜艇 | `unitCategory` / `unitType` / `Icon2D` |
| 3 | 地面目标/设施 | `unitCategory` / `unitType` / `Icon2D` |
| 6 | 武器 | `unitCategory`、`IsWeapon`、`Icon2D` |

**建议口径：** 我方视角样例中，部分接触目标的 `unitCategory` 和 `unitType` 统一为 0，因此目标类型识别顺序采用：

```text
Icon2D 路径 → IsWeapon → unitCategory/unitType → altitude 高度兜底
```

### 2.2 目标域

| 目标域 | 典型目标 | 对应射程字段 | 对应武器类型 |
|---|---|---|---|
| AIR | 飞机、空中导弹 | `rangeStrike_Air` | 防空导弹/空空导弹 |
| SURFACE | 水面舰艇、海平面目标 | `rangeStrike_Surface` | 反舰导弹 |
| SUBMARINE | 潜艇、水下鱼雷 | `rangeStrike_Submarine` | 反潜武器/鱼雷 |
| LAND | 地面设施、车辆 | `rangeStrike_Land` | 对地武器 |

> 需求中的“鱼类”按“鱼雷”处理。

### 2.3 关键事实

```text
own_id                         我方实体 ID
own_platform_type              我方平台类型
own_position                   我方经度、纬度、高度
own_heading_deg                我方当前航向角
own_altitude_above_sea_m       系统反馈的我方实体距海面高度
target_id                      目标 ID；接触目标优先使用 contactGuid
target_domain                  AIR / SURFACE / SUBMARINE / LAND
target_position                目标经度、纬度、高度
distance_km                    我方到目标的三维距离
target_bearing_deg             我方位置指向目标的方位角
heading_difference_deg         航向与目标方位的最小夹角
max_attack_range_km            针对目标域选择的最大攻击距离
expected_weapon_type           规则期望的武器类型；仅用于校验和解释
compatible_weapon_count        对应类型武器的可用数量
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
```

## 3. 规则优先级

| 优先级 | 规则组 | 说明 |
|---:|---|---|
| P0 | 事实非法/缺失 | 关键事实缺失时禁止危险动作 |
| P1 | 来袭导弹规避 | 直接威胁时覆盖攻击、追击等普通动作 |
| P2 | 目标选择 | 先得到合法候选目标，再判断攻击 |
| P3 | 并发与拦截弹限制 | 防止同一目标攻击者或拦截弹数量超限 |
| P4 | 射程与朝向约束 | 不满足则禁止立即攻击；只有飞机可转为追击/对准 |
| P5 | 武器和攻击高度选择 | 满足攻击条件后确定攻击参数 |
| P6 | 浮标部署 | 独立任务动作；不得与紧急规避冲突 |

**已明确：** P1 来袭导弹规避优先于普通攻击。

**建议口径：** 同一实体同一帧只输出一个主动作；紧急规避时关闭攻击、浮标部署和普通航路动作。

---

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

> 第 4 项的 30° 是为了让“指向我方”可计算而设置的建议默认值，后续如系统提供更精确的制导信息，应优先使用系统目标 ID。

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
`normalize(missile_heading + 90°)`；不进行随机左右选择。

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

## R-ALT-001 攻击高度调整

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

| 目标 | 我方平台 | 当前代码建议等级 |
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

**推理路径示例：**

```text
输入：我方=A01(飞机)；目标=S01(水面舰艇)；目标高度=0 m
1. 目标分类：Icon2D/高度判定 target_domain=SURFACE             → 通过
2. R-ALT-001：SURFACE + AIRCRAFT                               → 命中
3. 设置 attack_altitude_level=1                                → 输出
结论：攻击高度等级 1
依据：R-ALT-001
```

**已明确：** 5000 m 继续作为空中目标高度分界；等级对应的实际控制高度由系统负责映射，推理路径保留“等级 + 业务含义”。

---

## R-RNG-001～004 攻击距离约束

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

IF distance_km > max_attack_range_km
AND own_platform_type == AIRCRAFT
THEN immediate_attack_allowed = False
AND pursuit_allowed = True
AND conclusion = CHASE_TO_RANGE

IF distance_km > max_attack_range_km
AND own_platform_type != AIRCRAFT
THEN immediate_attack_allowed = False
AND pursuit_allowed = False
AND reason = OUT_OF_RANGE

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
3. R-RNG-002：155 > 148.16                                     → 命中
4. 禁止生成攻击动作                                             → 输出
结论：若 D01 是舰艇则 HOLD 或 RESELECT_TARGET；不允许超距追击
依据：R-RNG-001、R-RNG-002
```

**推理路径示例（飞机超距追击）：**

```text
输入：我方飞机 A01；目标 S01；distance=120 km；rangeStrike_Surface=100 km
1. 目标分类：S01 属于 SURFACE                                  → 通过
2. R-RNG-001：选择 rangeStrike_Surface=100 km                  → 通过
3. R-RNG-002：120 > 100                                        → 禁止立即攻击
4. 我方平台类型为 AIRCRAFT                                     → 允许追击
5. 生成追击动作；不生成攻击动作                                 → 输出
6. 进入 100 km 范围后重新执行全部攻击约束                       → 等待系统新态势
结论：CHASE_TO_RANGE
依据：R-RNG-001～004
```

---

## R-WPN-001～003 武器适配判断与发射请求

| 目标域 | 选择武器 | 规则 ID |
|---|---|---|
| AIR（飞机、空中导弹） | 防空导弹；飞机平台可使用空空导弹 | R-WPN-001 |
| SURFACE（水面舰艇） | 反舰导弹 | R-WPN-002 |
| SUBMARINE（潜艇、水下鱼雷） | 反潜武器/鱼雷 | R-WPN-003 |

**规则：**

```text
IF target_domain == AIR
THEN expected_weapon_type = AIR_DEFENCE_OR_AIR_TO_AIR_MISSILE

IF target_domain == SURFACE
THEN expected_weapon_type = ANTI_SHIP_MISSILE

IF target_domain == SUBMARINE
THEN expected_weapon_type = ANTI_SUBMARINE_WEAPON_OR_TORPEDO

IF compatible_weapon_count <= 0
THEN attack_allowed = False
AND reason = NO_MATCHING_WEAPON

IF compatible_weapon_count > 0
AND all_other_attack_constraints_passed == True
THEN fire_request = True
AND weapon_selection_delegated_to_system = True
```

**已明确：**

1. 算法能够获得对应类型武器的剩余数量；
2. 执行接口只能选择“是否发射”，不能显式指定武器类型；
3. `expected_weapon_type` 用于规则匹配、库存检查和解释输出，真正发射何种武器由系统根据目标自动选择；
4. 系统反馈发射失败时，最终执行状态记为 `ATTACK_FAILED`，并保留系统返回原因。

**推理路径示例：**

```text
输入：目标 S01 属于 SURFACE；我方平台具备反舰能力；反舰导弹数量=2
1. R-WPN-002：target_domain=SURFACE                            → 命中
2. 期望系统使用 ANTI_SHIP_MISSILE                              → 通过
3. 对应武器数量 2 > 0                                         → 通过
4. 向系统提交 fire_request=True；武器选择交由系统               → 输出
结论：REQUEST_ATTACK；expected_weapon_type=ANTI_SHIP_MISSILE
依据：R-WPN-002
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

## R-TGT-001 最近合法目标优先

**立即攻击候选目标必须同时满足：**

```text
目标有效且仍存活/仍被探测
目标域能够识别
平台具备该目标域的打击能力
distance_km <= max_attack_range_km
并发和拦截弹限制未超限
```

**规则：**

```text
attack_candidates = 所有满足立即攻击约束的敌方目标
pursuit_candidates = []

IF attack_candidates is not empty
THEN selected_target = min(attack_candidates, key=(distance_km, target_id))
AND conclusion = EVALUATE_IMMEDIATE_ATTACK

IF attack_candidates is empty
AND own_platform_type == AIRCRAFT
THEN pursuit_candidates = 所有目标有效、具备打击能力、并发未满但超出射程的敌方目标

IF attack_candidates is empty
AND pursuit_candidates is not empty
THEN selected_target = min(pursuit_candidates, key=(distance_km, target_id))
AND conclusion = CHASE_TO_RANGE

IF attack_candidates is empty
AND pursuit_candidates is empty
THEN conclusion = NO_ATTACKABLE_TARGET
```

**已明确：** 攻击范围内优先选择最近合法目标；只有飞机在没有射程内合法目标时，才可选择最近的射程外目标进行追击。距离相同时按 `target_id` 排序，保证同一输入得到稳定结论。

**推理路径示例：**

```text
输入：T1=20 km（合法）；T2=12 km（合法）；T3=5 km（超并发上限）
1. T1 通过类型、射程、并发检查                                  → 候选
2. T2 通过类型、射程、并发检查                                  → 候选
3. T3 已有 3 个攻击者                                            → 排除
4. R-TGT-001：在 T1、T2 中选择最近目标 T2                        → 命中
结论：selected_target=T2
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
2. 系统确认我方导弹成功发射后，将预留槽位转为正式占用，并记录 `slot_started_at`；
3. 系统反馈该导弹命中目标时立即释放槽位；
4. 自 `slot_started_at` 起满 10 分钟仍未收到命中反馈时，按超时释放槽位；
5. 攻击请求失败、尚未成功发射时，回滚预留槽位；
6. 导弹未命中或状态不明时，不提前释放，直到命中反馈或 10 分钟超时。

```text
IF system_feedback == MISSILE_HIT_TARGET
THEN release_concurrency_slot = True

IF current_time - slot_started_at >= 10 minutes
THEN release_concurrency_slot = True
AND release_reason = ATTACK_SLOT_TIMEOUT
```

**推理路径示例：**

```text
输入：目标 T1 当前攻击者={A1,A2,A3}；新攻击者=A4
1. R-CON-001：active_attackers_on_target=3                      → 达到上限
2. 不为 A4 预留 T1 槽位                                         → 输出
3. A4 尝试选择下一个最近合法目标                                 → 输出
结论：RESELECT_TARGET 或 HOLD
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

**建议口径：** 发射前先原子增加“预留数量”，执行失败再回滚，避免同帧并发超发。

**已明确：** “最多 4 发”是该导弹目标生命周期内的累计限制，不是同时在途限制。已经发射的数量不因拦截弹命中、失效或离开态势而回减；导弹目标被摧毁或永久丢失后清理该目标的计数状态。

**推理路径示例：**

```text
输入：目标 M1 是导弹；已发射拦截弹=4
1. 目标分类：M1 is_weapon=True 且属于 AIR                       → 通过
2. R-INT-001：interceptors_launched(4) < 4                      → 不通过
3. 禁止继续生成拦截弹攻击动作                                   → 输出
结论：HOLD_INTERCEPT_FIRE
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
incoming_missile
```

**规则：**

```text
IF own_platform_role == PATROL_AIRCRAFT
AND has_patrol_mission == True
AND inside_patrol_area == True
AND 0m <= own_altitude_above_sea_m <= 500m
AND sonobuoy_available == True
AND incoming_missile == False
THEN deploy_sonobuoy_allowed = True
ELSE deploy_sonobuoy_allowed = False
```

**已明确：**

1. 只有承担巡逻任务的巡逻机允许部署浮标；
2. 高度范围为距海面 `0～500 m`，0 m 和 500 m 两个边界均允许；
3. 高度取系统反馈的飞机距海面高度，不使用海拔高度自行推算；
4. 巡逻区域边界视为区域内，几何判断应采用包含边界的判断；
5. 紧急导弹规避时禁止部署浮标；
6. 本规则替代旧代码中的 `altitude > 150英尺 × 0.3048` 条件。

**推理路径示例（允许部署）：**

```text
输入：巡逻机 H1；承担巡逻任务；位于巡逻区边界；距海面高度=480m；浮标数量=3；无导弹威胁
1. R-BUOY-001：PATROL_AIRCRAFT 且 has_patrol_mission=True        → 通过
2. R-BUOY-001：inside_patrol_area=True（边界计入）               → 通过
3. R-BUOY-001：0 <= 480 <= 500                                  → 通过
4. R-BUOY-001：sonobuoy_available=True                           → 通过
5. R-BUOY-001：incoming_missile=False                            → 通过
结论：DEPLOY_SONOBUOY
依据：R-BUOY-001
```

**推理路径示例（拒绝部署）：**

```text
输入：实体 H1；位于巡逻区；高度=650m；浮标数量=3
1. R-BUOY-001：inside_patrol_area=True                           → 通过
2. R-BUOY-001：0 <= 650 <= 500                                  → 不通过
结论：HOLD_SONOBUOY
依据：R-BUOY-001；拒绝原因=ALTITUDE_TOO_HIGH
```

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
2. 目标分类：S01 属于 SURFACE                                  → 通过
3. R-RNG：60km <= 100km                                        → 通过
4. R-CON：当前攻击者 1 < 3                                    → 通过
5. R-AIM：|80°-90°|=10° < 30°                                 → 通过
6. R-WPN-002：反舰导弹数量 2 > 0；系统负责选择武器              → 通过
7. R-ALT-001：飞机攻击海平面目标，高度等级=1                   → 通过
8. 生成攻击请求并预留并发槽位                                  → 输出
9. 系统反馈发射成功后开始 10 分钟槽位计时                       → 等待反馈
结论：REQUEST_ATTACK S01
依据：R-RNG、R-CON-001、R-AIM-002、R-WPN-002、R-ALT-001
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

### 5.3 多约束同时失败

```text
输入：我方舰艇；目标距离超射程；目标已有 3 个攻击者

1. R-CON-001：并发数达到 3                                      → 不通过
2. R-RNG：目标超出射程                                           → 不通过
3. 舰艇不允许超距追击                                            → 不通过
4. 该目标从立即攻击候选集合中删除                                → 输出
5. 尝试选择下一个最近合法目标；没有则保持                         → 输出
结论：RESELECT_TARGET 或 HOLD
依据：R-CON-001、R-RNG
```

### 5.4 飞机在射程外追击后攻击

```text
输入：我方 A01=飞机；目标 S01=水面舰艇；距离 120km；最大对海射程 100km

1. R-TGT-001：当前没有射程内合法目标                            → 通过
2. R-RNG：120km > 100km                                        → 禁止立即攻击
3. own_platform_type=AIRCRAFT                                  → 允许追击
4. 输出 CHASE_TO_RANGE，不输出发射请求                          → 输出
5. 系统反馈新距离=95km                                          → 重新开始完整推理
6. 导弹威胁、目标有效性、并发、射程、朝向和弹药检查全部通过      → 通过
7. 输出 REQUEST_ATTACK；具体武器由系统选择                       → 输出
结论：先 CHASE_TO_RANGE，进入射程并复检通过后 REQUEST_ATTACK
依据：R-TGT-001、R-RNG-001～004、R-AIM、R-CON-001、R-WPN-002
```

### 5.5 最大射程边界与系统反馈

```text
输入：飞机 A01 到目标距离=100km；最大射程=100km；其他约束均通过

1. R-RNG：100km <= 100km                                       → 允许提交攻击请求
2. 输出 REQUEST_ATTACK                                          → 输出
3. 系统反馈发射成功                                              → execution_status=SUCCESS
   或系统反馈发射失败                                            → execution_status=FAILED
结论：规则允许不等于实际成功，最终结果以系统反馈为准
依据：R-RNG-001～004、系统执行反馈
```

## 6. 推理路径统一输出格式

每个结论建议至少输出以下结构：

```json
{
  "entity_id": "A01",
  "target_id": "S01",
  "conclusion": "REQUEST_ATTACK",
  "execution_status": "PENDING_SYSTEM_FEEDBACK",
  "expected_weapon_type": "ANTI_SHIP_MISSILE",
  "weapon_selection": "SYSTEM",
  "decisive_rule_id": "R-WPN-002",
  "reason": "目标为水面舰艇，且射程、朝向、并发和武器条件均满足",
  "path": [
    {
      "rule_id": "R-RNG-001",
      "matched": true,
      "evidence": ["distance_km=60", "rangeStrike_Surface=100"]
    },
    {
      "rule_id": "R-CON-001",
      "matched": true,
      "evidence": ["active_attackers_on_target=1", "limit=3"]
    },
    {
      "rule_id": "R-AIM-002",
      "matched": true,
      "evidence": ["heading_difference_deg=10", "limit<30"]
    },
    {
      "rule_id": "R-WPN-002",
      "matched": true,
      "evidence": ["target_domain=SURFACE", "anti_ship_missiles=2", "weapon_selection=SYSTEM"]
    },
    {
      "rule_id": "R-ALT-001",
      "matched": true,
      "evidence": ["own_platform=AIRCRAFT", "target_domain=SURFACE", "altitude_level=1"]
    }
  ]
}
```

## 7. 人工确认结果

- [x] 来袭导弹规避距离阈值：5 km，5 km 边界触发。
- [x] 无导弹目标 ID 时：允许采用航向和距离推断是否指向我方。
- [x] 规避方向：固定向右转 90°。
- [x] 普通高度等级：0=海面，1=最低高度层，3=中高空层，5=最高高度层。
- [x] 潜艇高度等级例外：1=浅层、3=中层、5=高层，均为潜艇专用水下层级，具体深度由系统映射。
- [x] 空中目标高度分界：继续使用 5000 m。
- [x] 最大射程边界：距离等于最大射程时允许提交攻击请求，最终成功与否以系统反馈为准。
- [x] 潜艇攻击朝向：必须满足与目标方位差严格小于 30°。
- [x] 超距追击：只有飞机允许；进入射程后重新检查并发、射程、朝向和弹药等条件。
- [x] 武器接口：可获得武器数量；算法只能选择是否发射，具体武器由系统选择。
- [x] 同目标并发槽位：我方导弹命中时释放，或自成功发射起 10 分钟超时释放。
- [x] 拦截弹数量：同一导弹目标生命周期内累计最多 4 发。
- [x] 浮标高度：距海面 0～500 m，0 m 和 500 m 边界均允许，采用系统反馈高度。
- [x] 巡逻区域：边界视为区域内；只有承担巡逻任务的巡逻机允许部署浮标。
