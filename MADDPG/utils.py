# utils.py
import torch
import pdb
import math
import random
import numpy as np
from entity import AIR_STATUS_MAP
from shapely.ops import nearest_points
from shapely.geometry import Polygon, Point, LineString
import math
import hashlib

from typing import Optional, Tuple

from entity import get_env_entity_ids, get_coordinate_from_encoded_data
from maddpg_rule_guard import missile_points_at_entity

# 根据你的 encoded_data 列含义自行调整
COL_TYPE = 13  # 0=Aircraft 1=Ship 2=Sub …
COL_AIR_STATUS = 23

# 开场前 3 帧强制侦察起飞
BOOTSTRAP_FRAMES = 10

DIR_FRONT = 6  # 方向码与 WayPointMoveActor 的映射一致


def bootstrap_recon(actions: np.ndarray,
                    encoded_state: torch.Tensor,  # [1,E,30]
                    mask: torch.Tensor,  # [1,E]
                    cur_step: int) -> np.ndarray:
    if cur_step >= BOOTSTRAP_FRAMES:
        return actions

    state = encoded_state[0]
    msk = mask[0]

    for ent_idx in range(msk.shape[0]):
        if not msk[ent_idx]:  # False = padding / 敌方
            continue

        typ = int(state[ent_idx, COL_TYPE].item())
        a_st = int(state[ent_idx, COL_AIR_STATUS].item())

        # 'TAXYING_TO_TAKE_OFF': 1,
        # 'TAKING_OFF': 2,

        if typ == 0 and a_st in [1, 2]:  # !!!!!这里要根据实际情况进行修改
            # 0  起飞
            actions[0, ent_idx, 0] = 0.8

            # 2  WayPoint 前出
            actions[2, ent_idx, 0] = 0.8  # 执行概率
            actions[2, ent_idx, 1] = DIR_FRONT  # direction_index
            actions[2, ent_idx, 2] = 0.0  # NONE
            actions[2, ent_idx, 3] = 3  # 速度量化
            actions[2, ent_idx, 4] = 2  # 高度量化

            # 5  打开雷达
            actions[5, ent_idx, 0] = 0.8  # SensorControl_prob
            actions[5, ent_idx, 1] = 0.8  # radar on
            actions[5, ent_idx, 2] = 0.0
            actions[5, ent_idx, 3] = 0.0
    return actions


# DIR_N ,DIR_NE,DIR_E,DIR_SE,DIR_S,DIR_SW,DIR_W,DIR_NW = range(8)
# def u_patrol_for_asw(actions: np.ndarray,
#                           encoded_state: torch.Tensor,
#                           mask: torch.Tensor,
#                           raw_data: list,
#                           cur_step: int,
#                           u_leg_steps: int = 40,
#                           fuel_rtb_th: float = 0.30):
#     """
#     左侧『U』巡逻：北 → 西 → 南，循环往复。
#     """
#     PATROL_AIRCRAFT_TYPES = {14}                   # 这里只保留 MPA，如需其它机型再加
#     state, mask = encoded_state[0], mask[0]
#     phase   = (cur_step // u_leg_steps) % 3        # 0=N ,1=W ,2=S

#     # ── 方向常量（与 WayPointMoveActor 保持一致）──
#     DIR_N ,DIR_NE,DIR_E,DIR_SE,DIR_S,DIR_SW,DIR_W,DIR_NW = range(8)

#     for ent_idx in range(mask.shape[0]):
#         if not mask[ent_idx]:                 # padding
#             continue
#         if int(state[ent_idx, 13]) != 0:        # 非 Aircraft
#             continue
#         if int(state[ent_idx, 33]) not in PATROL_AIRCRAFT_TYPES:
#             continue

#         # ========== 油量判定：返航 ==========
#         if float(state[ent_idx, 9]) < fuel_rtb_th:
#             actions[1, ent_idx, 0] = 1.0         # ReturnToBase
#             actions[2:, ent_idx, 0] = 0.01       # 其他动作压低
#             continue

#         # ========== 正常巡逻 ==========
#         dir_map = {0: DIR_N, 1: DIR_W, 2: DIR_S} # ← 左侧 U
#         patrol_dir = dir_map[phase]

#         actions[2, ent_idx, 0] = 0.80          # WayPointMoveActor.prob
#         actions[2, ent_idx, 1] = patrol_dir   # direction_index
#         actions[2, ent_idx, 3] = 2            # 速度量化：巡航
#         actions[2, ent_idx, 4] = 2            # 高度量化：低空/低海面
#     return actions


def evade_missiles(actions: np.ndarray,
                   encoded_state: torch.Tensor,  # [1,E,30]
                   mask: torch.Tensor,  # [1,E]
                   raw_data: list,
                   missile_threat_dist=2000, ):
    """
    对最近 missile_threat_dist 米内的导弹威胁，
    把 WayPointMoveActor 的 5-tuple 覆写成规避动作。
    """
    # === threat map ===
    urgent_flags = {}
    threat_map = find_incoming_threats(
        encoded_state[0].cpu().numpy(),
        mask[0].cpu().numpy(),
        raw_data,
        threat_distance_threshold=missile_threat_dist
    )
    if len(threat_map) == 0:
        return actions, urgent_flags  # 无威胁

    for ent_idx in range(mask.shape[1]):  # 遍历我方实体
        if not mask[0, ent_idx]:
            continue

        ent_id = raw_data[ent_idx]['mdlID']
        if ent_id not in threat_map:
            continue
        if not threat_map[ent_id]:  # 空的话跳过
            continue
        nearest = min(threat_map[ent_id], key=lambda t: t['distance'])  # 根据最近的威胁
        # if nearest['distance'] > missile_threat_dist:
        #     continue
        evade_idx = choose_evade_direction(math.degrees(nearest['bearing']))
        # === WayPointMoveActor 覆写 ===
        actions[2, ent_idx, 0] = 0.85  # 执行概率
        actions[2, ent_idx, 1] = evade_idx  # direction_index
        actions[2, ent_idx, 2] = 0.0  # 占位 NONE
        actions[2, ent_idx, 3] = 4  # 速度量化 (0-4)
        actions[2, ent_idx, 4] = 5  # 高度量化 (0-5)
        actions[3, ent_idx, 0] = 0.01
        urgent_flags[raw_data[ent_idx]['mdlID'], 2] = True  # (ent_id, actor_idx)
    return actions, urgent_flags


def geo_distance(lon1, lat1, lon2, lat2, alt_diff=None):
    """
    智能计算地理距离（自动适应2D/3D场景）

    参数：
        lon1, lat1: 点1的经度(度)、纬度(度)
        lon2, lat2: 点2的经度(度)、纬度(度)
        alt_diff: 可选参数，两点的高度差(米)。若为None则计算2D距离

    返回：
        距离（单位：米）
    """
    # 地球半径（单位：米）
    R = 6371000

    # 转成弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # 计算水平距离（Haversine公式）
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c

    # 如果提供了高度差，则计算3D距离
    if alt_diff is not None:
        return math.sqrt(distance ** 2 + alt_diff ** 2)

    return distance


'''def choose_evade_direction(missile_yaw_deg):
    """
    给定导弹朝向（角度，0~360°），选择一个横向90°机动方向。
    返回离散的方向index。
    """

    # 转成弧度
    missile_yaw_rad = math.radians(missile_yaw_deg)

    # 计算导弹飞行的朝向单位向量
    missile_dx = math.cos(missile_yaw_rad)
    missile_dy = math.sin(missile_yaw_rad)

    # 计算与导弹垂直方向的两个可能方向：顺时针90度，逆时针90度
    right_dx =  math.cos(missile_yaw_rad - math.pi/2)
    right_dy =  math.sin(missile_yaw_rad - math.pi/2)

    left_dx  =  math.cos(missile_yaw_rad + math.pi/2)
    left_dy  =  math.sin(missile_yaw_rad + math.pi/2)

    # 你有 DIRECTION_MAP 方向映射，找哪个离散方向最接近 right 或 left
    DIRECTION_VEC = {
        0: (0, +1),
        1: (+1, +1),
        2: (+1, 0),
        3: (+1, -1),
        4: (0, -1),
        5: (-1, -1),
        6: (-1, 0),
        7: (-1, +1),
    }

    best_right = None
    best_left = None
    best_right_cos = -2
    best_left_cos = -2

    for idx, (dx, dy) in DIRECTION_VEC.items():
        norm = math.sqrt(dx**2 + dy**2) + 1e-6
        dx /= norm
        dy /= norm

        # 计算向量夹角余弦，相似度
        cos_right = dx * right_dx + dy * right_dy
        cos_left = dx * left_dx + dy * left_dy

        if cos_right > best_right_cos:
            best_right_cos = cos_right
            best_right = idx

        if cos_left > best_left_cos:
            best_left_cos = cos_left
            best_left = idx

    # 随机选左或右机动
    if random.random() < 0.5:
        return best_right
    else:
        return best_left
'''


def choose_evade_direction(missile_yaw_deg):
    """
    给定导弹朝向（角度，0~360°），选择一个横向90°机动方向。
    返回离散的方向index。
    """
    # 将导弹朝向角度转换为索引
    direction_index = int((missile_yaw_deg + 22.5) % 360 / 45)

    # 计算横向90°方向的索引
    right_index = (direction_index + 2) % 8  # 顺时针90度
    return right_index


OUR_SIDE = 1
ENEMY_SIDE = 0

# from HG 的版本
# def find_incoming_threats(encoded_data: np.ndarray,
#                           mask: np.ndarray,
#                           raw_data: list,
#                           our_side: int = OUR_SIDE,
#                           enemy_side: int = ENEMY_SIDE,
#                           missile_type: int = 4,
#                           threat_distance_threshold: float = 150000) -> dict:
#     """
#     返回 { our_id : [ {missile_id, distance(m), bearing(rad)} , ...] }
#     """

#     # ---- 1. 筛索引 ----
#     our_idxs = get_env_entity_ids(encoded_data, mask,
#                                   target_side=our_side,
#                                   allowed_types=[0, 1, 2])  # 我方可控实体
#     missile_idxs = get_env_entity_ids(encoded_data, mask,
#                                       target_side=enemy_side,
#                                       allowed_types=[missile_type])  # 敌方导弹实体的索引
#     if not missile_idxs:
#         return {}

#     # ---- 2. 坐标 ----
#     our_coords = get_coordinate_from_encoded_data(
#         encoded_data, mask, target_side=our_side, allowed_types=[0, 1, 2])
#     missile_coords = get_coordinate_from_encoded_data(
#         encoded_data, mask, target_side=enemy_side, allowed_types=[missile_type])

#     # ---- 3. 初始化结果 ----
#     threats = {raw_data[i]['mdlID']: [] for i in our_idxs}

#     # ---- 4. 每枚导弹分配给最近目标 ----
#     our_entities = list(zip(our_idxs, our_coords))  # [(idx1, (lat1, lon1)), (idx2, (lat2, lon2)), ...]
#     missile_entities = list(zip(missile_idxs, missile_coords))  # [(idx, (lat, lon))]
#     for our_idx, our_coord in our_entities:
#         our_id = raw_data[our_idx]['mdlID']
#         for m_idx, m_coord in missile_entities:
#             dist = geo_distance(our_coord[0], our_coord[1], m_coord[0], m_coord[1])

#             if dist <= threat_distance_threshold:
#                 # 方位角（bearing）计算
#                 dx = our_coord[0] - m_coord[0]
#                 dy = our_coord[1] - m_coord[1]
#                 bearing = math.atan2(dx, dy) % (2 * math.pi)
                

#                 threats[our_id].append({
#                     'missile_id': raw_data[m_idx]['mdlID'],
#                     'distance': dist,
#                     'bearing': bearing
#                 })
#     return threats


# # try catch的版本
# def find_incoming_threats(encoded_data: np.ndarray,
#                           mask: np.ndarray,
#                           raw_data: list,
#                           our_side: int = OUR_SIDE,
#                           enemy_side: int = ENEMY_SIDE,
#                           missile_type: int = 4,
#                           threat_distance_threshold: float = 150000) -> dict:
#     """
#     返回 { our_id : [ {missile_id, distance(m), bearing(rad)} , ...] }
#     """

#     # ---- 1. 筛索引 ----
#     our_idxs = get_env_entity_ids(encoded_data, mask,
#                                   target_side=our_side,
#                                   allowed_types=[0, 1, 2])  # 我方可控实体
#     missile_idxs = get_env_entity_ids(encoded_data, mask,
#                                       target_side=enemy_side,
#                                       allowed_types=[missile_type])  # 敌方导弹实体的索引
#     if not missile_idxs:
#         return {}

#     # ---- 2. 坐标 ----
#     our_coords = get_coordinate_from_encoded_data(
#         encoded_data, mask, target_side=our_side, allowed_types=[0, 1, 2])
#     missile_coords = get_coordinate_from_encoded_data(
#         encoded_data, mask, target_side=enemy_side, allowed_types=[missile_type])

#     # ---- 3. 初始化结果 ----
#     threats = {raw_data[i]['mdlID']: [] for i in our_idxs}

#     # ---- 4. 每枚导弹分配给最近目标 ----
#     our_entities = list(zip(our_idxs, our_coords))  # [(idx1, (lat1, lon1)), (idx2, (lat2, lon2)), ...]
#     missile_entities = list(zip(missile_idxs, missile_coords))  # [(idx, (lat, lon))]
#     for our_idx, our_coord in our_entities:
#         our_id = raw_data[our_idx]['mdlID']
#         for m_idx, m_coord in missile_entities:
#             dist = geo_distance(our_coord[0], our_coord[1], m_coord[0], m_coord[1])

#             if dist <= threat_distance_threshold:
#                 # 方位角（bearing）计算
#                 dx = our_coord[0] - m_coord[0]
#                 dy = our_coord[1] - m_coord[1]
#                 bearing = math.atan2(dx, dy) % (2 * math.pi)
#                 try:
#                     missile_id = raw_data[m_idx]['mdlID']
#                 except IndexError:
#                     # 建议打日志排查是哪一步丢失了同步
#                     print(f"[WARN] raw_data 长度={len(raw_data)}, m_idx={m_idx} 越界，已跳过该导弹")
#                     continue

#                 threats[our_id].append({
#                     'missile_id': missile_id,
#                     'distance': dist,
#                     'bearing': bearing
#                 })
#     return threats



# 使用映射版本
def find_incoming_threats(encoded_data: np.ndarray,
                          mask: np.ndarray,
                          raw_data: list,
                          our_side: int = OUR_SIDE,
                          enemy_side: int = ENEMY_SIDE,
                          missile_type: int = 4,
                          threat_distance_threshold: float = 150000) -> dict:
    """
    返回 { our_id : [ {missile_id, distance(m), bearing(rad)} , ...] }
    """
    # ---- 0. 预先构建行号到 raw_data 条目的映射 ----
    raw_lookup = raw_data
    # raw_lookup = {item['row_idx']: item for item in raw_data}

    # ---- 1. 筛索引 ----
    our_idxs = get_env_entity_ids(encoded_data, mask,
                                  target_side=our_side,
                                  allowed_types=[0, 1, 2],
                                  require_manageable=True)  # 我方可控实体
    missile_idxs = get_env_entity_ids(encoded_data, mask,
                                      target_side=enemy_side,
                                      allowed_types=[missile_type],
                                      require_manageable=False)  # 敌方导弹实体的索引
    if not missile_idxs:
        return {}

    # ---- 2. 坐标 ----
    our_coords = get_coordinate_from_encoded_data(
        encoded_data, mask, target_side=our_side, allowed_types=[0, 1, 2],
        require_manageable=True)
    missile_coords = get_coordinate_from_encoded_data(
        encoded_data, mask, target_side=enemy_side, allowed_types=[missile_type],
        require_manageable=False)

    # ---- 3. 初始化结果 ----
    threats = {raw_data[i]['mdlID']: [] for i in our_idxs}

    # ---- 4. 每枚导弹分配给最近目标 ----
    our_entities = list(zip(our_idxs, our_coords))  # [(idx1, (lat1, lon1)), (idx2, (lat2, lon2)), ...]
    missile_entities = list(zip(missile_idxs, missile_coords))  # [(idx, (lat, lon))]
    for our_idx, our_coord in our_entities:
        our_id = raw_lookup[our_idx]['mdlID'] 
        for m_idx, m_coord in missile_entities:
            dist = geo_distance(our_coord[0], our_coord[1], m_coord[0], m_coord[1])

            if dist <= threat_distance_threshold:
                # 方位角（bearing）计算
                dx = our_coord[0] - m_coord[0]
                dy = our_coord[1] - m_coord[1]
                bearing = math.atan2(dx, dy) % (2 * math.pi)
                missile_raw = raw_lookup[m_idx]
                if missile_points_at_entity(
                    missile_raw=missile_raw,
                    own_id=our_id,
                    missile_longitude=m_coord[0],
                    missile_latitude=m_coord[1],
                    own_longitude=our_coord[0],
                    own_latitude=our_coord[1],
                ):
                    attitude = missile_raw.get('attitude', {})
                    missile_heading = attitude.get('yaw') if isinstance(attitude, dict) else None
                    if missile_heading is not None:
                        bearing = math.radians(float(missile_heading)) % (2 * math.pi)
                    threats[our_id].append({
                        'missile_id': missile_raw['mdlID'],
                        'distance': dist,
                        'bearing': bearing
                    })
    return threats


def log_entity_actions(execute_results, actor_types, logger):
    for entity_id, action_is_performs in execute_results.items():
        # 获取执行的动作ID列表
        performed_action_ids = [i for i, is_performed in enumerate(action_is_performs) if is_performed]

        # 转换为动作名称列表（按ACTOR_TYPES顺序）
        performed_actions = [actor_types[i].__name__ + '1' for i in performed_action_ids]

        # 记录日志（如果执行了至少一个动作）
        if performed_actions:
            logger.info(f"实体 {entity_id} 执行了以下动作: {', '.join(performed_actions)}")


def is_point_in_area(point, area_polygon):
    """
    使用 shapely 判断点是否在区域内。
    """
    polygon = Polygon(area_polygon)
    return polygon.contains(Point(point))


# def get_area_center(area_polygon):
#     """
#     获取区域（多边形）的几何中心点（centroid）。

#     参数：
#         area_polygon: List[Tuple[float, float]]，区域顶点坐标

#     返回：
#         Tuple[float, float]：几何中心点坐标 (x, y)
#     """
#     polygon = Polygon(area_polygon)
#     center = polygon.centroid
#     return (center.x, center.y)

# def get_area_center(area_polygon):
#     """
#     获取区域（多边形）的几何中心点（centroid）。

#     参数：
#         area_polygon: List[Tuple[float, float]]，区域顶点坐标

#     返回：
#         Tuple[float, float]：几何中心点坐标 (x, y)
#     """
#     if isinstance(area_polygon, Polygon):
#         polygon = area_polygon
#     else:
#         coords = [(pt.x, pt.y) if isinstance(pt, Point) else pt for pt in area_polygon]
#         polygon = Polygon(coords)
#     center = polygon.centroid
#     return (center.x, center.y)


def get_entity_offset(entity_id, max_offset=0.7):
    """
    根据entity_id生成一个确定性的经纬度偏移量。

    参数：
        entity_id: 实体的唯一标识符
        max_offset: 最大偏移量（单位：度，约100米）

    返回：
        Tuple[float, float]: (经度偏移量, 纬度偏移量)
    """
    # 使用hash算法确保相同的entity_id总是得到相同的偏移量
    hash_obj = hashlib.md5(str(entity_id).encode())
    hash_int = int(hash_obj.hexdigest(), 16)

    # 生成-1到1之间的随机数
    lon_ratio = (hash_int % 2000) / 1000 - 1
    lat_ratio = ((hash_int // 2000) % 2000) / 1000 - 1

    # 应用最大偏移量
    return (lon_ratio * max_offset, lat_ratio * max_offset)


def get_area_target_point(entity_id, area_polygon, max_offset=0.7):
    """
    获取实体在巡逻区域的目标点（中心点+确定性偏移）。

    参数：
        entity_id: 实体的唯一标识符
        area_polygon: List[Tuple[float, float]]，区域顶点坐标
        max_offset: 最大偏移量（单位：度，约100米）

    返回：
        Tuple[float, float]: 目标点坐标 (经度, 纬度)
    """
    # 获取中心点
    if isinstance(area_polygon, Polygon):
        polygon = area_polygon
    else:
        coords = [(pt.x, pt.y) if isinstance(pt, Point) else pt for pt in area_polygon]
        polygon = Polygon(coords)
    center = polygon.centroid

    # 获取确定性偏移
    lon_offset, lat_offset = get_entity_offset(entity_id, max_offset)

    # 应用偏移
    target_lon = center.x + lon_offset
    target_lat = center.y + lat_offset

    # 确保目标点在多边形内（如果偏移后超出范围，则使用最近的点）
    target_point = Point(target_lon, target_lat)
    if not polygon.contains(target_point):
        target_point = polygon.exterior.interpolate(polygon.exterior.project(target_point))

    return (target_point.x, target_point.y)


def compute_direction_index(from_coord, to_coord):
    """
    计算从 from_coord 到 to_coord 的大致方向索引。
    方向索引如下（共 8 个方向，以正北为0开始，顺时针旋转）：
        0: 北（0° ±22.5°）
        1: 东北（45° ±22.5°）
        2: 东（90° ±22.5°）
        3: 东南（135° ±22.5°）
        4: 南（180° ±22.5°）
        5: 西南（225° ±22.5°）
        6: 西（270° ±22.5°）
        7: 西北（315° ±22.5°）
    参数：
        from_coord: 当前坐标 (lon, lat)
        to_coord: 目标坐标 (lon, lat)
    返回：
        方向索引（0~7），0 表示正北方向
    """
    dx = to_coord[0] - from_coord[0]  # 经度方向 = x
    dy = to_coord[1] - from_coord[1]  # 纬度方向 = y

    # 计算角度（以正北为0度，顺时针）
    angle_rad = math.atan2(dx, dy)  # 注意：dx, dy 顺序反过来，才能以正北为起点
    angle_deg = (math.degrees(angle_rad) + 360) % 360  # 转换为角度并归一化到 0~360

    # 将角度映射到 8 个方向索引（每个方向占 45°，中心±22.5°）
    direction_index = int((angle_deg + 22.5) // 45) % 8
    return direction_index


def get_coord(encoded_data: np.ndarray,
              mask: np.ndarray,
              raw_data: list,
              ent_id: str,
              our_side: int = OUR_SIDE):
    """
    根据 mdlID(ent_id) 找到它在 encoded_data / raw_data 中对应的经纬度坐标：
      1. 只在“我方可控实体”([0,1,2]类型)里搜索
      2. 如果找到了就返回 (lon, lat)，否则返回 None
    """
    return_coord = None
    # 1) 拿到我方可控实体的索引和坐标列表
    our_idxs = get_env_entity_ids(
        encoded_data, mask,
        target_side=our_side,
        allowed_types=[0, 1, 2],
        require_manageable=True
    )
    our_coords = get_coordinate_from_encoded_data(
        encoded_data, mask,
        target_side=our_side,
        allowed_types=[0, 1, 2],
        require_manageable=True
    )

    # 2) 在 raw_data 里匹配 mdlID，然后返回对应的坐标
    for idx, coord in zip(our_idxs, our_coords):
        if raw_data[idx]['mdlID'] == ent_id:
            return_coord = coord
            return return_coord

    # 3) 都没找到
    if return_coord is None:
        for dt in raw_data:
            if dt['mdlID'] == ent_id:
                return_coord = (dt['entitySpatialCoord']['longitude'], dt['entitySpatialCoord']['latitude'])
                break

    return return_coord


# "Random-patrol": generate_random_patrol(area,current_pos),  随机 漏洞形
# "Bow-shaped-patrol": generate_bow_patrol(area, current_pos,num_area=5), 贪吃蛇形
# "Z-patrol": generate_Z_patrol(area, current_pos,num_slash=3), Z形
# "Spiral": generate_spiral_patrol(area, current_pos,num_loops=4) 螺旋缩小的
def generate_random_patrol(polygon: Polygon,
                           current_pos: tuple,
                           margin_ratio: float = 0.05):
    """
    · 6 个固定巡逻点：左下 → 顶中 → 右下 → 右上 → 底中 → 左上
    · 从离当前坐标最近的点出发
    · **优先往左（x 减小方向）** 巡逻；除非已在最左边，再按顺时针。
    """
    # ---------- 1. 留白后的 6 个基准点 ----------
    minx, miny, maxx, maxy = polygon.bounds
    w, h = maxx - minx, maxy - miny
    dx, dy = w * margin_ratio, h * margin_ratio
    xmin, xmax = minx + dx, maxx - dx
    ymin, ymax = miny + dy, maxy - dy
    xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2

    # ---------- 2. 根据 current_pos 和随机数，收缩到某一半区 ----------
    r = random.random()
    if current_pos[1] < ymid:
        # 当前在下半区
        if r < 0.6:
            # 80% 保持下半
            ymax = ymid
        else:
            # 20% 切换到上半
            ymin = ymid
    else:
        # 当前在上半区
        if r < 0.6:
            # 80% 保持上半
            ymin = ymid
        else:
            # 20% 切换到下半
            ymax = ymid

    # 重新计算区域中心 X
    xmid = (xmin + xmax) / 2

    pts = [
        (xmin, ymin),  # 0 左下
        (xmid, ymax),  # 1 顶中
        (xmax, ymin),  # 2 右下
        (xmax, ymax),  # 3 右上
        (xmid, ymin),  # 4 底中
        (xmin, ymax)  # 5 左上
    ]

    # ---------- 2. 找离 current_pos 最近的基准点 ----------
    def dist(p):
        return math.hypot(p[0] - current_pos[0],
                          p[1] - current_pos[1])

    start_idx = min(range(len(pts)), key=lambda i: dist(pts[i]))

    # ---------- 3. 决定巡逻方向：先向左 ----------
    n = len(pts)
    cur_x = pts[start_idx][0]

    # 若当前不是最左侧列，则选“下一步 x 更小”的方向
    next_idx_fwd = (start_idx + 1) % n
    prev_idx = (start_idx - 1) % n
    dir_step = -1  # 默认逆着列表走
    if cur_x > xmin:  # 还有更左空间
        # 比较哪个方向第一步更向左
        if pts[prev_idx][0] >= cur_x:  # 往 prev 仍没变小
            dir_step = +1  # 只能向右（顺列表）
    else:
        dir_step = +1  # 已在最左，按顺列表

    # ---------- 4. 构造巡逻路径 ----------
    path = []
    idx = start_idx
    for _ in range(n):
        path.append(pts[idx])
        idx = (idx + dir_step) % n
    path = [(round(x, 2), round(y, 2)) for x, y in path]
    return path


def generate_bow_patrol(polygon: Polygon,
                        current_pos: tuple,
                        num_area: int = 5,
                        margin_ratio: float = 0.05,
                        edge_threshold: float = 0.10):
    """
    • 起点所在“最近网格线 x” 开始；
    • 若起点在左侧 edge_threshold(默认 10 %) 内 → 向右扫；否则向左扫；
    • 起点 y 靠近底边 → 先下后上；否则先上后下；
    • 垂线之间保持之字形。
    """

    # ---------- 有效边界 ----------
    min_x, min_y, max_x, max_y = polygon.bounds

    w, h = max_x - min_x, max_y - min_y
    dx, dy = w * margin_ratio, h * margin_ratio
    xmin, xmax = min_x + dx, max_x - dx
    ymin, ymax = min_y + dy, max_y - dy

    line_spacing = (xmax - xmin) / (num_area)

    # ---------- 半区收缩逻辑（新增） ----------
    ymid = (ymin + ymax) / 2
    r = random.random()
    cx, cy = current_pos
    if cy < ymid:
        # 当前在“下半区”
        if r < 0.6:
            ymax = ymid  # 80% 保持下半
        else:
            ymin = ymid  # 20% 切到上半
    else:
        # 当前在“上半区”
        if r < 0.6:
            ymin = ymid  # 80% 保持上半
        else:
            ymax = ymid  # 20% 切到下半

    # ---------- 起点信息 ----------
    cx, cy = current_pos
    cx = min(max(cx, xmin), xmax)  # 若起点落在留白外，钳到内部

    # ---------- 扫描方向：左 10 % → 向右，否则向左 ----------
    left_band = min_x + w * edge_threshold
    scan_dir = +1 if cx <= left_band else -1

    # ---------- 首条垂直线网格对齐 ----------
    #  将 cx 对齐到最近的 line_spacing 网格线
    grid_idx = round((cx - xmin) / line_spacing)
    first_x = xmin + grid_idx * line_spacing
    first_x = min(max(first_x, xmin), xmax)  # 再次钳制

    # ---------- y 方向：起点靠下 ? 先下后上 : 先上后下 ----------
    downwards = False if cy < (ymin + ymax) / 2 else True  # False=下→上，True=上→下

    # ---------- 生成所有垂直线 ----------
    lines, cur_x = [], first_x
    cond = (lambda x: x <= xmax + 1e-9) if scan_dir > 0 else (lambda x: x >= xmin - 1e-9)

    while cond(cur_x):
        if downwards:  # 先上后下
            p0, p1 = (cur_x, ymax), (cur_x, ymin)
        else:  # 先下后上
            p0, p1 = (cur_x, ymin), (cur_x, ymax)

        lines.append((p0, p1))
        cur_x += scan_dir * line_spacing  # 向左或向右推进
        downwards = not downwards  # 之字翻转

    # ---------- 组装路径 ----------
    path = []

    def add(pt):  # 加点时去重
        if not path or pt != path[-1]:
            path.append(pt)

    for i, (p0, p1) in enumerate(lines):
        add(p0)
        add(p1)

        if i < len(lines) - 1:
            next_x = lines[i + 1][0][0]
            turn = (next_x, p1[1])
            if Point(turn).within(polygon):
                add(turn)
    path = [(round(x, 2), round(y, 2)) for x, y in path]
    return path


def generate_Z_patrol(polygon: Polygon,
                      current_pos: tuple,
                      num_slash: int = 3,
                      margin_ratio: float = 0.05,
                      edge_threshold: float = 0.10):
    path, visited = [], set()

    def add(pt):
        if pt not in visited:  # 全局判重
            path.append(pt)
            visited.add(pt)

    # ---------- 1. 有效扫描矩形 ----------
    minx, miny, maxx, maxy = polygon.bounds
    w, h = maxx - minx, maxy - miny
    dx, dy = w * margin_ratio, h * margin_ratio
    xmin, xmax = minx + dx, maxx - dx
    ymin, ymax = miny + dy, maxy - dy
    cx, cy = current_pos

    # --- 新增：半区收缩逻辑（80% 本半区，20% 对面半区） ---
    ymid = (ymin + ymax) / 2
    if cy < ymid:
        # 当前在“下半区”
        if random.random() < 0.8:
            ymax = ymid
        else:
            ymin = ymid
    else:
        # 当前在“上半区”
        if random.random() < 0.8:
            ymin = ymid
        else:
            ymax = ymid

    # ---------- 2. 判断扫描方向 ----------
    vertical_down_first = cy > (ymin + ymax) / 2
    left_space, right_space = cx - xmin, xmax - cx
    need_space = w * edge_threshold

    if left_space >= need_space:  # 向左扫
        x_start, x_end = cx, xmin
        slash_down = not vertical_down_first
    else:  # 向右扫
        x_start, x_end = cx, xmax
        slash_down = not vertical_down_first

    # ---------- 3. 划分列 ----------
    tot_cols = num_slash * 2 + 1
    x_cols = np.linspace(x_start, x_end, tot_cols)

    # ---------- 4. 生成路径 ----------
    down_flag = vertical_down_first
    for idx, x in enumerate(x_cols):
        if idx % 2 == 0:  # 竖线
            start = (x, ymax) if down_flag else (x, ymin)
            end = (x, ymin) if down_flag else (x, ymax)
            add(start);
            add(end)
            down_flag = not down_flag
        else:  # 斜线
            xl, xr = x_cols[idx - 1], x_cols[idx + 1]
            first = (xl, ymax) if slash_down else (xl, ymin)
            second = (xr, ymin) if slash_down else (xr, ymax)
            if not path or path[-1] != first:  # 避免与上一个节点重复
                add(first)
            add(second)
    path = [(round(x, 2), round(y, 2)) for x, y in path]
    return path


def build_rectangular_spiral(xmin, xmax, ymin, ymax,
                             step,
                             start_corner="BR",  # "BR" "TR" "BL" "TL"
                             direction="ccw"):  # "ccw" 或 "cw"
    """
    纯水平 / 垂直矩形螺旋。
    start_corner:
        BR = 右下, TR = 右上, BL = 左下, TL = 左上
    direction 只决定顺时针 / 逆时针，不会改变“首段沿边”：
        ● BR 起点:  ccw=↑←↓→…   cw=←↑→↓…
        ● TR 起点:  ccw=←↓→↑…   cw=↓←↑→…
        ● BL 起点:  ccw=→↓←↑…   cw=→↑←↓…
        ● TL 起点:  ccw=↓→↑←…   cw=→↓←↑…
    """
    # 为了少写 4 份代码：把左起点镜像到右起点来复用
    mirror = start_corner in ("BL", "TL")
    if mirror:
        xmin, xmax = -xmax, -xmin  # X 轴镜像（左右互换）

    # 选右侧对应起点、方向
    if start_corner in ("BR", "BL"):
        path = [(xmax, ymin)]
        first_turn = ("up", "left") if direction == "ccw" else ("left", "up")
    else:  # "TR" / "TL"
        path = [(xmax, ymax)]
        first_turn = ("down", "left") if direction == "cw" else ("left", "down")

    L, R = xmin, xmax
    B, T = ymin, ymax
    while L < R and B < T:
        if first_turn == ("up", "left"):  # ↑ ← ↓ →
            path.extend([(R, T), (L, T), (L, B), (R, B)])
        elif first_turn == ("left", "up"):  # ← ↑ → ↓
            path.extend([(L, B), (L, T), (R, T), (R, B)])
        elif first_turn == ("down", "left"):  # ↓ ← ↑ →
            path.extend([(R, B), (L, B), (L, T), (R, T)])
        else:  # ← ↓ → ↑
            path.extend([(L, T), (L, B), (R, B), (R, T)])

        # 收缩边界
        R -= step;
        T -= step;
        L += step;
        B += step

    # 追加中心点
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    if path[-1] != (cx, cy):
        path.append((cx, cy))

    # 若做过镜像，记得再镜回来
    if mirror:
        path = [(-x, y) for x, y in path]

    return path


def generate_spiral_patrol(polygon: Polygon,
                           current_pos: tuple = None,
                           num_loops: int = 3,
                           margin_ratio: float = 0.05,
                           edge_threshold: float = 0.10,
                           rebase_to_current: bool = False):
    """


    左侧 10 % 区域 ⇒ 向右搜索（起点在左上 / 左下，首段水平向右）
    其余位置     ⇒ 原来的“下→右下起步逆时针，上→右上起步顺时针”
    """
    # ---------- 1) 有效边界 ----------
    minx, miny, maxx, maxy = polygon.bounds
    W, H = maxx - minx, maxy - miny
    dx, dy = W * margin_ratio, H * margin_ratio
    xmin0, xmax0 = minx + dx, maxx - dx
    ymin0, ymax0 = miny + dy, maxy - dy
    mid_y = (ymin0 + ymax0) / 2

    # ---------- 2) “左 10 %” 判定 ----------
    in_left_band = False
    if current_pos:
        left_band = xmin0 + W * edge_threshold
        in_left_band = current_pos[0] <= left_band

    # ---------- 3) 确定螺旋所在子矩形 ----------
    sub_w = (xmax0 - xmin0) / 3
    if in_left_band:
        xmin, xmax = xmin0, xmin0 + sub_w  # 固定用最左 1/3
    elif current_pos is None:
        xmin, xmax = xmax0 - sub_w, xmax0  # 默认右 1/3
    else:
        cx = np.clip(current_pos[0], xmin0, xmax0)
        if cx - xmin0 >= sub_w:
            xmin, xmax = cx - sub_w, cx  # 左 1/3
        elif xmax0 - cx >= sub_w:
            xmin, xmax = cx, cx + sub_w  # 右 1/3
        else:
            xmin, xmax = xmin0, xmax0  # 全宽
    ymin, ymax = ymin0, ymax0

    # ---------- 半区收缩逻辑（新增） ----------
    ymid = (ymin + ymax) / 2
    r = random.random()
    cx, cy = current_pos
    if cy < ymid:
        # 当前在“下半区”
        if r < 0.6:
            ymax = ymid  # 80% 保持下半
        else:
            ymin = ymid  # 20% 切到上半
    else:
        # 当前在“上半区”
        if r < 0.6:
            ymin = ymid  # 80% 保持上半
        else:
            ymax = ymid  # 20% 切到下半

    # ---------- 4) 计算步长 ----------
    loops = max(1, num_loops)
    while True:
        step = min((xmax - xmin) / (2 * loops),
                   (ymax - ymin) / (2 * loops))
        if step >= 0.5 or loops == 2:  # step 后面的数越大， 圈与圈之间间距越大
            break
        loops = max(1, loops // 2)

    # ---------- 5) 起点 / 方向 ----------
    if in_left_band:
        # 如果在左侧区域，区分“左上”还是“左下”
        if current_pos and current_pos[1] > mid_y:
            # 左上 → 优先逆时针
            start_corner, direction = "TL", "ccw"
        else:
            # 左下 或 current_pos=None → 优先顺时针
            start_corner, direction = "BL", "ccw"
    else:
        # 不在左侧时保持原来逻辑：
        # 在上半区 → “TR” + 顺时针；在下半区 → “BR” + 逆时针
        if current_pos and current_pos[1] > mid_y:
            start_corner, direction = "TR", "cw"
        else:
            start_corner, direction = "BR", "ccw"

    # ---------- 6) 生成路径 ----------
    path = build_rectangular_spiral(xmin, xmax, ymin, ymax,
                                    step,
                                    start_corner=start_corner,
                                    direction=direction)

    # ---------- 7) 需要“就近起步”可循环旋转 ----------
    if rebase_to_current and current_pos:
        i0 = min(range(len(path)),
                 key=lambda i: math.hypot(path[i][0] - current_pos[0],
                                          path[i][1] - current_pos[1]))
        path = path[i0:] + path[:i0]
    path = [(round(x, 2), round(y, 2)) for x, y in path]
    return path

def get_aircraft_on_air_num(our_entity_indices, now_state, aircraft_unitcategory_num):
    aircraft_unitcategory_num.clear()
    for entity_idx in our_entity_indices:
        if now_state['encoded_data'][entity_idx][13] == 0:
            unitCategoty = int(now_state['encoded_data'][entity_idx][33].item())
            a_st = int(now_state['encoded_data'][entity_idx][COL_AIR_STATUS].item())
            height = now_state['encoded_data'][entity_idx][5]
            if a_st not in [17] and height > 0:
                aircraft_unitcategory_num[unitCategoty] += 1
