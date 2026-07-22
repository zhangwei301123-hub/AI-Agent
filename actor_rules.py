from engagement_rules import *
from utils import *
import math
from execute  import *
from maddpg_rule_guard import (
    AttackTargetInput,
    DEFAULT_ATTACK_QUANTITY,
    decide_attack,
    sonobuoy_deployment_allowed,
)

def _rad_diff(a, b):
    """最小角差 ∈ [0, π]
    例如从 10° 转到 350°，有两种路径：顺时针 340°，逆时针 20°，最小角差是 20°
    """
    d = (a - b + math.pi) % (2*math.pi) - math.pi
    return abs(d)

def _bearing(lon1, lat1, lon2, lat2):
    """返回两个经纬度之间的正北方向方位角（弧度）"""
    # 转为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    angle = math.atan2(x, y)
    return angle % (2 * math.pi)  # 保证结果在 [0, 2π)

def get_waypoint_index(diff_lon, diff_lat): # 左右上下
        angle_rad = math.atan2(diff_lon, diff_lat)
        if angle_rad < 0:
            angle_rad += 2 * math.pi

        return round(angle_rad / (math.pi / 4)) % 8

def handle_return_to_base_rule(actions, now_state, i, entity_idx, entity_id, actions_dict):
    """
    处理返航硬规则
    """
    own_type = int(now_state['encoded_data'][entity_idx, 13])
    oil_status = float(now_state['encoded_data'][entity_idx, 9])  # 燃油状态
    need_rtb = (oil_status == 1)

    if need_rtb and own_type == 0:
        # if now_state['encoded_data'][entity_idx, 23] == '12':  # 直到状态是12，也就是parked 才结束返航
        #     list_actions_for_entity = actions[:, i, :].tolist()
        #     actions_dict[entity_id] = list_actions_for_entity
        #     return False

        actions[:, i, :][1][0] = 0.81
        for a_idx in (0, 2, 3, 4, 5, 6, 7):
            actions[:, i, :][a_idx][0] = 0.01
        list_actions_for_entity = actions[:, i, :].tolist()
        actions_dict[entity_id] = list_actions_for_entity
        return True  # 表示已处理返航，后续逻辑不再执行

    elif not need_rtb and own_type == 0:
        actions[:, i, :][1][0] = 0.01
        list_actions_for_entity = actions[:, i, :].tolist()
        actions_dict[entity_id] = list_actions_for_entity
    return False


def calculate_scale_rate(direction_index, attack_point_list):
    if not attack_point_list:
        return 1.0

    offsets = DIRECTION_OFFSET[direction_index]

    # 处理单点和复合路径的偏移
    if isinstance(offsets[0], list):
        dlon_list, dlat_list = offsets
    else:
        dlon_list = [offsets[0]]
        dlat_list = [offsets[1]]

    # 计算路径的原始跨度
    if not dlon_list or not dlat_list:
        return 1.0

    max_dlon, min_dlon = max(dlon_list), min(dlon_list)
    max_dlat, min_dlat = max(dlat_list), min(dlat_list)
    span_dlon = max_dlon - min_dlon
    span_dlat = max_dlat - min_dlat

    # 计算交战区的经纬度跨度
    attack_lons = [p[0] for p in attack_point_list]
    attack_lats = [p[1] for p in attack_point_list]
    attack_span_lon = max(attack_lons) - min(attack_lons)
    attack_span_lat = max(attack_lats) - min(attack_lats)

    # 计算所需的缩放比例
    scale_lon = attack_span_lon / span_dlon if span_dlon != 0 else 0
    scale_lat = attack_span_lat / span_dlat if span_dlat != 0 else 0

    # 确定最终缩放比例
    scale_rate = max(scale_lon, scale_lat) if max(scale_lon, scale_lat) > 0 else 1.0
    scale_rate = max(0.5, min(scale_rate, 5.0))  # 限制缩放范围

    return scale_rate

'''
def handle_mobile(
    list_actions_for_entity: List, 
    actions: Any, 
    i: int, 
    entity_idx: int, 
    our_entity_coordinate: List[Tuple[float, float]], 
    raw_data: List[Dict], 
    now_state: Dict, 
    mission_dicts: Dict, 
    urgent_flags: Dict, 
    entity_id: str,
    patrol_count: int
) -> int:
    """处理移动实体的行动逻辑
    
    Args:
        list_actions_for_entity: 实体动作列表
        actions: 全局动作数组
        i: 实体索引
        entity_idx: 原始数据中的实体索引
        our_entity_coordinate: 实体当前坐标列表
        raw_data: 原始实体数据
        now_state: 当前状态数据
        mission_dicts: 任务字典
        urgent_flags: 紧急标志字典
        entity_id: 实体ID
        patrol_count: 当前巡逻计数
        
    Returns:
        更新后的巡逻计数
    """
    waypoint_action = list_actions_for_entity[2]  # [prob, direction_index, NONE, altitude, velocity]
    direction_index = int(waypoint_action[1])
    
    if direction_index not in DIRECTION_OFFSET:
        raise ValueError(f"Invalid direction index: {direction_index}")
    
    cur_lon, cur_lat = our_entity_coordinate[i]
    
    # 获取任务区域
    mission_id = raw_data[entity_idx]['missionId']
    if mission_id not in mission_dicts:
        raise ValueError(f"Invalid mission id: {mission_id}")
    attack_area = build_convex_hull(mission_dicts[mission_id]['area_points'])
    
    # 检查是否在巡逻状态
    is_patrolling = (
        now_state['encoded_data'][entity_idx][13] == 0 and 
        now_state['encoded_data'][entity_idx][33] in [8, 9, 13, 14, 17, 23, 24, 25] and
        not urgent_flags.get((entity_id, 2), False)
    )
    
    # 检查是否在区域内
    in_area = is_point_in_area((cur_lon, cur_lat), attack_area)
    
    # 油量检查
    has_sufficient_oil = raw_data[entity_idx].get('logisticStates', {}).get('oil', -1) > 0.6
    
    # 处理巡逻逻辑
    if is_patrolling:
        patrol_count = handle_patrolling_movement(
            list_actions_for_entity, 
            actions, 
            i, 
            direction_index, 
            cur_lon, 
            cur_lat, 
            attack_area,
            in_area,
            patrol_count
        )
    else:
        patrol_count = handle_normal_movement(
            list_actions_for_entity, 
            actions, 
            i, 
            direction_index, 
            cur_lon, 
            cur_lat, 
            attack_area,
            in_area,
            has_sufficient_oil,
            entity_idx,
            raw_data,
            patrol_count,
            entity_id
        )
    
    return patrol_count

def handle_patrolling_movement(
    list_actions_for_entity: List,
    actions: Any,
    i: int,
    direction_index: int,
    cur_lon: float,
    cur_lat: float,
    attack_area: Any,
    in_area: bool,
    patrol_count: int
) -> int:
    """处理巡逻状态下的移动逻辑"""
    if in_area:
        if direction_index in range(8):  # 0-7的方向用巡逻模式替代
            direction_index = random.randint(8, 11)
            actions[:, i, :][2][1] = direction_index
        
        path = generate_patrol_path(direction_index, attack_area, (cur_lon, cur_lat))
        update_movement_actions(list_actions_for_entity, path)
        return patrol_count + 1
    
    return patrol_count

def handle_normal_movement(
    list_actions_for_entity: List,
    actions: Any,
    i: int,
    direction_index: int,
    cur_lon: float,
    cur_lat: float,
    attack_area: Any,
    in_area: bool,
    has_sufficient_oil: bool,
    entity_idx: int,
    raw_data: Dict,
    patrol_count: int,
    entity_id: str
) -> int:
    """处理正常状态下的移动逻辑"""
    if in_area:
        dlon, dlat = DIRECTION_OFFSET[direction_index]
        
        if len(dlon) == 1:  # 单值移动
            new_lon = cur_lon + dlon
            new_lat = cur_lat + dlat
            list_actions_for_entity[2][1] = new_lon
            list_actions_for_entity[2][2] = new_lat
        else:  # 路径移动
            path = generate_patrol_path(direction_index, attack_area, (cur_lon, cur_lat))
            update_movement_actions(list_actions_for_entity, path)
    elif has_sufficient_oil:  # 不在区域内但有足够油量
        move_to_area_center(
            list_actions_for_entity, 
            actions, 
            i, 
            cur_lon, 
            cur_lat, 
            attack_area,
            entity_id
        )
        return patrol_count + 1
    
    return patrol_count

def generate_patrol_path(
    direction_index: int, 
    attack_area: Any, 
    current_pos: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """根据方向索引生成巡逻路径"""
    cur_lon, cur_lat = current_pos
    
    if direction_index == 8:
        return generate_random_patrol(attack_area, (cur_lon, cur_lat))
    elif direction_index == 9:
        return generate_bow_patrol(attack_area, (cur_lon, cur_lat), num_area=5)
    elif direction_index == 10:
        return generate_Z_patrol(attack_area, (cur_lon, cur_lat), num_slash=3)
    else:  # 默认螺旋巡逻
        return generate_spiral_patrol(attack_area, (cur_lon, cur_lat), num_loops=4)

def update_movement_actions(
    list_actions_for_entity: List,
    path: List[Tuple[float, float]]
) -> None:
    """更新移动动作数据"""
    path_lon, path_lat = zip(*path)
    list_actions_for_entity[2][1] = path_lon  # 经度序列
    list_actions_for_entity[2][2] = path_lat  # 纬度序列
    
    # 保持高度和速度与路径点数量一致
    altitude = list_actions_for_entity[2][3] 
    velocity = list_actions_for_entity[2][4] 
    list_actions_for_entity[2][3] = [altitude] * len(path_lon)
    list_actions_for_entity[2][4] = [velocity] * len(path_lon)

def move_to_area_center(
    list_actions_for_entity: List,
    actions: Any,
    i: int,
    cur_lon: float,
    cur_lat: float,
    attack_area: Any,
    entity_id
) -> None:
    """移动至区域中心"""
    new_lon, new_lat = get_area_target_point(entity_id, attack_point_list)
    list_actions_for_entity[2][1] = new_lon
    list_actions_for_entity[2][2] = new_lat 
    new_direction_index = compute_direction_index((cur_lon, cur_lat), (new_lon, new_lat))
    actions[:, i, :][2][1] = new_direction_index


'''

def handle_mobile(list_actions_for_entity, actions, i, entity_idx, our_entity_coordinate, raw_data, now_state, mission_dicts,
                  urgent_flags, entity_id, patrol_count):
    waypoint_action = list_actions_for_entity[2]  # [prob, direction_index, NONE, altitude, velocity]
    direction_index = int(waypoint_action[1])  # 注意：方向索引存储在 [2][1]，即 waypoint_action[1]
    attack_point_list = []

    if direction_index not in DIRECTION_OFFSET:
        raise ValueError(f"Invalid direction index: {direction_index}")
    
    cur_lon, cur_lat = our_entity_coordinate[i]

    mission_id = raw_data[entity_idx]['missionId']
    normal = False
    if mission_id not in mission_dicts.keys():
        normal = True
        # print(f"Invalid mission id: {mission_id}")
    else:
        attack_point_list = build_convex_hull(mission_dicts.get(mission_id)['area_points'])

    # 对于没有分配任务的实体
    if normal:
        if direction_index in [8, 9, 10, 11]:
            rd = random.random()
            if rd < 0.30:
                direction_index = 6
            elif rd < 0.30 + 0.25:
                direction_index = 4
            elif rd < 0.30 + 0.25 + 0.25:
                direction_index = 0
            else:
                # 剩下的方向 [1, 2, 3, 5, 7]
                direction_index = random.choice([1, 2, 3, 5, 7])
            actions[:, i, :][2][1] = direction_index
        dlon, dlat = DIRECTION_OFFSET[direction_index]

        new_lon = cur_lon + dlon
        new_lat = cur_lat + dlat
        # 替换动作中的经纬度位置
        list_actions_for_entity[2][1] = new_lon  # 经度
        list_actions_for_entity[2][2] = new_lat  # 纬度
        return

    # 对于要巡逻的实体
    if now_state['encoded_data'][entity_idx][13] == 0 and now_state['encoded_data'][entity_idx][33] in [8, 9, 13, 14, 17, 23, 24, 25 ]\
                    and not urgent_flags.get((entity_id, 2), False):

        if is_point_in_area((cur_lon, cur_lat), attack_point_list):
            if direction_index in [0, 1, 2, 3, 4, 5, 6, 7]: # 不要0-7的方向 用巡逻的
                direction_index = random.randint(8, 11)
                actions[:, i, :][2][1] = direction_index
            dlon, dlat = DIRECTION_OFFSET[direction_index]
            
            if direction_index==8:
                path = generate_random_patrol(attack_point_list, (cur_lon,cur_lat))
                path_lon, path_lat = zip(*path)
            if direction_index==9:
                path = generate_bow_patrol(attack_point_list, (cur_lon,cur_lat), num_area=5)
                path_lon, path_lat = zip(*path)
            if direction_index==10:
                path  = generate_Z_patrol(attack_point_list, (cur_lon,cur_lat), num_slash=3)
                path_lon, path_lat = zip(*path)
            else:
                path = generate_spiral_patrol(attack_point_list, (cur_lon,cur_lat), num_loops=4)
                path_lon, path_lat = zip(*path)


            list_actions_for_entity[2][1] = path_lon  # 经度序列
            list_actions_for_entity[2][2] = path_lat  # 纬度序列
            altitude = list_actions_for_entity[2][3] 
            velocity = list_actions_for_entity[2][4] 
            list_actions_for_entity[2][3] = [altitude] * len(dlon)
            list_actions_for_entity[2][4] = [velocity] * len(dlon)
            
            patrol_count += 1
        
        else: # 实体不在交战在区域内
            if raw_data[entity_idx].get('logisticStates', {}).get('oil', -1) > 0.6:
                new_lon, new_lat = get_area_target_point(entity_id, attack_point_list)
                list_actions_for_entity[2][1] = new_lon
                list_actions_for_entity[2][2] = new_lat 
                new_direction_index = compute_direction_index((cur_lon, cur_lat), (new_lon, new_lat))
                actions[:, i, :][2][1] = new_direction_index 
                
                patrol_count += 1
    
    else:
        if is_point_in_area((cur_lon, cur_lat), attack_point_list):
            dlon, dlat = DIRECTION_OFFSET[direction_index]
            if not isinstance(dlon, list): #  单值
                new_lon = cur_lon + dlon
                new_lat = cur_lat + dlat
                # 替换动作中的经纬度位置
                list_actions_for_entity[2][1] = new_lon  # 经度
                list_actions_for_entity[2][2] = new_lat  # 纬度
            else:
                # 复合动作，包含一系列路径点
                if direction_index==8:
                    path = generate_random_patrol(attack_point_list,(cur_lon,cur_lat))
                    path_lon, path_lat = zip(*path)
                if direction_index==9:
                    path = generate_bow_patrol(attack_point_list,(cur_lon,cur_lat),num_area=5)
                    path_lon, path_lat = zip(*path)
                if direction_index==10:
                    path  = generate_Z_patrol(attack_point_list,(cur_lon,cur_lat),num_slash=3)
                    path_lon, path_lat = zip(*path)
                else:
                    path = generate_spiral_patrol(attack_point_list,(cur_lon,cur_lat),num_loops=4)
                    path_lon, path_lat = zip(*path)

                list_actions_for_entity[2][1] = path_lon  # 经度序列
                list_actions_for_entity[2][2] = path_lat  # 纬度序列
                altitude = list_actions_for_entity[2][3] 
                velocity = list_actions_for_entity[2][4] 
                list_actions_for_entity[2][3] = [altitude] * len(dlon)
                list_actions_for_entity[2][4] = [velocity] * len(dlon)
        
        else: # 实体不在交战在区域内
            if raw_data[entity_idx].get('logisticStates', {}).get('oil', -1) > 0.6:
                new_lon, new_lat = get_area_target_point(entity_id, attack_point_list)
                list_actions_for_entity[2][1] = new_lon
                list_actions_for_entity[2][2] = new_lat 
                new_direction_index = compute_direction_index((cur_lon, cur_lat), (new_lon, new_lat))
                actions[:, i, :][2][1] = new_direction_index 
                
                patrol_count += 1
        
            
            

def handle_waypoint_move(list_actions_for_entity, i, entity_id, our_entity_coordinate, actions):
    """
    处理航路机动索引->经纬度
    i 表示实体的索引 表述第i个实体
    """
    scale_rate = 1.0
    
    waypoint_action = list_actions_for_entity[2]  # [prob, direction_index, NONE, altitude, velocity]
    direction_index = int(waypoint_action[1])  # 注意：方向索引存储在 [2][1]，即 waypoint_action[1]
 
    if direction_index in DIRECTION_OFFSET:
        if direction_index in range(8, 12):
            direction_index = random.randint(0, 7)
            # waypoint_action[1] = direction_index
            actions[:, i, :][2][1] = direction_index
        cur_lon, cur_lat = our_entity_coordinate[i]
        dlon, dlat = DIRECTION_OFFSET[direction_index]

        new_lon = cur_lon + dlon * scale_rate
        new_lat = cur_lat + dlat * scale_rate
        # 替换动作中的经纬度位置
        list_actions_for_entity[2][1] = new_lon  # 经度
        list_actions_for_entity[2][2] = new_lat  # 纬度


def handle_waypoint_move_patrol(list_actions_for_entity, i, entity_id, our_entity_coordinate, actions, attack_point_list):
    """
    巡逻的情况
    处理航路机动索引->经纬度
    i 表示实体的索引 表述第i个实体

    判断是不是巡逻机，如果是巡逻机，就向交战区域发送航路机动指令 while 循环判断是否在交战区。 
    根据交战区域， 修改scale_rate。 
    需要找一个算法，输入一个区域，输出一个scale_rate。
    """
    waypoint_action = list_actions_for_entity[2]  # [prob, direction_index, NONE, altitude, velocity]
    direction_index = int(waypoint_action[1])  # 注意：方向索引存储在 [2][1]，即 waypoint_action[1]
    scale_rate = 0.1
 
    if direction_index in DIRECTION_OFFSET:
        cur_lon, cur_lat = our_entity_coordinate[i]
        if is_point_in_area((cur_lon, cur_lat) , attack_point_list): # 实体在在区域内 
            if direction_index in [0, 1, 2, 3, 4, 5, 6, 7]: # 不要0-7的方向 用巡逻的
                direction_index = random.randint(8, 11)
                # waypoint_action[1] = direction_index
                actions[:, i, :][2][1] = direction_index
            dlon, dlat = DIRECTION_OFFSET[direction_index]
            if len(dlon) == 1: #  单值,但是没有用，因为前面强制给设置成8 ~ 11了
                new_lon = cur_lon + dlon * scale_rate
                new_lat = cur_lat + dlat * scale_rate
                # 替换动作中的经纬度位置
                list_actions_for_entity[2][1] = new_lon  # 经度
                list_actions_for_entity[2][2] = new_lat  # 纬度
            else:
                # 复合动作，包含一系列路径点
                if direction_index==8:
                    path = generate_random_patrol(attack_point_list,(cur_lon,cur_lat))
                    path_lon, path_lat = zip(*path)
                if direction_index==9:
                    path = generate_bow_patrol(attack_point_list,(cur_lon,cur_lat),num_area=5)
                    path_lon, path_lat = zip(*path)
                if direction_index==10:
                    path  = generate_Z_patrol(attack_point_list,(cur_lon,cur_lat),num_slash=3)
                    path_lon, path_lat = zip(*path)
                else:
                    path = generate_spiral_patrol(attack_point_list,(cur_lon,cur_lat),num_loops=4)
                    path_lon, path_lat = zip(*path)


                # 将路径写入动作（你需要根据后续系统接受的数据格式来设定）
                list_actions_for_entity[2][1] = path_lon  # 经度序列
                list_actions_for_entity[2][2] = path_lat  # 纬度序列
                altitude = list_actions_for_entity[2][3] 
                velocity = list_actions_for_entity[2][4] 
                list_actions_for_entity[2][3] = [altitude] * len(dlon)
                list_actions_for_entity[2][4] = [velocity] * len(dlon)

        else: # 实体不在交战在区域内)
            new_lon, new_lat = get_area_target_point(entity_id, attack_point_list)
            list_actions_for_entity[2][1] = new_lon
            list_actions_for_entity[2][2] = new_lat 
            new_direction_index = compute_direction_index((cur_lon, cur_lat), (new_lon, new_lat))
            actions[:, i, :][2][1] = new_direction_index  # 替换方向索引





def handle_attack_decision(actions, now_state, i, entity_idx, entity_id, our_entity_indices, our_can_attack_ids,
                           enemy_ids, enemy_indices, enemy_coordinate, actions_dict, not_aim_count, 
                           out_of_range_count, actor_property, list_actions_for_entity, raw_data,
                           our_entity_coordinate, sub2sub_max_range_nm, logger, urgent_flag, no_enemy_count):
    def disable_attack(reason):
        actions[:, i, :][4, 0] = np.float32(0.01)
        list_actions_for_entity[4][0] = np.float32(0.01)
        list_actions_for_entity[4][1] = reason

    own_raw = raw_data[entity_idx]
    inventory = own_raw.get('weaponNumber', {})
    if not isinstance(inventory, dict):
        inventory = {}
    enough_ammo = any(float(inventory.get(key, 0) or 0) > 0 for key in ('airNum', 'subNum', 'shipNum'))
    oil = float(own_raw.get('logisticStates', {}).get('oil', -1) or 0)
    fuel_low = 0 < oil <= 0.20
    jammed = int(own_raw.get('innerstates', {}).get('IsJamReaction', 0) or 0) > 0

    if our_entity_indices[i] not in our_can_attack_ids or not enough_ammo:
        disable_attack('平台没有可用的对应域武器')
        return not_aim_count, out_of_range_count, no_enemy_count
    if fuel_low or jammed:
        disable_attack('燃油或安全条件不满足')
        return not_aim_count, out_of_range_count, no_enemy_count + 1

    own_type = int(now_state['encoded_data'][entity_idx, 13])
    own_alt = float(now_state['encoded_data'][entity_idx, 5])
    own_heading = float(now_state['encoded_data'][entity_idx, 4])
    cur_lon, cur_lat = our_entity_coordinate[i]
    target_inputs = []
    for target_position, enemy_idx_abs in enumerate(enemy_indices):
        if target_position >= len(enemy_ids) or target_position >= len(enemy_coordinate):
            continue
        tgt_lon, tgt_lat = enemy_coordinate[target_position]
        target_inputs.append(AttackTargetInput(
            index=target_position,
            target_id=str(enemy_ids[target_position]),
            raw=raw_data[enemy_idx_abs],
            entity_type=int(now_state['encoded_data'][enemy_idx_abs, 13]),
            altitude_m=float(now_state['encoded_data'][enemy_idx_abs, 5]),
            longitude=float(tgt_lon),
            latitude=float(tgt_lat),
        ))

    decision = decide_attack(
        own_raw=own_raw,
        own_type=own_type,
        own_altitude_m=own_alt,
        own_heading_deg=own_heading,
        own_longitude=float(cur_lon),
        own_latitude=float(cur_lat),
        targets=target_inputs,
    )
    if logger is not None:
        logger.debug(
            '[MADDPG规则] entity=%s conclusion=%s rule=%s target=%s reason=%s',
            entity_id,
            decision.conclusion,
            decision.rule_id,
            decision.candidate.target_id if decision.candidate else None,
            decision.reason,
        )

    candidate = decision.candidate
    if decision.conclusion == 'REQUEST_ATTACK' and candidate is not None:
        probability = max(float(actions[:, i, :][4, 0]), 0.81)
        actions[:, i, :][4, 0] = np.float32(probability)
        list_actions_for_entity[4][0] = np.float32(probability)
        list_actions_for_entity[4][1] = candidate.target_id
        list_actions_for_entity[4][2] = 0.0
        list_actions_for_entity[4][3] = 0.0
        # 新 AttackTarget 接口使用该字段作为数量；旧接口会安全忽略数值数量。
        list_actions_for_entity[4][4] = int(decision.quantity or DEFAULT_ATTACK_QUANTITY)
        return not_aim_count, out_of_range_count, no_enemy_count

    disable_attack(decision.reason)
    if decision.conclusion in ('CHASE_TO_RANGE', 'CHASE_AND_ALIGN') and candidate is not None:
        if urgent_flag:
            return not_aim_count, out_of_range_count, no_enemy_count
        attack_altitude = calculate_attack_altitude(own_type, candidate.altitude_m)
        direction = get_waypoint_index(candidate.longitude - cur_lon, candidate.latitude - cur_lat)
        actions[:, i, :][2][0] = np.float32(0.8)
        actions[:, i, :][2][1] = np.float32(direction)
        actions[:, i, :][2][3] = np.float32(5)
        actions[:, i, :][2][4] = np.float32(attack_altitude)
        actions[:, i, :][3][0] = np.float32(0.01)
        list_actions_for_entity[2][0] = np.float32(0.8)
        list_actions_for_entity[2][1] = candidate.longitude
        list_actions_for_entity[2][2] = candidate.latitude
        list_actions_for_entity[2][3] = np.float32(5)
        list_actions_for_entity[2][4] = np.float32(attack_altitude)
        list_actions_for_entity[3][0] = np.float32(0.01)
        if decision.conclusion == 'CHASE_AND_ALIGN':
            not_aim_count += 1
        else:
            out_of_range_count += 1
    else:
        no_enemy_count += 1

    return not_aim_count, out_of_range_count, no_enemy_count

def calculate_attack_altitude(own_type, tgt_alt):
    """
    计算攻击高度
    """
    attack_alt_type = 0
    if tgt_alt > 0: # 打击敌方高于0，我方如果是飞机的话，就让我方到最大高度攻击，我方是舰艇或潜艇的话，在水面攻击
        if own_type == 0: # 如果我方是飞机，就让我方到最大高度攻击
            if tgt_alt > 5000:
                attack_alt_type = 5
            else:
                attack_alt_type = 3
        elif own_type == 1: # 如果我方是舰艇，就让我方到水面进行攻击
            attack_alt_type = 0
        elif own_type == 2: # 如果我方是潜艇，就让我方到水面进行攻击
            attack_alt_type = 1
    elif tgt_alt < 0: # 敌方低于0，我方如果是飞机的话，就让我方到较低高度攻击，我方是舰艇在水面攻击，我方是潜艇的话，在最深处进行攻击
        if own_type == 0:
            attack_alt_type = 0
        elif own_type == 1 :
            attack_alt_type = 0
        elif own_type == 2:
            attack_alt_type = 3
    else: # 打水面的船
        if own_type == 0: # 如果我方是飞机，就让我方到次低高度攻击
            attack_alt_type = 1
        elif own_type == 1: # 如果我方是舰艇，就让我方到水面进行攻击
            attack_alt_type = 0
        elif own_type == 2: # 如果我方是潜艇，就让我方到水面进行攻击
            attack_alt_type = 1
    return attack_alt_type

def judge_max_distance(own_type, own_raw, tgt_type, tgt_type2):
    scale = 0
    if own_type == 0:
        scale = 1.5
    elif own_type == 1:
        scale = 1.2
    elif own_type == 2:
        scale = 1.1

    scale4tgt = 0
    if tgt_type == 0:
        scale4tgt = 1
    elif tgt_type == 1:
        scale4tgt = 1
    elif tgt_type == 2:
        scale4tgt = 20
    elif tgt_type == 4:
        scale4tgt = 1

    max_ranges = own_raw.get('maxRange', {})
    dis_air    = max_ranges.get('maxAir', 0)
    dis_sub    = max_ranges.get('maxSubsurface', 0)
    dis_ship   = max_ranges.get('maxSurface', 0)

    max_dis_chase = 0
    max_dis = 0
    if tgt_type == 0:
        max_dis = dis_air * 1842.0
        max_dis_chase = max_dis * scale * scale4tgt
    elif tgt_type == 1:
        max_dis = dis_ship * 1842.0
        max_dis_chase = max_dis * scale * scale4tgt
    elif tgt_type == 2:
        max_dis = dis_sub * 1842.0
        max_dis_chase = max_dis * scale * scale4tgt
    elif tgt_type == 4:
        if tgt_type2 in [0, 1, 11, 13]: # 水上武器
            max_dis = dis_air * 1842.0
        elif tgt_type2 in [2, 3, 9, 10, 16, 17]: # 水下
            max_dis = dis_sub * 1842.0
        max_dis_chase = max_dis * scale * scale4tgt


    return  max_dis, max_dis_chase


def check_attack_range(now_state, i, entity_idx, tgt_alt, attack_distant, own_raw, own_type, tgt_type, tgt_type2):
    """
    检查是否在攻击范围内
    """
    # range_index = 29 if tgt_alt > 0 else 30 if tgt_alt < 0 else 31
    # within_attack_range = (float(now_state['encoded_data'][entity_idx, range_index] * 1842) >= attack_distant)  # 因为无对地 所以仅通过目标高度来判断是对空还是对水下还是对水面

    # # 潜艇对潜艇的攻击距离特殊处理
    # chase_allowed = True                         # 默认允许追击
    # if own_type == 2:
    #     sub2sub_range_m = sub2sub_max_range_nm * 1842.0
    #     if attack_distant > sub2sub_range_m:
    #         within_attack_range = False           # 强制判为“超出范围”
    #         chase_allowed = False
    #

    within_attack_range = True
    chase_allowed = True  # 默认允许追击
    max_dis, max_dis_chase = judge_max_distance(own_type, own_raw, tgt_type, tgt_type2)
    if attack_distant > max_dis:
        within_attack_range = False  # 强制判为“超出范围”
    if attack_distant > max_dis_chase:
        chase_allowed = False # 如果超出了最大追击范围则不进行追击

    return within_attack_range, chase_allowed

def check_aim_range(now_state, i, entity_idx, tgt_lon, tgt_lat, cur_lon, cur_lat, own_type):
    """
    检查是否对准目标
    """
    cur_yaw_deg = float(now_state['encoded_data'][entity_idx, 4].item())   # 当前航向  角度
    cur_head    = math.radians(cur_yaw_deg) # 当前航向  角度->弧度
    tgt_bearing = _bearing(cur_lon, cur_lat, tgt_lon, tgt_lat) #返回两点之间的航向弧度差
    ANGLE_TH = math.radians(30)          # 允许误差 30°   弧度
    # 判断是否对准
    within_aim_range = (abs(_rad_diff(cur_head, tgt_bearing)) < ANGLE_TH)
    if own_type == 1: # 我方是舰艇，就不考虑对准与否的问题
        within_aim_range = True
    return within_aim_range

def handle_chase(within_aim_range, within_attack_range, enemy_ids, target_idx, tgt_lon, tgt_lat, cur_lon, cur_lat, 
                 chase_allowed, list_actions_for_entity, actions, i, attack_alt_type, not_aim_count, out_of_range_count,
                 urgent_flag, tgt_type, own_type, attack_distant):
    """
    处理追击逻辑
    """
    if within_aim_range and within_attack_range: # 如果对准了，就进行攻击
        list_actions_for_entity[4][1] = enemy_ids[target_idx]  # enemy_ids[target_idx]拿到的其实是敌方的mdlID
        if tgt_type == 2  and own_type == 0:
            list_actions_for_entity[4][0] = 0.82
            list_actions_for_entity[4][2] = tgt_lon
            list_actions_for_entity[4][3] = tgt_lat

    else:
        if urgent_flag:
            actions[:, i, :][4][0] = np.float32(0.0123)
            list_actions_for_entity[4][0] = np.float32(0.0123)
            return
        # 未对准 or 距离太远：把本帧 attack_prob 压到极低，修改航路机动的指令参数 以及修改action
        actions[:, i, :][4][0] = np.float32(0.0123)
        list_actions_for_entity[4][0] = 0.0123
        if not within_aim_range:
            not_aim_count+=1
            list_actions_for_entity[4][1] = '没有对准' + str(enemy_ids[target_idx])
            list_actions_for_entity[4][0] = 0.88
            list_actions_for_entity[4][2] = tgt_lon
            list_actions_for_entity[4][3] = tgt_lat
        if not  within_attack_range:
            out_of_range_count+=1
            list_actions_for_entity[4][1] = '超出范围' + str(enemy_ids[target_idx])
        if chase_allowed: #允许追击
            diff_lon = tgt_lon - cur_lon
            diff_lat = tgt_lat - cur_lat
            # 1. 让机动 Actor 生效(修改all_action和执行的命令) 同时冻结调整高度和速度
            actions[:, i, :][2][0] = np.float32(0.8)
            actions[:, i, :][2][1] = np.float32(get_waypoint_index(diff_lon, diff_lat))
            # actions[:, i, :][2][1] = None
            actions[:, i, :][2][3] = np.float32(5) # 速度量化
            actions[:, i, :][2][4] = np.float32(attack_alt_type) # 高度量化
            actions[:, i, :][3][0] = np.float32(0.01)
            list_actions_for_entity[2][0] = np.float32(0.8)
            list_actions_for_entity[2][1] = tgt_lon
            list_actions_for_entity[2][2] = tgt_lat
            list_actions_for_entity[2][3] = np.float32(5)
            list_actions_for_entity[2][4] = np.float32(attack_alt_type)
            list_actions_for_entity[3][0] = np.float32(0.01)
            if attack_distant < 2000:
                actions[:, i, :][2][3] = np.float32(1)  # 速度量化
                list_actions_for_entity[2][3] = np.float32(1)

            # 2. 冻结攻击
            actions[:, i, :][4][0] = np.float32(0.0123)
            list_actions_for_entity[4][0] = np.float32(0.0123)


            

def handle_deploy(list_actions_for_entity, actions, i, entity_idx, our_entity_coordinate, raw_data, mission_dicts, altitude, actor_property):
    deploy_action = list_actions_for_entity[6]
    if deploy_action[0] <= actor_property:
        return 0

    own_raw = raw_data[entity_idx]
    mission_id = own_raw.get('missionId')
    mission = mission_dicts.get(mission_id, {})
    area_points = mission.get('area_points', [])
    cur_lon, cur_lat = our_entity_coordinate[i]
    allowed = sonobuoy_deployment_allowed(
        own_raw,
        cur_lon,
        cur_lat,
        altitude,
        area_points,
    )
    if not allowed:
        list_actions_for_entity[6][0] = np.float32(0.01)
        actions[:, i, :][6][0] = np.float32(0.01)
        return 1

    # 与符号推理一致：浅层主动声呐浮标，并禁止同帧巡逻航路覆盖部署。
    list_actions_for_entity[6][0] = np.float32(max(float(deploy_action[0]), 0.81))
    list_actions_for_entity[6][1] = np.float32(1.0)
    list_actions_for_entity[6][2] = np.float32(1.0)
    actions[:, i, :][6][0] = list_actions_for_entity[6][0]
    actions[:, i, :][6][1] = np.float32(1.0)
    actions[:, i, :][6][2] = np.float32(1.0)
    list_actions_for_entity[2][0] = np.float32(0.01)
    actions[:, i, :][2][0] = np.float32(0.01)
    return 0

MAX_ON_AIR_NUM = 5
def handle_take_off_num_rule(actions, now_state, i, entity_idx, entity_id, aircraft_unitcategory_num, actor_property):
    if now_state['encoded_data'][entity_idx][13] == 0 and now_state['encoded_data'][entity_idx][33] != -1: # 保证是飞机 之后可以加一个战斗机要在有敌人的情况才起飞？？
        if actions[:, i, :][0][0] >= actor_property: # 如果确实要起飞
            # 判断飞机在空字典中该类飞机是否满员
            unitCategoty = int(now_state['encoded_data'][entity_idx][33].item())
            if aircraft_unitcategory_num[unitCategoty] < MAX_ON_AIR_NUM:
                if random.random() < 0.7: # 随机选择该飞机起飞 避免前面的飞机一直能占位置
                    aircraft_unitcategory_num[unitCategoty] += 1
                    actions[:, i, :][0][0] = 0.88
                else:
                    actions[:, i, :][0][0] = 0.22
                return 0
            else:
                actions[:, i, :][0][0] = 0.22
                return 1
        return 0
    else:
        return 0
