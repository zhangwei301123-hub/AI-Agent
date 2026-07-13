import json
import numpy as np
from typing import Dict, Optional
import torch
from numpy.core.defchararray import upper
from torch.utils.hipify.hipify_python import value

from entity import *
import os
import execute


# attack type  潜艇  船 武器  飞机 un-hit
base_attack_event_reward = np.array([
    [  2.0,   1.5,   1.0,   0.8,   0.0],   # AircraftTakeOffActor
    [  0.0,   0.0,   0.0,   0.0,   0.0],   # ReturnToBaseActor
    [  2.0,   1.5,   1.0,   0.8,   0.0],   # WayPointMoveActor
    [  3.0,   2.0,   1.0,   1.0,   0.0],   # MobilityActor
    [ 20.0,   15.0,  13.0, 12.0, -10.0],   # AttackTargetActor（命中得分最高，如果未命中则-5）
    [  2.5,   2.0,   1.5,   1.0,   0.0],   # SensorControlActor
    [  3.0,   2.0,   1.0,   3.0,   0.0],   # DeploySonobuoyActor
    [  0.0,   0.0,   0.0,   0.0,   0.0],   # CancelAttackActor
], dtype=float)

# loss type  潜艇  船  飞机
base_loss_event_reward=np.array([
    #    Submarine, Ship, Weapon, Aircraft
    [     0,         0,     -3],   # AircraftTakeOffActor
    [     0,         0,     -2],   # ReturnToBaseActor
    [    -4,        -4,     -3],   # WayPointMoveActor
    [    -5,        -4,     -3],   # MobilityActor
    [    -3,        -2,     -1],   # AttackTargetActor
    [    -3,        -3,     -2],   # SensorControlActor
    [    -6,        -5,     -4],   # DeploySonobuoyActor
    [     0,         0,      0],   # CancelAttackActor
], dtype=float)

# 潜艇  船 武器  飞机 未知
base_find_event_reward=np.array([
    [  1.2,   1.1,   1.0,   0.8, 0.8],   # AircraftTakeOffActor 飞机起飞
    [  0,   0.1,   0.1,   0.1, 0.1],   # ReturnToBaseActor
    [  12.0,   7,   5.0,   4, 4 ],   # WayPointMoveActor 航线机动
    [  3.0,   2.0,   1.0,   1.0, 1.0],   # MobilityActor
    [  0.0,   0.0,   0.0,   0.0, 0.0],   # AttackTargetActor
    [  2.5,   2.0,   1.5,   1.0, 1.0],   # SensorControlActor
    [  3.0,   2.0,   1.5,   1.5, 1.5],   # DeploySonobuoyActor
    [  0.0,   0.0,   0.0,   0.0, 0.0],   # CancelAttackActor
]) * 0.8
 

class CombatRewardCalculator:
    DEFAULT_CONFIG = {

        # 公共参数
        "time_decay": 0.0001,
        # "max_reward": 15.0,
        # "min_reward": -8.0,
        # # 攻击事件奖励配置（各 actor 可根据实际情况在配置中扩展字段）
        # "attack_reward": 2.0,
        # # 损失事件惩罚配置（全局惩罚，可在具体 actor 配置中覆盖）
        # # 上一次该实体执行的动作的惩罚
        # "loss_penalty": -1.5,
        # # 新目标发现奖励（用于敌方新增检测）
        # "discovery_reward": 3.0,

    }


    def __init__(self, actor_probability):

        self.actor_probability = actor_probability
        self.config = self.DEFAULT_CONFIG

    def load_attack_info(self, responses):  # 接口版本
        result = True
        if responses:
            raw_data = responses
            attack_ids = []
            fire_time = []
            attack_type = []
            target_damage_degree = []
            
            for e in raw_data:
                if not isinstance(e, dict):
                    continue
                    
                # Save current lengths in case we need to revert
                current_len = len(attack_ids)
                
                try:
                    # Try to process all fields
                    temp_attack_id = e.get('launch_platform_id', '')
                    temp_fire_time = int(e.get('fire_time', 0))
                    
                    enemy_type = e.get('attack_target_type', '').upper()
                    if enemy_type == 'SUBMARINE':
                        temp_attack_type = 0
                    elif enemy_type == 'SHIP':
                        temp_attack_type = 1
                    elif enemy_type == 'WEAPON':
                        temp_attack_type = 2
                    elif enemy_type == 'AIRCRAFT':
                        temp_attack_type = 3
                    else:
                        temp_attack_type = 4  # 未命中
                        
                    temp_damage = int(e.get('target_damage_degree', 0))  # 0~100
                    
                    # If we get here without errors, append all values
                    attack_ids.append(temp_attack_id)
                    fire_time.append(temp_fire_time)
                    attack_type.append(temp_attack_type)
                    target_damage_degree.append(temp_damage)
                    
                except Exception as ex:
                    # If any error occurs, revert by truncating lists to previous length
                    attack_ids = attack_ids[:current_len]
                    fire_time = fire_time[:current_len]
                    attack_type = attack_type[:current_len]
                    target_damage_degree = target_damage_degree[:current_len]
                    continue
                    
                    # else:


            return result, attack_ids, fire_time, attack_type, target_damage_degree
        else:
            return [], [], [], []

    def load_loss_info(self, response):  # 接口版
        """加载损失信息
        参数:
            response: 原始响应数据
        返回:
            tuple: (处理结果, 损失ID列表, 损失类型列表, 损伤程度列表)
        """
        if not response:
            return False, [], [], []
        
        result = True
        loss_ids = []
        loss_type = []
        loss_damage_degree = []
        
        for e in response:
            if not isinstance(e, dict):
                continue
                
            # 保存当前列表长度以便回滚
            current_len = len(loss_ids)
            
            try:
                # 尝试处理所有字段
                temp_id = e.get('id')
                temp_type = e.get('type', '').upper()
                temp_damage = int(e.get('damage_degree', 0))
                
                # 处理类型转换
                type_mapping = {
                    'SUBMARINE': 0,
                    'SHIP': 1,
                    'AIRCRAFT': 2,
                    'WEAPON': 3
                }
                temp_type_code = type_mapping.get(temp_type, -1)  # -1表示未知类型
                if temp_type_code == 3:
                    raise "导弹损失"

                # 如果所有处理成功，则添加到列表
                loss_ids.append(temp_id)
                loss_type.append(temp_type_code)
                loss_damage_degree.append(temp_damage)
                
            except Exception as ex:
                # 发生错误时回滚当前迭代的更改
                loss_ids = loss_ids[:current_len]
                loss_type = loss_type[:current_len]
                loss_damage_degree = loss_damage_degree[:current_len]
                continue
                
        return result, loss_ids, loss_type, loss_damage_degree


    def load_detection_report(self, response):  # 接口版本
        """加载探测报告信息
        参数:
            response: 原始响应数据
        返回:
            tuple: (处理结果, 目标ID列表, 目标类型列表, 探测ID列表, 
                探测类型列表, 探测步骤列表, 是否声呐浮标列表)
        """
        if not response:
            return False, [], [], [], [], [], []
        
        result = True
        target_ids = []
        target_types = []
        detector_ids = []
        detector_types = []
        detect_step = []
        is_sonobuoy = []
        
        for e in response:
            if not isinstance(e, dict):
                continue
                
            # 保存当前列表长度以便回滚
            current_len = len(target_ids)
            
            try:
                # 尝试处理所有字段
                temp_target_id = e.get('targetId')
                temp_detector_id = e.get('detectorId')
                temp_step = int(e.get('detectStep', 0))
                temp_sonobuoy = bool(e.get('isSonobuoy', False))
                
                # 处理目标类型转换
                target_type = e.get('targetType', 'SHIP').upper() # 防止bug 默认用SHIP
                target_type_mapping = {
                    'SUBMARINE': 0,
                    'SHIP': 1,
                    'AIRCRAFT': 2,
                    'WEAPON': 3
                }
                temp_target_type = target_type_mapping.get(target_type, 4)  # 4表示未知类型
                
                # 处理探测类型转换
                detector_type = e.get('detectorType', '').upper()
                detector_type_mapping = {
                    'SUBMARINE': 0,
                    'SHIP': 1,
                    'AIRCRAFT': 2
                }
                temp_detector_type = detector_type_mapping.get(detector_type, -1)  # -1表示未知类型
                
                # 如果所有处理成功，则添加到列表
                target_ids.append(temp_target_id)
                target_types.append(temp_target_type)
                detector_ids.append(temp_detector_id)
                detector_types.append(temp_detector_type)
                detect_step.append(temp_step)
                is_sonobuoy.append(temp_sonobuoy)
                
            except Exception as ex:
                # 发生错误时回滚当前迭代的更改
                target_ids = target_ids[:current_len]
                target_types = target_types[:current_len]
                detector_ids = detector_ids[:current_len]
                detector_types = detector_types[:current_len]
                detect_step = detect_step[:current_len]
                is_sonobuoy = is_sonobuoy[:current_len]
                continue
                
        return result, target_ids, target_types, detector_ids, detector_types, detect_step, is_sonobuoy
        
    def update_attack_buffer_reward(self, maddpg, attack_step, attack_id, attack_type, target_damage_degree, logger, entity_action_frame_link):
        '''

        attack_type  
        0 :SUBMARINE
        1 :SHIP
        2 :Weapon
        3 :Aircraft
        4 :un-hit
        '''
        # print('--------------update_attack_buffer_reward')
        logger.info('--------------update_attack_buffer_reward')
        try:
            # 首先构建反向映射：本地帧->服务器帧（如果sever_our是{服务器帧:本地帧}）
            local_to_server = {v:k for k,v in maddpg.sever_our.items()} if hasattr(maddpg, 'sever_our') else {}
            for i in range(len(attack_step)):
                # 这里会出现下发攻击指令的帧数和实际记录攻击指令的帧数不一致 往往实际记录的会略迟于真实发布的
                current_server_frame = attack_step[i]
                entity_id = attack_id[i]
                
                # 1. 在entity_action_frame_link中查找该实体最近的攻击动作
                candidate_frames = []
                if entity_id in entity_action_frame_link:
                    for action_record in entity_action_frame_link[entity_id]:
                        for local_frame, actions in action_record.items():
                            # 检查是否包含攻击动作（4是AttackTargetActor的索引）
                            if 4 in actions: 
                                # 获取该本地帧对应的服务器帧
                                server_frame = local_to_server.get(local_frame, None)
                                if server_frame is not None and server_frame <= current_server_frame:
                                    candidate_frames.append((local_frame, server_frame))
                
                # 2. 选择最接近当前服务器帧的记录
                if candidate_frames:
                    # 按服务器帧降序排序，取最接近但不超过当前服务器帧的记录
                    candidate_frames.sort(key=lambda x: x[1], reverse=True)
                    local_step, _ = candidate_frames[0]
                    # logger.info(f"实体 {entity_id} 在服务器帧 {current_server_frame} 对应本地帧 {local_step}")
                else:
                    logger.warning(f"找不到实体 {entity_id} 在服务器帧 {current_server_frame} 之前的攻击记录")
                    continue
                
                store_experience = maddpg.buffer.get_latest_episode()[local_step]

                action_entity_ids = store_experience['action_entity_id']

                # action_entity_id = [j for j, id in enumerate(action_entity_ids) if id == attack_id[i]][0]
                # 为了防止错误，先判断是否在列表中，不在则返回-1，就不再判断该实体
                action_entity_id = next((j for j, id in enumerate(action_entity_ids) if id == attack_id[i]), -1)
                if action_entity_id == -1:
                    continue  #


                actions = store_experience['actions']  # shape: (9, num_entities, 5)
                rewards = store_experience['rewards']  # shape: (9)
                actions_executed = store_experience['actions_executed']

                num_actors = actions.shape[0]
                action_entity_id_idx = np.full(num_actors, action_entity_id)
                # 这一步提取的是：每个 Actor 对该实体（比如第 2 个）的动作值。
                selected_actions = actions[np.arange(num_actors), action_entity_id_idx, 0]
                actor_execution = (selected_actions > self.actor_probability).astype(int).reshape(-1)  # 这里其实可以不用了

                actions_executed_np = np.array(actions_executed[action_entity_id], dtype=np.bool_)
                execute_musk = actor_execution.astype(np.bool_) & actions_executed_np.astype(np.bool_)
                reward = np.zeros_like(rewards, dtype=float)

                reward[execute_musk] = base_attack_event_reward[execute_musk, attack_type[i]]

                time_decay = self.config.get("time_decay", 1.0)
                factor = (1 / (1 + time_decay * local_step)) * (target_damage_degree[i] / 100) if attack_type[i] != 4 else (2 - 1 / (1 + time_decay * local_step))
                reward *= factor
                # print("reward:", reward)
                logger.info(f"reward: {reward}")
                rewards += reward
                # print('origin reward:', maddpg.buffer.get_latest_episode()[local_step]['rewards'])
                maddpg.buffer.get_latest_episode()[local_step]['rewards'] = rewards
                # print('change reward:', maddpg.buffer.get_latest_episode()[local_step]['rewards'])
                logger.info(f"change reward: {maddpg.buffer.get_latest_episode()[local_step]['rewards']}")
        except Exception as e:
            logger.error(e)
            return


    def update_loss_buffer_reward(self, maddpg, loss_entity_id, loss_type, loss_damage_degree, logger):
        '''
        aloss_type  
        0 :SUBMARINE
        1 :SHIP
        3 :Weapon
        2 :Aircraft
        '''
        # print('--------------update_loss_buffer_reward')
        logger.info('--------------update_loss_buffer_reward')
        try:
            flags = [False] * len(loss_entity_id)
            for store_experience in reversed(maddpg.buffer.get_latest_episode()):  # 倒着取每一帧
                if all(flags):
                    break

                for i, loss_entity in enumerate(loss_entity_id):
                    if flags[i]:
                        continue

                    step = store_experience['step']
                    step = maddpg.sever_our[step]  # local的step/frame
                    action_entity_ids = store_experience['action_entity_id']  # 拿到全部的动作实体ids

                    # 拿到执行动作的实体id的索引   因为是个只有一个值的列表 所以拿第0个作为值
                    action_entity_id = [i for i, id in enumerate(action_entity_ids) if (id == loss_entity)]

                    actions = store_experience['actions']  # shape: (9, num_entities, 5)
                    rewards = store_experience['rewards']  # shape: (9)

                    num_actors = actions.shape[0]
                    # 这是为了防止  ValueError('could not broadcast input array from shape (3,) into shape (8,)')
                    if not action_entity_id:
                        logger.warning(f"loss_entity {loss_entity} not found in action_entity_ids: {action_entity_ids}")
                        continue
                    entity_idx = action_entity_id[0]  # 只取第一个匹配的 index
                    action_entity_id_idx = np.full(num_actors, entity_idx)


                    # action_entity_id_idx = np.full(num_actors, action_entity_id)  # 填充一个列表, 填充的值是 action_entity_id
                    # 这一步提取的是：每个 Actor 对该实体（比如第 2 个）的动作值(0.8,x,x,x,x )
                    selected_actions = actions[np.arange(num_actors), action_entity_id_idx, 0]
                    actor_execution = (selected_actions > self.actor_probability).astype(int).reshape(-1)

                    execute_musk = actor_execution.astype(np.bool_)  # 每个 Actor 对该实体（比如第 2 个）是否有执行动作
                    if not np.any(execute_musk[:-1]):  # 如果没有执行动作，则跳过该帧
                        continue
                    else:
                        flags[i] = True  # 如果有执行动作，就认为该实体找到了所在帧率，已经被解决

                    reward = np.zeros_like(rewards, dtype=float)
                    reward[execute_musk] = base_loss_event_reward[execute_musk, loss_type[i]]

                    time_decay = self.config.get("time_decay", 1.0)
                    factor = (2 - 1 / (1 + time_decay * step)) * (loss_damage_degree[i] / 100)
                    reward *= factor
                    # print("reward:", reward)
                    logger.info(f"reward: {reward}")
                    rewards += reward
                    # print('origin reward:', maddpg.buffer.get_latest_episode()[step]['rewards'])
                    maddpg.buffer.get_latest_episode()[step]['rewards'] = rewards
                    # print('change reward:', maddpg.buffer.get_latest_episode()[step]['rewards'])
                    logger.info(f"change reward: {maddpg.buffer.get_latest_episode()[step]['rewards']}")

        except Exception as e:
            logger.error(e)
            return



    def update_detection_report_reward(self, maddpg, entity_action_frame_link, target_types, detector_ids, detector_types, detect_step,
                                       is_sonobuoy, logger):
        # print('--------------update_detection_report_reward')
        logger.info('--------------update_detection_report_reward')
        # 动作类型索引
        ACTION_TYPES = {
            "AircraftTakeOffActor": 0,
            "ReturnToBaseActor": 1,
            "WayPointMoveActor": 2,
            "MobilityActor": 3,
            "AttackTargetActor": 4,
            "SensorControlActor": 5,
            "DeploySonobuoyActor": 6,
            "CancelAttackActor": 7
        }
        try:
            for i in range(len(is_sonobuoy)):
                if is_sonobuoy[i]:
                    # 处理声呐浮标发现的情况（原有逻辑）
                    detector_id = detector_ids[i]
                    detect_frame = detect_step[i] - 1
                    detect_frame = maddpg.sever_our[detect_frame]  # 得变成我们的frame
                    target_type_index = target_types[i]

                    # 对该帧的声呐浮标动作进行奖励
                    maddpg.buffer.get_latest_episode()[detect_frame][ACTION_TYPES["DeploySonobuoyActor"]] += \
                    base_find_event_reward[
                        ACTION_TYPES["DeploySonobuoyActor"]][target_type_index]

                    # 通过 entity_action_frame_link 找到该实体对应的部署声呐浮标帧数往前推
                    action_list = entity_action_frame_link.get(detector_id, [])
                    deploy_index = None
                    for idx, frame_info in enumerate(action_list):
                        frame = list(frame_info.keys())[0]
                        if frame == detect_frame and ACTION_TYPES["DeploySonobuoyActor"] in frame_info[frame]:
                            deploy_index = idx
                            break

                    if deploy_index is not None:
                        last_action_frame = {}
                        to_break = False
                        for j in range(deploy_index - 1, -1, -1):
                            frame_info = action_list[j]
                            frame = list(frame_info.keys())[0]
                            actions = frame_info[frame]

                            for action in actions:
                                if action not in last_action_frame:
                                    last_action_frame[action] = frame
                                    if action == ACTION_TYPES["AircraftTakeOffActor"]:
                                        # 对起飞进行奖励
                                        maddpg.buffer.get_latest_episode()[frame][action] += \
                                        base_find_event_reward[action][target_type_index]
                                        # 对起飞帧内的其他动作进行奖励
                                        for other_action in actions:
                                            if other_action != action and other_action not in last_action_frame:
                                                last_action_frame[other_action] = frame
                                                maddpg.buffer.get_latest_episode()[frame][other_action] += \
                                                base_find_event_reward[other_action][
                                                    target_type_index]
                                        to_break = True
                                        break
                                    elif action != ACTION_TYPES["DeploySonobuoyActor"]:
                                        # 对其他动作类型进行奖励
                                        maddpg.buffer.get_latest_episode()[frame][action] += \
                                        base_find_event_reward[action][target_type_index]
                            if to_break:
                                break
                else:
                    # 处理非声呐浮标发现的情况 只能是当帧发现
                    detector_id = detector_ids[i]
                    detect_frame = max(maddpg.sever_our.values())
                    target_type_index = target_types[i]

                    # 找到该实体在 detect_frame 之前最近的有效帧（允许包含声呐浮标动作，但需有其他动作）
                    action_list = entity_action_frame_link.get(detector_id, [])
                    last_valid_frame = None
                    last_valid_actions = []

                    # 从后往前遍历，找到第一个在 detect_frame 之前且至少有一个非声呐浮标动作的帧
                    for frame_info in reversed(action_list):
                        frame = list(frame_info.keys())[0]
                        if frame >= detect_frame:
                            continue  # 跳过当前帧及之后的帧
                        actions = frame_info[frame]
                        # 检查是否有非声呐浮标动作
                        if any(action != ACTION_TYPES["DeploySonobuoyActor"] for action in actions):
                            last_valid_frame = frame
                            last_valid_actions = actions
                            break

                    if last_valid_frame is not None:
                        # 对有效帧中的非声呐浮标动作进行奖励
                        for action in last_valid_actions:
                            if action != ACTION_TYPES["DeploySonobuoyActor"]:
                                maddpg.buffer.get_latest_episode()[last_valid_frame]['rewards'][action] += \
                                base_find_event_reward[action][target_type_index]
                        # print(maddpg.buffer.get_latest_episode()[last_valid_frame]['rewards'])
                        logger.info(f"{maddpg.buffer.get_latest_episode()[last_valid_frame]['rewards']}")

        except Exception as e:
            logger.error(e)
            return




# import numpy as np
# def test1():

#     # 修复后的 action 结构（假设 num_actor=4，max_entity=4）
#     action = np.array([
#         [ # Actor 0
#             [0.8, 0,0,0,0], 
#             [0.5, 0,0,0,0],
#             [0.8, 0,0,0,0],
#             [0.7, 0,0,0,0]
#         ], 
#         [ # Actor 1
#             [0.8, 0,0,0,0], 
#             [0.5, 0,0,0,0],
#             [0.4, 0,0,0,0],
#             [0.3, 0,0,0,0]
#         ], 
#         [ # Actor 2
#             [0.8, 0,0,0,0], 
#             [0.8, 0,0,0,0],
#             [0.8, 0,0,0,0],
#             [0.2, 0,0,0,0]
#         ], 
#         [ # Actor 3
#             [0.8, 0,0,0,0], 
#             [0.5, 0,0,0,0],
#             [0.8, 0,0,0,0],
#             [0.2, 0,0,0,0]
#         ]
#     ])


#     num_actor = action.shape[0]
#     action_entity_id = np.full(num_actor, 2)  # [2,2,2,2]

#     # 提取每个actor的第3个实体的第一个参数
#     selected_actions = action[np.arange(num_actor), action_entity_id, 0]

#     # 判断是否>0.7并转为0/1
#     actor_execution = (selected_actions > 0.7).astype(int).reshape(-1, 1)

#     print(actor_execution)




# def test2():
#     from maddpg_ import MADDPG
#     from actor import AircraftTakeOffActor, ReturnToBaseActor, WayPointMoveActor, MobilityActor, AttackTargetActor, SensorControlActor, DeploySonobuoyActor, CancelAttackActor
#     #test update_attack_buffer_reward
#     actor_types = [
#         AircraftTakeOffActor,
#         ReturnToBaseActor,
#         WayPointMoveActor,
#         MobilityActor,
#         AttackTargetActor,
#         SensorControlActor,
#         DeploySonobuoyActor,
#         CancelAttackActor
#     ]
        
#     maddpg = MADDPG(actor_types)
#     maddpg.buffer = [{} for _ in range(10)]  # 假设buffer总长度10
#     maddpg.buffer[0] = {
#             'action_entity_id': np.array(['101', '102']),  # 当前批次选择两个实体
#             'actions': np.array([
#                 # 9个actor，每个actor有5个实体，每个实体5个参数
#                 # 仅第一个参数用于执行判断（actions[:, :, 0]）
                
#                     [ [0.8,0,0,0,0], [0.6,0,0,0,0], [0.9,0,0,0,0], [0.7,0,0,0,0], [0.2,0,0,0,0] ],  # Actor0
#                     [ [0.5,0,0,0,0], [0.3,0,0,0,0], [0.4,0,0,0,0], [0.1,0,0,0,0], [0.0,0,0,0,0] ],  # Actor1
#                     [ [0.9,0,0,0,0], [0.8,0,0,0,0], [0.7,0,0,0,0], [0.6,0,0,0,0], [0.5,0,0,0,0] ],  # Actor2
#                     [ [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0] ],  # Actor3（边界）
#                     [ [0.6,0,0,0,0], [0.6,0,0,0,0], [0.6,0,0,0,0], [0.6,0,0,0,0], [0.6,0,0,0,0] ],  # Actor4（全不执行）
#                     [ [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0] ],  # Actor5（全执行）
#                     [ [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0] ],  # Actor6
#                     [ [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0] ],  # Actor7（全不执行）
#                     [ [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0] ],  # Actor8（边界）
                
#             ]),
#             'rewards': np.zeros(9)  # 初始化为0
#         }
#         # 经验条目2 ---------------------------------------------------
#     maddpg.buffer[1] ={
#             'action_entity_id': np.array(['104', '105']),  # 不同实体选择
#             'actions': np.array([
                
#                     [ [0.6,0,0,0,0], [0.9,0,0,0,0], [0.5,0,0,0,0], [0.7,0,0,0,0], [0.1,0,0,0,0] ],  # Actor0
#                     [ [0.8,0,0,0,0], [0.7,0,0,0,0], [0.8,0,0,0,0], [0.2,0,0,0,0], [0.0,0,0,0,0] ],  # Actor1
#                     [ [0.9,0,0,0,0], [0.6,0,0,0,0], [0.4,0,0,0,0], [0.3,0,0,0,0], [0.5,0,0,0,0] ],  # Actor2
#                     [ [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0] ],  # Actor3
#                     [ [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0] ],  # Actor4（全执行）
#                     [ [0.5,0,0,0,0], [0.5,0,0,0,0], [0.5,0,0,0,0], [0.5,0,0,0,0], [0.5,0,0,0,0] ],  # Actor5（全不执行）
#                     [ [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0] ],  # Actor6
#                     [ [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0] ],  # Actor7
#                     [ [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0] ],  # Actor8
                
#             ]),
#             'rewards': np.zeros(9)
#         }

#     tester = CombatRewardCalculator()

#     tester.update_attack_buffer_reward(maddpg, attack_step=[0,1], attack_id= ['101','105'], attack_type=[0,4])
#     pdb.set_trace()


# def test3():
#     from maddpg_ import MADDPG
#     from actor import (AircraftTakeOffActor, ReturnToBaseActor, 
#                       WayPointMoveActor, MobilityActor, 
#                       AttackTargetActor, SensorControlActor,
#                       DeploySonobuoyActor,
#                       CancelAttackActor)

#     # 初始化 MADDPG 和测试数据 -------------------------------------------------
#     actor_types = [
#         AircraftTakeOffActor,
#         ReturnToBaseActor,
#         WayPointMoveActor,
#         MobilityActor,
#         AttackTargetActor,
#         SensorControlActor,
#         DeploySonobuoyActor,
#         CancelAttackActor
#     ]
#         # 创建 MADDPG 实例
#     maddpg = MADDPG(actor_types)
#     maddpg.buffer = [{} for _ in range(2)]  # 假设buffer总长度10
#     maddpg.buffer[0] = {
#             'action_entity_id': np.array(['101', '102']),  # 当前批次选择两个实体
#             'actions': np.array([
#                 # 9个actor，每个actor有5个实体，每个实体5个参数
#                 # 仅第一个参数用于执行判断（actions[:, :, 0]）
                
#                     [ [0.8,0,0,0,0], [0.6,0,0,0,0], [0.9,0,0,0,0], [0.7,0,0,0,0], [0.2,0,0,0,0] ],  # Actor0
#                     [ [0.5,0,0,0,0], [0.3,0,0,0,0], [0.4,0,0,0,0], [0.1,0,0,0,0], [0.0,0,0,0,0] ],  # Actor1
#                     [ [0.9,0,0,0,0], [0.8,0,0,0,0], [0.7,0,0,0,0], [0.6,0,0,0,0], [0.5,0,0,0,0] ],  # Actor2
#                     [ [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0] ],  # Actor3（边界）
#                     [ [0.6,0,0,0,0], [0.6,0,0,0,0], [0.6,0,0,0,0], [0.6,0,0,0,0], [0.6,0,0,0,0] ],  # Actor4（全不执行）
#                     [ [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0] ],  # Actor5（全执行）
#                     [ [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0] ],  # Actor6
#                     [ [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0] ],  # Actor7（全不执行）
#                     [ [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0] ],  # Actor8（边界）
                
#             ]),
#             'rewards': np.zeros(9)  # 初始化为0
#         }
#         # 经验条目2 ---------------------------------------------------
#     maddpg.buffer[1] ={
#             'action_entity_id': np.array(['104', '105']),  # 不同实体选择
#             'actions': np.array([
                
#                     [ [0.6,0,0,0,0], [0.9,0,0,0,0], [0.5,0,0,0,0], [0.7,0,0,0,0], [0.1,0,0,0,0] ],  # Actor0
#                     [ [0.8,0,0,0,0], [0.7,0,0,0,0], [0.8,0,0,0,0], [0.2,0,0,0,0], [0.0,0,0,0,0] ],  # Actor1
#                     [ [0.9,0,0,0,0], [0.6,0,0,0,0], [0.4,0,0,0,0], [0.3,0,0,0,0], [0.5,0,0,0,0] ],  # Actor2
#                     [ [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0] ],  # Actor3
#                     [ [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0], [0.8,0,0,0,0] ],  # Actor4（全执行）
#                     [ [0.5,0,0,0,0], [0.5,0,0,0,0], [0.5,0,0,0,0], [0.5,0,0,0,0], [0.5,0,0,0,0] ],  # Actor5（全不执行）
#                     [ [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0], [0.9,0,0,0,0] ],  # Actor6
#                     [ [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0], [0.0,0,0,0,0] ],  # Actor7
#                     [ [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0], [0.7,0,0,0,0] ],  # Actor8
                
#             ]),
#             'rewards': np.zeros(9)
#         }
    
#     # 注入测试用的 base_find_event_reward
#     tester = CombatRewardCalculator()
#     tester.base_find_event_reward = base_find_event_reward  # 假设类中有此属性
#     tester.update_find_buffer_reward(maddpg, entity_type=[0, 1])
    
    

    

# if __name__ == "__main__":
#     test3() 