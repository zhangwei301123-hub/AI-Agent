# maddpg_.py
import copy
import pdb
import pickle
import time
from http.client import responses
import math
from operator import index

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from numpy.ma.core import true_divide

from entity import *
from actor import *
from critic import Critic, TransformerBlock

from execute import reset, get_DetectionReports, get_AttackReports, get_Situaction4test, get_UseUpReports, pause, start, getEndSignal

from reward import CombatRewardCalculator
import os
import random

from log import Log


from utils import bootstrap_recon, evade_missiles, geo_distance, log_entity_actions
from latch_manager import *
from engagement_rules import ENGAGEMENT_RULES,MAX_RANGE,DIRECTION_OFFSET

from  execute import TIME_SPEED_MAP, SPEED_RATE

from actor_rules import *
from collections import defaultdict




# 改成常量，方便以后调
ATTACK_CD_SEC   = 1800          # 每对 attacker→target 的 CD
MAX_PAR_ATTACK  = 3            # 同一 target 允许并发的攻击者数
class AttackThrottle:
    """
    限制同一 target 同时被多少个 attacker 攻击；
    并给「同一对 attacker→target」加冷却（单位：帧）。
    """
    def __init__(self, cd_sec=180, max_parallel=3):
        self.cd_steps = TIME_SPEED_MAP[get_speed_rate()]
        self.max_parallel = max_parallel
        # {target_id: {attacker_id: 剩余CD}}
        # ★ 全局攻击表：{target_id: {attacker_id: 剩余CD}}
        self.table: Dict[str, Dict[str, int]] = {}

    # 每帧调用，所有 CD-1，≤0 时自动删除
    def tick(self):
        for tgt in list(self.table.keys()):
            for atk in list(self.table[tgt].keys()):
                self.table[tgt][atk] -= self.cd_steps
                if self.table[tgt][atk] <= 0:
                    self.table[tgt].pop(atk, None)
            if not self.table[tgt]:
                self.table.pop(tgt, None)

    # 判断某 attacker 本帧能否打这个 target
    def can_attack(self, attacker_id: str, target_id: str) -> bool:
        attackers = self.table.get(target_id, {})
        if attacker_id in attackers:               # 自己还在 CD 内
            self.register(attacker_id, target_id)
            return True
        return len(attackers) < self.max_parallel  # 并发数是否超限 T表示能攻击 有攻击实体的额度

    # 记录一次新的攻击
    def register(self, attacker_id: str, target_id: str):
        self.table.setdefault(target_id, {})[attacker_id] = self.cd_steps

ACTOR_PROBABILITY = 0.6

ACTOR_TYPES = [
    AircraftTakeOffActor,
    ReturnToBaseActor,
    WayPointMoveActor,
    MobilityActor,
    AttackTargetActor,
    SensorControlActor,
    DeploySonobuoyActor,
    CancelAttackActor
]


ATTACK_CAPABLE_AIRCRAFT_TYPES = {0, 1, 2, 4, 6, 7, 13} 

SUB2SUB_MAX_RANGE_NM = 14   # 潜艇对潜艇最大允许射程 (海里)

OUR_SIDE = 0
ENEMY_SIDE = 1

BUFFER_CAP = 50000


MISSILE_THREAT_DIST = 300000

def sample_direction():
    rd = random.random()
    if rd < 0.30:
        return 6
    elif rd < 0.30 + 0.25:
        return 4
    elif rd < 0.30 + 0.25 + 0.25:
        return 0
    else:
        # 剩下的方向 [1, 2, 3, 5, 7]
        return random.choice([1, 2, 3, 5, 7])
    
def apply_exploration(action_tensor: torch.Tensor, epsilon: float, actor_index: int):
    """
    根据 actor 类型结构，对动作张量加探索噪声。支持连续 p。
    """
    action_tensor = action_tensor.clone()
    num_entities, dim = action_tensor.shape

    for i in range(num_entities):
        rd = random.random()
        if rd > epsilon:
            continue

        action_tensor[i, 0] = torch.rand(1)


        # === 特定 actor 的离散字段探索 ===
        if actor_index == 2:  # 航路机动
            trd = sample_direction()
            action_tensor[i, 1] = torch.tensor(trd)  # 方向index
            action_tensor[i, 3] = torch.randint(2, 5, (1,))  # 速度
            action_tensor[i, 4] = torch.randint(1, 6, (1,))  # 高度

        elif actor_index == 3:  # 调整速度
            action_tensor[i, 1] = torch.randint(2, 5, (1,))   # 速度
            action_tensor[i, 2] = torch.randint(1, 6, (1,))   # 高度


        elif actor_index == 5:  # 传感器控制
            action_tensor[i, 1] = torch.rand(1) # 雷达
            action_tensor[i, 2] = torch.rand(1)  # 声呐
            action_tensor[i, 3] = torch.rand(1)  # 电战
            

        elif actor_index == 6:  # 浮标
            if action_tensor[i, 0] < 1e-6: #已经因为CD时间被置为0的浮标，则不进行探索
                continue
            else:
                action_tensor[i, 1] = torch.tensor(1 if random.random() < 0.8 else 0.0)
                action_tensor[i, 2] = torch.rand(1) #深/浅

    return action_tensor


class MADDPGAgent(nn.Module):
    """单个智能体管理类"""

    def __init__(self,
                 actor_type: nn.Module,
                 state_dim=256,
                 action_feat_dim=128,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=0.01,
                 min_lr=1e-6):
        super().__init__()
        # 超参数存储
        self.gamma = gamma
        self.tau = tau
        self.epsilon = 0.5  # 初始探索概率，可以调大调小
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995  # 每轮衰减

        # 初始化网络组件
        self.actor = actor_type()
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_feat_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.action_encoder = ActionEncoder(max_entity_len=MAX_ACTION_ENTITIES)

        # 优化器配置
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 学习率调度器
        self.critic_scheduler = ReduceLROnPlateau(
            self.critic_optimizer,
            mode='min',
            factor=0.9,
            patience=80,
            min_lr=min_lr
        )
        self.actor_scheduler = ReduceLROnPlateau(
            self.actor_optimizer,
            mode='min',
            factor=0.9,
            patience=100,
            min_lr=min_lr
        )
        # 超参数存储
        self.gamma = gamma
        self.tau = tau

    def decay_epsilon(self):
        """
        每轮训练后调用此函数，逐步衰减 epsilon（用于探索）
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class MADDPG(nn.Module):  
    """
    多智能体协同管理框架
    一共9个agent
    """
    def __init__(self, 
                 actor_types: list,
                 state_dim=256,
                 action_feat_dim=128,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=0.01,
                 batch_size=128
                 ):
        super().__init__() 
        # 初始化智能体群组
        self.agents = nn.ModuleList([
            MADDPGAgent(actor_type, state_dim, action_feat_dim, actor_lr, critic_lr, gamma, tau)
            for actor_type in actor_types
        ])
        
        # 全局状态编码器
        self.global_encoder = GlobalStateEncoder()
        self.self_encoder = SelfStateEncoder()
        
        # 经验回放缓冲区  
        # 按需求添加 modify
        self.buffer = ReplayBuffer(capacity=BUFFER_CAP)
        
        # 训练参数
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sever_our = {} # sever_frame:our_frame
        self.our_sever = {} # our_frame:sever_frame

        # self.device = torch.device("cpu")  # 默认 device
        # self.to(self.device)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def store_experience(self, states, actions, actions_mask, rewards, next_states, dones, step, action_entity_id, actions_executed):
        """存储经验到回放缓冲区"""
        # [batch, max_entity, dim]
        self.buffer.add({
            'states': states, # 一帧的所有单装信息  <class 'dict'>  encoded_data和mask  
            'actions': actions, # action_data [num_agents, num_entities->MAX_ACTION_ENTITIES, 5] 因为每一帧的num_entities可能不同，所以在保存为actions时（存入经验池时），需要对action_data进行padding，让num_entities达到MAX_ACTION_ENTITIES
            'actions_mask': actions_mask, # action_mask [num_agents, num_entities->MAX_ACTION_ENTITIES] 
            'rewards': rewards, # reward [num_agents]
            'next_states': next_states, # 下一帧的所有单装信息
            'dones': dones, # done
            'step':step,
            'action_entity_id': action_entity_id, # 当帧我方可执行命令实体id [num_entities]
            'actions_executed': actions_executed, # 我方可执行命令的实体 动作是否真正被执行 [num_entities, 8(actor_type_len)]
        })

    def _encode_global_and_self_state(self, raw_states, logger):
        """
        编码全局状态和自身状态。
        
        该函数通过编码器处理给定的原始状态数据，生成每个实体的全局特征和自身状态特征。
        它首先对传入的数据进行编码，然后通过注意力机制生成全局特征，并筛选出感兴趣的实体。
        最后，它将每个实体的自身状态特征与全局特征融合，并返回这些特征。
    
        参数:
        raw_states (dict): 包含'encoded_data'和'mask'的字典，分别表示编码后的数据和对应的掩码。
    
        返回:
        all_frame_all_our_entities (Tensor): [batch_size, max_entity_len, dim]，所有帧中所有实体的融合特征。
        global_feat (Tensor): [batch_size, dim]，全局特征。
        all_frame_all_enemies: [batch_size, max_targets, dim]
        """
        # pdb.set_trace()
        # 提取并编码输入数据
        encoded_data = raw_states['encoded_data'] # [batch_size, max_entity, dim]
        mask = raw_states['mask'] # [batch_size, max_entity]

        device = self.device if hasattr(self, 'device') else encoded_data.device
        encoded_data = encoded_data.to(device)
        mask = mask.to(device)

  
        attn_output, global_feat = self.global_encoder(encoded_data[:, :, :37], mask) # [batch_size, max_entity, dim] [batch_size, dim] 排除最后一个是否可操纵的特征
    
        # 初始化列表，用于存储所有帧中所有实体的融合特征
        all_frame_all_our_entities = []
        len_entities = []
        all_frame_all_enemies = []
        dummy_tensor = torch.cat([
            self.self_encoder(attn_output[0][0]),     # 一个随便的实体特征
            global_feat[0]                             # 一个随便的全局特征
        ], dim=0)  # [512]

        dummy_tensor2 = self.self_encoder(attn_output[0][0])     # 一个随便的实体特征 256

        # 新增坐标信息存储
        all_frame_our_coords = []
        all_frame_enemy_coords = []
        all_frame_enemy_velocitys = []

        for i in range(encoded_data.shape[0]):
            # 获取当前帧中需要关注的实体索引
            filtered_index = get_env_entity_ids(encoded_data[i].cpu().numpy(), mask[i].cpu().numpy(), target_side=OUR_SIDE, allowed_types=[0, 1, 2]) # 按需求添加 modify
            len_entities.append(len(filtered_index))
            single_frame_all_our_entities = []
        
            for index in filtered_index:
                # 编码当前实体的自身状态
                self_state = self.self_encoder(attn_output[i][index]) # [dim]
                # 将自身状态特征与全局特征融合
                fusion_result = torch.cat([self_state, global_feat[i]], dim=0) # [dim]
                single_frame_all_our_entities.append(fusion_result)
            if len(single_frame_all_our_entities) == 0:
                # single_frame_all_our_entities = [torch.zeros_like(fusion_result)]
                single_frame_all_our_entities = [torch.zeros_like(dummy_tensor)]
            # 将当前帧中所有我方可执行的实体的融合特征堆叠起来
            single_frame_all_our_entities = torch.stack(single_frame_all_our_entities) # [num_entities, dim(512)]
            all_frame_all_our_entities.append(single_frame_all_our_entities)

            # 获取当前帧中所有目标实体的索引
            enemies_index = get_env_entity_ids(encoded_data[i].cpu().numpy(), mask[i].cpu().numpy(), target_side=ENEMY_SIDE, allowed_types=[0, 1, 2, 4]) # 按需求添加 modify
            single_frame_all_enemies = []
            for index in enemies_index:
                single_frame_all_enemies.append(self.self_encoder(attn_output[i][index]))
            if len(single_frame_all_enemies) == 0:
                single_frame_all_enemies = [torch.zeros_like(dummy_tensor2)]
            single_frame_all_enemies = torch.stack(single_frame_all_enemies)
            all_frame_all_enemies.append(single_frame_all_enemies)


            # 添加位置编码
            our_coords = get_coordinate_from_encoded_data(encoded_data[i].cpu().numpy(),
                                                     mask[i].cpu().numpy(),
                                                     target_side=OUR_SIDE,
                                                     allowed_types=[0, 1, 2])
            enemy_coords = get_coordinate_from_encoded_data(encoded_data[i].cpu().numpy(),
                                                       mask[i].cpu().numpy(),
                                                       target_side=ENEMY_SIDE,
                                                       allowed_types=[0, 1, 2, 4])
            enemy_speed = get_velocity_from_encoded_data(encoded_data[i].cpu().numpy(),
                                           mask[i].cpu().numpy(),
                                           target_side=ENEMY_SIDE,
                                           allowed_types=[0,1,2,4])
            

            # 对齐到最大实体数并进行填充
            our_coords = self._pad_coordinates(our_coords, MAX_ACTION_ENTITIES)
            enemy_coords = self._pad_coordinates(enemy_coords, MAX_TARGETS)
            enemy_speed = self._pad_speeds(enemy_speed, MAX_TARGETS)
            all_frame_our_coords.append(torch.tensor(our_coords))
            all_frame_enemy_coords.append(torch.tensor(enemy_coords))
            all_frame_enemy_velocitys.append(torch.tensor(enemy_speed))

        # 将坐标信息转为Tensor
        our_coords_batch = torch.stack(all_frame_our_coords).to(device)
        enemy_coords_batch = torch.stack(all_frame_enemy_coords).to(device)
        enemy_velocitys_batch = torch.stack(all_frame_enemy_velocitys).to(device)



        # max_entity_len = max(len_entities)
        max_entity_len = MAX_ACTION_ENTITIES
        padded_batch = []
        padding_masks = []
        for i, entities in enumerate(all_frame_all_our_entities):
            # 填充实体维度
            num_pad = max_entity_len - entities.shape[0]
            padded = torch.cat([
                entities,
                torch.zeros((num_pad, entities.shape[1]), device=entities.device)
            ], dim=0) # [max_entity_len, dim]

            mask_ = torch.cat([
                torch.ones(entities.shape[0], dtype=torch.bool),
                torch.zeros(num_pad, dtype=torch.bool)
            ], dim=0).to(entities.device) # [max_entity_len]

            padded_batch.append(padded)
            padding_masks.append(mask_)
        
        max_enemy_len = MAX_TARGETS
        padded_batch_enemies = []
        padding_masks_enemies = []
        try:
            for i, enemy in enumerate(all_frame_all_enemies):
                # 填充实体维度
                num_pad = max_enemy_len - enemy.shape[0]
                padded = torch.cat([
                    enemy,
                    torch.zeros((num_pad, enemy.shape[1]), device=enemy.device)
                ], dim=0) # [max_enemy_len, dim]

                mask_ = torch.cat([
                    torch.ones(enemy.shape[0], dtype=torch.bool),
                    torch.zeros(num_pad, dtype=torch.bool)
                ], dim=0).to(enemy.device)

                padded_batch_enemies.append(padded)
                padding_masks_enemies.append(mask_)
        except Exception as e:
            pdb.set_trace()
            # print(e)
            logger.error(e)


        # 将所有帧的实体融合特征堆叠起来
        all_frame_all_our_entities = torch.stack(padded_batch) # [batch_size, max_entity_len, dim(512)]
        padding_masks = torch.stack(padding_masks) # [batch_size, max_entity_len]
        all_frame_all_enemies = torch.stack(padded_batch_enemies) # [batch_size, max_enemy_len, dim(256)]
        padding_masks_enemies = torch.stack(padding_masks_enemies) # [batch_size, max_enemy_len]

        return all_frame_all_our_entities, global_feat, len_entities, padding_masks, all_frame_all_enemies, padding_masks_enemies, our_coords_batch, enemy_coords_batch, enemy_velocitys_batch
        # 前者是actor的输入 后者是critic的一半输入 len_entities是每个batch中可执行动作的实体数
    
    def _pad_coordinates(self, coords, max_len):
        """将坐标填充到固定长度"""
        pad_len = max_len - len(coords)
        if pad_len > 0 and pad_len != max_len:
            return np.pad(coords, ((0, pad_len), (0, 0)), mode='constant')
        elif pad_len == max_len:
            return np.pad([(0, 0)], ((0, pad_len - 1), (0, 0)), mode='constant')


    def _pad_speeds(self, speeds, max_len):
        """
        将速度列表/数组填充或截断为固定长度。

        参数
        ----
        speeds  : List[float] | np.ndarray   # [N]
        max_len : int                       # 目标长度

        返回
        ----
        np.ndarray                           # [max_len]
        """
        speeds = np.asarray(speeds, dtype=np.float32)      # 转成 1D 数组
        cur_len = len(speeds)


        # 情况 1：长度 >= max_len —— 直接截断
        if cur_len >= max_len:
            return speeds[:max_len]

        # 情况 2：长度 < max_len —— 末尾补 0
        pad_len = max_len - cur_len
        if cur_len == 0:
            # 原列表为空，直接返回全 0
            return np.zeros(max_len, dtype=np.float32)
        else:
            return np.pad(speeds, (0, pad_len), mode='constant')


    def prepare_state(self, states):
        batch_size = self.batch_size
        max_entity = MAX_ENTITIES
        dim = states[0]['encoded_data'][:, :37].shape[1]

        encoded_data_batch = torch.zeros((batch_size, max_entity, dim))  # [batch_size, max_entity, dim]
        mask_batch = torch.zeros((batch_size, max_entity), dtype=torch.bool)  # [batch_size, max_entity]

        for i in range(batch_size):
            encoded_data_batch[i] = states[i]['encoded_data'][:, :37]
            mask_batch[i] = states[i]['mask']  

        return {'encoded_data':encoded_data_batch, 'mask': mask_batch}

    def update(self, logger):


        """执行训练更新"""
        agent_losses = [
            {"critic": [], "actor": []} 
            for _ in range(len(ACTOR_TYPES))
        ]

        agent_qs = [
            {"current": [], "target": []}  # 与 agents 一一对应
            for _ in range(len(ACTOR_TYPES))
        ]

        device = self.device
        # 采样批量数据
        batch = self.buffer.sample(self.batch_size)
        states = batch['states']
        states = self.prepare_state(states)
        all_agent_actions = torch.tensor(batch['actions']).to(device) # [batch_size, num_agents, num_entities->MAX_ACTION_ENTITIES, 5]
        all_agent_actions_mask = torch.tensor(batch['actions_mask']).to(device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(device)
        next_states = batch['next_states']
        next_states = self.prepare_state(next_states)
        dones = torch.tensor(batch['dones'], dtype=torch.float32).to(device) # 这里要改成0/1
        
        # 全局状态编码
        actors_input, critics_half_input, _, actors_input_masks, attack_target_f, attack_target_masks ,our_coords, enemy_coords, enemy_speeds= self._encode_global_and_self_state(states, logger)

        next_actors_input, next_critics_half_input, _, next_actors_input_masks, next_attack_target_f, next_attack_target_masks ,next_our_coords, next_enemy_coords ,next_enemy_speeds= self._encode_global_and_self_state(next_states, logger)
        for qs in agent_qs:
            qs["current"].clear()
            qs["target"].clear()
        # 并行更新所有智能体
        for idx, agent in enumerate(self.agents):

            device  = self.device
            # 编码动作特征
            action_feats = agent.action_encoder(all_agent_actions[:, idx].to(device), all_agent_actions_mask[:, idx].to(device)) # [batch_size, action_feat_dim(128)]

            if isinstance(agent.actor, AttackTargetActor):
                # 注意：attck actor的输入不只是next_actors_input 还有 target_features [batch_size, max_targets, 256] (256是SelfStateEncoder的输出)
                
                # Critic更新
                agent.critic_optimizer.zero_grad()
                with torch.no_grad():

                    next_output = agent.actor_target(
                    next_actors_input,
                    target_features=next_attack_target_f,
                    target_mask=next_attack_target_masks,
                    self_mask=next_actors_input_masks,
                    target_coords=next_enemy_coords,
                    self_coords=next_our_coords,  # 新增坐标参数
                    target_speeds = next_enemy_speeds
                )
                    next_actions = actor_output_to_action(next_output, max_entity_len=next_actors_input.shape[1]) # [batch_size, max_entity_len, 5]
                    next_action_feats = agent.action_encoder(next_actions, next_actors_input_masks) # [batch_size, action_feat_dim(128)]
                    
                    target_q = agent.critic_target(next_critics_half_input, next_action_feats).squeeze(1)
                    target_q = rewards[:, idx] + (1 - dones) * agent.gamma * target_q
                    
                current_q = agent.critic(critics_half_input, action_feats).squeeze(1)
                critic_loss = F.mse_loss(current_q, target_q)
                critic_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
                agent.critic_optimizer.step()
                
                agent_losses[idx]["critic"].append(critic_loss.item())

                # Actor更新
                agent.actor_optimizer.zero_grad()
                


                action_outputs = agent.actor(
                    actors_input, 
                    attack_target_f, 
                    target_mask=attack_target_masks,   
                    self_mask=actors_input_masks,
                    target_coords=enemy_coords,
                    self_coords=our_coords,  # 新增坐标参数
                    target_speeds = enemy_speeds

                    )
                actions = actor_output_to_action(action_outputs, max_entity_len=actors_input.shape[1])
                action_feats = agent.action_encoder(actions, actors_input_masks)
                actor_loss = -agent.critic(critics_half_input, action_feats).mean()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                agent.actor_optimizer.step()

                agent_losses[idx]["actor"].append(actor_loss.item())
                agent_qs[idx]["current"].append(current_q.mean().item())
                agent_qs[idx]["target"].append(target_q.mean().item())
            

            else:
                
                # Critic更新
                agent.critic_optimizer.zero_grad()
                with torch.no_grad():
                    next_output = agent.actor_target(next_actors_input)
                    next_actions = actor_output_to_action(next_output, max_entity_len=next_actors_input.shape[1]) # [batch_size, max_entity_len, 5]
                    next_action_feats = agent.action_encoder(next_actions, next_actors_input_masks) # [batch_size, action_feat_dim(128)]
                    target_q = agent.critic_target(next_critics_half_input, next_action_feats).squeeze(1)
                    target_q = rewards[:, idx] + (1 - dones) * agent.gamma * target_q
                    
                current_q = agent.critic(critics_half_input, action_feats).squeeze(1)
                critic_loss = F.mse_loss(current_q, target_q)
                if math.isnan(critic_loss.item()):
                    print('nan')
                critic_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
                agent.critic_optimizer.step()

                agent_losses[idx]["critic"].append(critic_loss.item())
                

                # Actor更新
                agent.actor_optimizer.zero_grad()
                action_outputs = agent.actor(actors_input)
                actions = actor_output_to_action(action_outputs, max_entity_len=actors_input.shape[1])
                action_feats = agent.action_encoder(actions, actors_input_masks)
                actor_loss = -agent.critic(critics_half_input, action_feats).mean()
                if math.isnan(actor_loss.item()):
                    print('nan')
                actor_loss.backward(retain_graph=(idx != len(self.agents) - 1))
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                agent.actor_optimizer.step()
                
                agent_losses[idx]["actor"].append(actor_loss.item())
                agent_qs[idx]["current"].append(current_q.mean().item())
                agent_qs[idx]["target"].append(target_q.mean().item())

            # 软更新目标网络
            self._soft_update(agent.actor, agent.actor_target, agent.tau)
            self._soft_update(agent.critic, agent.critic_target, agent.tau)

            
        ######################## 在方法最后输出损失 ########################
        # print("=" * 60)
        logger.info("=" * 60)
        # print("当前更新步骤的损失：")
        logger.info("当前更新步骤的损失：")
        # for idx, losses in enumerate(self.agent_losses):
        for idx, (losses, qs) in enumerate(zip(agent_losses, agent_qs)):
            a_name = ACTOR_TYPES[idx].__name__
            critic_l = losses["critic"][0]
            actor_l  = losses["actor"][0]
            cur_q    = qs["current"][0]
            tgt_q    = qs["target"][0]
            logger.info(
                f"Agent {a_name}: "
                f"CriticLoss={critic_l:.4f}, ActorLoss={actor_l:.4f}, "
                f"Q={cur_q:.4f}, TargetQ={tgt_q:.4f}"
            )
            # 获取最新记录的Critic和Actor损失
            latest_critic_loss = losses["critic"][0]
            latest_actor_loss = losses["actor"][0]
            if math.isnan(latest_critic_loss) or math.isnan(latest_actor_loss):
                logger.warning("Nan Loss!")
            # print("Agent {}: Critic Loss = {:.4f}, Actor Loss = {:.4f}".format(ACTOR_TYPES[idx].__name__, latest_critic_loss, latest_actor_loss))
            # logger.info("Agent {}: Critic Loss = {:.4f}, Actor Loss = {:.4f}".format(ACTOR_TYPES[idx].__name__, latest_critic_loss, latest_actor_loss))

        # # ==================== 新增 epsilon 衰减 ====================
        # for agent in self.agents:
        #     if hasattr(agent, "epsilon"):
        #         agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)


    def _soft_update(self, local_model, target_model, tau):
        """执行目标网络软更新"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class ReplayBuffer:
    def __init__(self, capacity, max_episode_length = None):
        self.capacity = capacity          # 总步数限制
        if max_episode_length   == None:
            self.max_episode_length   = int(self.capacity / 2.5)
        else:
            self.max_episode_length   = max_episode_length     
        self.buffer_groups = []           # 每组是一个 episode：list[experience]
        self.total_steps = 0              # 当前已存样本条数

    def start_new_episode(self):
        """开始一个新 episode（不管上一轮是否空）"""
        self.buffer_groups.append([])

    def add(self, experience):
        if not self.buffer_groups:
            self.start_new_episode()
        current_episode = self.buffer_groups[-1]
        # 如果当前组已满，则先移除最前面的经验
        if len(current_episode) == self.max_episode_length:
            current_episode.pop(0)
            self.total_steps -= 1  # 因为删了一条

        # 添加到当前组
        self.buffer_groups[-1].append(experience)
        self.total_steps += 1

        while self.total_steps >= int(self.capacity * 0.9):
            for i, group in enumerate(self.buffer_groups):
                if group:  # 找到非空组
                    self.total_steps -= len(group)
                    self.buffer_groups[i] = []  # 清空这个组
                    break
                else: # 找到空组
                    continue

    def _has_valid_mask(self, exp):
        """
        判断 exp["states"]["mask"] 是否至少包含一个 True。
        若缺失字段，则默认有效（兼容旧格式）。
        """
        try:
            mask = np.asarray(exp["states"]["mask"], dtype=bool)
            next_mask = np.asarray(exp["next_states"]["mask"], dtype=bool)
            return mask.any() and next_mask.any()
        except Exception:
            # states 或 mask 字段不存在 → 视为有效
            return True

    def sample(self, batch_size, recent_ratio=0.7, min_old_samples=1):
        """从经验池采样，优先新数据，但保证旧数据有一定比例"""
        flat_buffer = [exp for group in self.buffer_groups if group for exp in group]
        if len(flat_buffer) < batch_size:
            raise ValueError(f"样本不足，总可采样数为 {len(flat_buffer)}")

        # 划分新旧数据
        recent_start = int(len(flat_buffer) * (1 - recent_ratio))
        recent_buffer = flat_buffer[recent_start:]  # 新数据（后 recent_ratio%）
        old_buffer = flat_buffer[:recent_start]     # 旧数据（前 1-recent_ratio%）

        # 计算新数据和旧数据的采样数量
        num_recent = min(batch_size - min_old_samples, len(recent_buffer))
        num_old = batch_size - num_recent

        # 从新数据和旧数据中分别采样
        sampled_recent = random.sample(recent_buffer, num_recent) if num_recent > 0 else []
        sampled_old = random.sample(old_buffer, num_old) if num_old > 0 else []


        # 合并采样结果
        sampled = sampled_recent + sampled_old


        # 2) 校验与替换
        # 建立一个“备选池”，
        pool = [e for e in flat_buffer]

        i = 0
        while i < len(sampled):
            if self._has_valid_mask(sampled[i]):
                i += 1
                continue   # 该样本有效
            else:
                sampled.pop(sampled[i]) # → 无效，弹出
            while True:
                new_sampled = random.sample(pool, 1)[0]
                if self._has_valid_mask(new_sampled):
                    i += 1
                    sampled.append(new_sampled)
                    break
                else:
                    continue


        assert len(sampled) == batch_size


        return {
            "states": np.array([exp["states"] for exp in sampled]),
            "actions": np.array([exp["actions"] for exp in sampled]),
            "actions_mask": np.array([exp["actions_mask"] for exp in sampled]),
            "rewards": np.array([exp["rewards"] for exp in sampled]),
            "next_states": np.array([exp["next_states"] for exp in sampled]),
            "dones": np.array([exp["dones"] for exp in sampled]),
            "step": np.array([exp["step"] for exp in sampled]),
            "action_entity_id": np.array([exp["action_entity_id"] for exp in sampled]),
            "actions_executed": np.array([exp["actions_executed"] for exp in sampled]),
        }

    def get_latest_episode(self):
        """返回最后一个 episode 的全部经验"""
        return self.buffer_groups[-1] if self.buffer_groups else []

    def __len__(self):
        return self.total_steps

    def num_groups(self):
        return len(self.buffer_groups)
    


class SimulatedEnv:
    def __init__(self, max_entities=MAX_ENTITIES, max_steps=int(BUFFER_CAP/2.5), state_dir="states"):
        self.raw_data = None
        self.entity_encoder = EntityEncoder(max_entities=max_entities)

        self.time_step = 1  # to cut
        self.state_dir = state_dir  

        self.max_steps = max_steps

        self.current_step = 0

        self.current_state = None
        self.current_mask = None

        self.max_entities = max_entities   # 最大实体数
        self.N = 50  # 帧延迟后

        self.rewarder = CombatRewardCalculator(actor_probability=ACTOR_PROBABILITY)
        
        # 初始化实体 ID: 动作 - 帧数链路
        self.entity_action_frame_link = {}

        # 初始化飞机在空字典 类型：数量
        self.aircraft_unitcategory_num = defaultdict(int)

        self._prev_threatened = set()
        self.attck_center_point = None
        self.attck_point_list = []
        self.pending_wp = {} 
        
        self.latch = CommandLatch0()
        # self.tick_thread = None

        self.attack_throttle = AttackThrottle(cd_sec=ATTACK_CD_SEC, max_parallel=MAX_PAR_ATTACK)


    def reset_entity_action_frame_link(self):
        self.entity_action_frame_link = {}


    def reset(self, logger):
        """
        重置+读取场景的 json化的 所有单装信息 
        返回 {encoded_data，mask}

        """ 

        self.current_step = 0
        self.entity_action_frame_link = {}

        self.raw_data = execute.reset4test(logger)
        self._prev_threatened.clear()
        self.pending_wp = {}

        self.aircraft_unitcategory_num = defaultdict(int)

        self.latch = CommandLatch0()

        # self.stop()
        # self.latch = CommandLatch()
        # self.tick_thread = threading.Thread(target=self.latch.tick, daemon=True)
        # self.tick_thread.start()


        self.current_state, self.current_mask = self.entity_encoder.encode(self.raw_data)

        return {
            'encoded_data': torch.FloatTensor(self.current_state),
            'mask': torch.BoolTensor(self.current_mask)
        }

    def pause(self):
        # if self.latch:
        #     self.latch.pause()
        pass

    def resume(self):
        # if self.latch:
        #     self.latch.resume()
        pass

    def stop(self):
        # if self.latch:
        #     self.latch.stop()
        # if self.tick_thread:
        #     self.tick_thread.join()
        pass

    def get_attack_area(self):
        attack_area = execute.get_attack_area()
        return attack_area

    def step(self, actions, now_state, logger):
        """
        执行动作并返回环境反馈
        param 
        actions: 所有智能体的动作列表 
        now_state : 一个单装信息的所有entity
        :return: (next_state, reward, done, {})
        """
        # if not self.latch or not self.latch.running.is_set():
        #     return False
        mission_dicts = execute.get_mission_dicts() # {missionId: areapoints}
        
        encoded_state = now_state['encoded_data'].unsqueeze(0)
        state_mask = now_state['mask'].unsqueeze(0)
        #根据来袭的导弹进行机动
        actions, urgent_flags = evade_missiles(actions, encoded_state, state_mask, self.raw_data, MISSILE_THREAT_DIST)
        number_urgent = len(urgent_flags)

        #我方全部可以执行动作的实体的索引
        our_entity_indices = get_env_entity_ids(now_state['encoded_data'].cpu().numpy(), now_state['mask'].cpu().numpy(), target_side=OUR_SIDE, allowed_types=[0, 1, 2])
        our_can_attack_ids = is_can_attack(now_state['encoded_data'].cpu().numpy(), now_state['mask'].cpu().numpy(), target_side=OUR_SIDE, allowed_types=[0, 1, 2])
        #我方全部实体的ID
        our_entity_ids = [ entity.get('mdlID') for i ,entity in enumerate(self.raw_data) if i in our_entity_indices]
        #我方实体的坐标
        our_entity_coordinate = get_coordinate_from_encoded_data(now_state['encoded_data'].cpu().numpy(), now_state['mask'].cpu().numpy(), target_side=OUR_SIDE, allowed_types=[0, 1, 2])

        #敌方全部实体的索引
        enemy_indices = get_env_entity_ids(now_state['encoded_data'].cpu().numpy(), now_state['mask'].cpu().numpy(), target_side=ENEMY_SIDE, allowed_types=[0, 1, 2, 4])
        #敌方全部实体的ID 
        enemy_ids = [ entity.get('mdlID') for i ,entity in enumerate(self.raw_data) if i in enemy_indices]
        enemy_coordinate = get_coordinate_from_encoded_data(now_state['encoded_data'].cpu().numpy(), now_state['mask'].cpu().numpy(), target_side=ENEMY_SIDE, allowed_types=[0, 1, 2, 4])
        actions_dict = {}

        get_aircraft_on_air_num(our_entity_indices, now_state, self.aircraft_unitcategory_num)
        not_aim_count = 0
        out_of_range_count = 0
        return2b_count = 0
        patrol_count = 0
        no_enemy_count = 0
        deploy_count = 0
        takeoff_count = 0
        for i, (entity_idx, entity_id) in enumerate(zip(our_entity_indices, our_entity_ids)): # entity_idx是这个实体在encodeddata中的索引位置 entity_id是这个实体的ID
            # ============== 先做“返航”硬规则 ==============
            if handle_return_to_base_rule(actions, now_state, i, entity_idx, entity_id, actions_dict):
                return2b_count += 1
                continue

            # ============== 起飞 限制起飞数量 ==============
            takeoff_count += handle_take_off_num_rule(actions, now_state, i, entity_idx, entity_id, self.aircraft_unitcategory_num, ACTOR_PROBABILITY)

            # 提取 actions 中对应 entity_id 的 8*5 矩阵 [8, 5]
            list_actions_for_entity = actions[:, i, :].tolist()  # 提取所有智能体对当前实体的动作
            
            # ==============航路机动索引->经纬度==============
            handle_mobile(list_actions_for_entity, actions, i, entity_idx, our_entity_coordinate, self.raw_data, now_state,
                          mission_dicts, urgent_flags, entity_id, patrol_count)
            
            # if now_state['encoded_data'][entity_idx][13] == 0 and now_state['encoded_data'][entity_idx][33] in [8, 9, 13, 14, 17, 23, 24, 25 ]\
            #         and not urgent_flags.get((entity_id, 2), False):
            #     handle_waypoint_move_patrol(list_actions_for_entity, i, entity_id, our_entity_coordinate, actions, self.attck_point_list)
            #     patrol_count += 1
            # else:
            #     handle_waypoint_move(list_actions_for_entity, i, entity_id, our_entity_coordinate, actions)

            # ==============部署浮标规则==============
            altitude = float(now_state['encoded_data'][entity_idx][5])
            deploy_count += handle_deploy(list_actions_for_entity, actions, i, entity_idx, our_entity_coordinate, self.raw_data, 
                                          mission_dicts, altitude, ACTOR_PROBABILITY)
                
            # ==============攻击判定和执行==============
            urgent_flag = urgent_flags.get((entity_id, 2), False)
            not_aim_count, out_of_range_count, no_enemy_count = handle_attack_decision(
                actions, now_state, i, entity_idx, entity_id, our_entity_indices, our_can_attack_ids,
                enemy_ids, enemy_indices, enemy_coordinate, actions_dict, not_aim_count, 
                out_of_range_count, ACTOR_PROBABILITY, list_actions_for_entity, self.raw_data,
                our_entity_coordinate, SUB2SUB_MAX_RANGE_NM, logger, urgent_flag, no_enemy_count)
            actions_dict[entity_id] = list_actions_for_entity  # 我方实体id 执行的动作


        # ← 在这里加一段“只留首次”的过滤
        filtered_urgent = {}
        for (ent_id, actor_idx), v in urgent_flags.items():
            if ent_id not in self._prev_threatened:
                filtered_urgent[(ent_id, actor_idx)] = True
                self._prev_threatened.add(ent_id)

        for actor_idx in range(actions.shape[0]):
            for ent_pos, ent_id in enumerate(our_entity_ids):
                flag = filtered_urgent.get((ent_id, actor_idx), False)
                if (not flag) and self.latch.is_locked(ent_id, actor_idx):
                    actions[actor_idx, ent_pos, 0] = 0.01
                    actions_dict[ent_id][actor_idx][0] = 0.01


        self.attack_throttle.tick()              # 1) 先全体 -1
        
        for atk_entity_id, acts in actions_dict.items():
            # index 4 约定为 AttackTargetActor，概率在 pos0，目标索引在 pos1
            atk_prob, target_id = acts[4][0], str(acts[4][1])
            if atk_prob < ACTOR_PROBABILITY and atk_prob != 0.0123:
                continue
            if target_id not in enemy_ids:
                continue
            if not self.attack_throttle.can_attack(atk_entity_id, target_id): # 超额不能攻击了
                acts[4][0] = 0.01                             # 直接禁掉攻击
                if atk_prob == 0.0123:                        # 追击还要关机动
                    acts[2][0] = 0.01
                continue
            self.attack_throttle.register(atk_entity_id, target_id)
            
        # for atk_entity_id, acts in actions_dict.items():
        #     # index 4 约定为 AttackTargetActor，概率在 pos0，目标索引在 pos1
        #     atk_prob, target_id = acts[4][0], str(acts[4][1])
        #     if atk_prob == 0.0123 or atk_prob > ACTOR_PROBABILITY:  # 如果是追击或者直接攻击
        #         if not self.attack_throttle.can_attack(atk_entity_id, target_id):  #
        #             if atk_prob == 0.0123:
        #                 acts[2][0] = 0.01
        #             continue
        #         else:
        #             self.attack_throttle.register(atk_entity_id, target_id)
                
        execute_results, rewards = execute.execute_actions(actions_dict, enemy_ids, probablity = ACTOR_PROBABILITY, logger=logger)
        # 记录实体执行的动作
        logger.info(" ")
        logger.info(self.current_step)
        log_entity_actions(execute_results, ACTOR_TYPES, logger)

        ship_and_sub_ids = [entity.get('mdlID') for i, entity in enumerate(self.raw_data) if i in our_entity_indices
                            and entity.get('mdlType', '').upper() in ['SHIP', 'SUBMARINE']]
        
        # ---------- 依据执行结果写回冷却 ----------
        for ent_id, performed in execute_results.items(): # ent_id: mdID , performed: 执行结果
            for a_idx, is_perf in enumerate(performed): # a_idx:  actor_idx     is_perf 真正的执行结果
                # 没有执行成功 就不记录cd
                if not is_perf: 
                    continue
                # 下面是执行成功的
                # 1.计算本身应使用的 CD
                cd = int(DEFAULT_COOLDOWN.get(a_idx, 0))
                if a_idx == 2:   
                    des_lon = actions_dict[ent_id][2][1] 
                    if not isinstance(des_lon, (list, tuple)):
                        des_lon = [des_lon]

                    if len(des_lon) > 1:
                        entry  = actions_dict[ent_id][2]
                        lon_tuple = entry[1]  
                        last_lon  = lon_tuple[-1]  
                        lat_tuple = entry[2]
                        last_lat  = lat_tuple[-1]
                        tgt_coord = (last_lon, last_lat)
                        self.pending_wp[ent_id] = tgt_coord 
                        self.latch.lock(ent_id, a_idx, cooldown=int(long_WayPointMove))

                    else:
                        if ent_id in ship_and_sub_ids:
                            self.latch.lock(ent_id, a_idx, cooldown=900)
                        else:
                            self.latch.lock(ent_id, a_idx, cooldown=cd)

                # 如果这是一次“紧急机动”(urgent flag)，或者别的特例，就重写 cd=25的机动
                flag = urgent_flags.get((ent_id, a_idx), False) 
                if a_idx == 2 and flag:
                    cd = int(25)
                # 2.先给本 actor 加锁
                if a_idx != 2: # 上面的大判断已经处理了大航路机动
                    self.latch.lock(ent_id, a_idx, cooldown=cd)
                # 3.如果是“航路机动”成功，再把（高度速度）机动动作 一并锁上，用它自己默认 CD
                if a_idx == 2:                                   # WayPointMove 成功
                    cd_mob = DEFAULT_COOLDOWN.get(3, 0)         # MobilityActor 的 CD
                    self.latch.lock(ent_id, 3, cooldown=cd_mob)  # 给索引 3 加锁

        # ---------- 提前解锁：航点到达 ----------
        to_del = []
        for ent_id, wp in self.pending_wp.items():
            # ① 仅当 WayPointMoveActor 仍被锁
            # 要么这个实体从来没进过大机动的 pending 列表
            # 要么它的 CD 正常走完了，已经不需要再“到点”去解锁
            if not self.latch.is_locked(ent_id, 2):
                to_del.append(ent_id)
                continue

            cur_coord = get_coord(now_state['encoded_data'].cpu().numpy(), now_state['mask'].cpu().numpy(), self.raw_data, ent_id,our_side=OUR_SIDE)
            # ② 判断距离是否足够近（阈值自定，单位米）
            if cur_coord is None or geo_distance(cur_coord[0],cur_coord[1],wp[0],wp[1]) < 300:   # 例如 300 m
                self.latch.table[ent_id][2] = 0     # 立刻清零
                to_del.append(ent_id)

        # 清理已完成航点
        for k in to_del:
            self.pending_wp.pop(k, None)
            
        # ---------- 所有计时器 -1 ---------- 
        # key 是mdID Value是 每个actor的冷却时间
        self.latch.tick()


        for i in range(not_aim_count + out_of_range_count):
            rewards += [0,0,0,1,1,0,0,0]
        for i in range (return2b_count):
            rewards += [0,1,0,0,0,0,0,0]
        for _ in range(patrol_count + number_urgent):
            rewards += [0,0,1,0,0,0,0,0]
        for _ in range(no_enemy_count):
            rewards += [0,0,0,0,1,0,0,0]
        for _ in range(deploy_count):
            rewards += [0,0,0,0,0,0,1,0]

        # execute.start(logger)


        # 更新ID: 动作 - 帧数链路
        self.entity_action_frame_link = update_entity_action_frame_link(self.entity_action_frame_link, execute_results, self.current_step)
        
        # 读取下一步状态
        self.raw_data = self.read_next_state(logger)
        next_state, next_mask = self.entity_encoder.encode(self.raw_data)
        next_state =  {'encoded_data': torch.FloatTensor(next_state), 'mask': torch.BoolTensor(next_mask)}
        self.current_step += 1

        # done = 1 if ((self.current_step >= self.max_steps) or self.read_done() == 1) else 0
        done = 1 if (self.current_step >= self.max_steps) else 0
        return next_state, rewards, done, our_entity_ids, list(execute_results.values())
    
    def read_done(self):
        response = execute.getEndSignal()
        return response.code

    def read_next_state(self, logger):
        """从 JSON 文件中读取下一个状态"""
        return execute.get_Situaction4test(logger)
        
    
class SignalListener:
    def __init__(self, env: SimulatedEnv, logger):
        self.env = env
        self.logger = logger
        self.state = "idle"
        self.control_event = threading.Event()
        self.simulating = False

    def handle_signal(self, sig):
        if sig == "start":
            if self.state in ["idle", "stopped"]:
                self.env.reset(self.logger)
                self.simulating = True
                self.control_event.set()
            elif self.state == "paused":
                self.env.resume()
                # self.simulating = True
                self.control_event.set()
            self.state = "running"

        elif sig == "pause":
            self.env.pause()
            self.control_event.clear()
            self.state = "paused"

        elif sig == "stop":
            self.env.stop()
            self.control_event.clear()
            self.simulating = False
            self.state = "stopped"

        elif sig == "restart":
            self.env.stop()
            self.env.reset(self.logger)
            self.simulating = True
            self.control_event.set()
            self.state = "running"

        elif sig == "running": # 不做处理
            self.state = "running"

        # self.logger.info(f"[SIGNAL] Received signal: {sig} → new state: {self.state}")

    def simulate_signals(self):
        # signals = ["start", "pause", "start", "stop", "restart", "stop"]
        # for sig in signals:
        #     time.sleep(5)
        #     self.handle_signal(sig)

        while True:
            time.sleep(1.0) # 这样可以限制一秒决策一次
            try:
                speed = execute.get_speed()
                execute.set_speed_rate(speed)
                signal = execute.get_control_signal()  # 阻塞或轮询获取
                self.handle_signal(signal)
                print(f'speed: {speed}, signal: {signal}')
            except Exception as e:
                print(e)
                time.sleep(1)
        




def update_entity_action_frame_link(entity_action_frame_link, execute_result, step):
    for entity_id, action_is_performs in execute_result.items():
        # 找出执行的动作 ID
        performed_action_ids = [i for i, is_performed in enumerate(action_is_performs) if is_performed]

        # 如果实体 ID 不在结果字典中，初始化一个空列表
        if entity_id not in entity_action_frame_link:
            entity_action_frame_link[entity_id] = []

        # 添加当前步骤的执行动作 ID 列表
        entity_action_frame_link[entity_id].append({step: performed_action_ids})

    return entity_action_frame_link



import pickle
def save_checkpoint(maddpg, episode, iter, filename, logger):
    """保存模型和其他数据（分开存储）"""
    model_checkpoint = {
        'model': maddpg.state_dict(),
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_checkpoint, f)
    logger.info(f"模型已保存到 {filename}")

    others_filename = f"{os.path.dirname(filename)}/maddpg_others.pt"  # 同目录下
    others_checkpoint = {
        'buffer': maddpg.buffer,
        'episode': episode,
        'iter': iter,
    }
    with open(others_filename, 'wb') as f:
        pickle.dump(others_checkpoint, f)
    logger.info(f"其他数据已保存到 {others_filename}")

def save_checkpoint2(maddpg, episode, iter, filename, logger):
    """保存模型和其他数据（分开存储）"""
    model_checkpoint = {
        'model': maddpg.state_dict(),
    }
    torch.save(model_checkpoint, filename)
    logger.info(f"模型已保存到 {filename}")

    others_filename = f"{os.path.dirname(filename)}/maddpg_others.pt"  # 同目录下
    others_checkpoint = {
        'buffer': maddpg.buffer,
        'episode': episode,
        'iter': iter,
    }
    torch.save(others_checkpoint, others_filename)
    logger.info(f"其他数据已保存到 {others_filename}")

def load_checkpoint(maddpg, filename, logger):
    """加载模型和其他数据（从固定文件 maddpg_others.pt 读取）"""
    with open(filename, 'rb') as f:
        model_checkpoint = pickle.load(f)
    maddpg.load_state_dict(model_checkpoint['model'])
    logger.info(f"模型已从 {filename} 加载")

    others_filename = f"{os.path.dirname(filename)}/maddpg_others.pt"
    with open(others_filename, 'rb') as f:
        others_checkpoint = pickle.load(f)
    maddpg.buffer = others_checkpoint['buffer']
    maddpg.buffer.capacity = BUFFER_CAP  # 恢复 buffer 容量
    logger.info(f"其他数据已从 {others_filename} 加载")

    return others_checkpoint.get("episode", 0), others_checkpoint.get("iter", 0)

def load_checkpoint2(maddpg, filename, logger):
    """加载模型和其他数据（从固定文件 maddpg_others.pt 读取）"""
    if torch.cuda.is_available():
        model_checkpoint = torch.load(filename)
    else:
        model_checkpoint = torch.load(filename, map_location='cpu')
    maddpg.load_state_dict(model_checkpoint['model'])
    logger.info(f"模型已从 {filename} 加载")

    others_filename = f"{os.path.dirname(filename)}/maddpg_others.pt"
    if torch.cuda.is_available():
        others_checkpoint = torch.load(others_filename)
    else:
        others_checkpoint = torch.load(others_filename, map_location='cpu')

    maddpg.buffer = others_checkpoint['buffer']
    maddpg.buffer.capacity = BUFFER_CAP  # 恢复 buffer 容量
    logger.info(f"其他数据已从 {others_filename} 加载")

    return others_checkpoint.get("episode", 0), others_checkpoint.get("iter", 0)

import execute


def main_loop(env, controller, logger, maddpg, device):


    # ----------- 重置环境 ------------
    maddpg.buffer.start_new_episode()
    for ag in maddpg.agents:
        ag.decay_epsilon()
    env.reset_entity_action_frame_link()
    state = env.reset(logger)
    maddpg.sever_our = {}
    maddpg.our_sever = {}

    done = False
    # 获取交战区域  （左上 左下 右上角 右下角）
    # env.attck_point_list = env.get_attack_area()
    # env.attck_center_point = get_area_center(env.attck_point_list)

    step_count = 0
    while controller.simulating:
        controller.control_event.wait()  # 阻塞直到 start/resume
        if not controller.simulating:
            break

        # ------------重置动作---------------
        all_actions = [[] for _ in range(len(maddpg.agents))]
        action_masks = [[] for _ in range(len(maddpg.agents))]
        # ---------- 获取当前状态 ------------
        encoded_state = state['encoded_data'].unsqueeze(0).to(device)
        state_mask = state['mask'].unsqueeze(0).to(device)
        # ---------- 动作推理 ------------            
        with torch.no_grad():
            actors_input, _, _, actors_mask, target_features, target_mask,our_entity_coordinate,enemy_coordinate,enemy_speeds = maddpg._encode_global_and_self_state({
                'encoded_data': encoded_state.to(device),
                'mask': state_mask.to(device),
            }, logger)

            for i, agent in enumerate(maddpg.agents):

                if isinstance(agent.actor, AttackTargetActor):
                    action = agent.actor(
                        actors_input.to(device),
                        target_features=target_features.to(device),
                        self_mask=actors_mask.to(device),
                        target_mask=target_mask.to(device),
                        self_coords=our_entity_coordinate.to(device),   # [B, E, 2]
                        target_coords=enemy_coordinate.to(device) ,  # [B, T, 2]
                        target_speeds=enemy_speeds.to(device)
                    )
                else:
                    action = agent.actor(actors_input.to(device))

                action_tensor = actor_output_to_action(action, max_entity_len=actors_input.shape[1]) # [1, max_entity_len, 5]

                # -------------- 衰减探索 --------------
                agent_epsilon = agent.epsilon if hasattr(agent, "epsilon") else 0.1
                action_tensor = apply_exploration(action_tensor.squeeze(0), epsilon=agent_epsilon, actor_index=i)

                all_actions[i] = (action_tensor.squeeze(0)) # action_tensor.squeeze(0) -> [max_entity_len, 5]
                action_masks[i] = (actors_mask.squeeze(0))

        # 执行动作

        all_actions = np.stack([a.cpu().numpy() for a in all_actions])
        action_masks = np.stack([a.cpu().numpy() for a in action_masks])


        all_actions = bootstrap_recon(all_actions, encoded_state, state_mask, cur_step=env.current_step) #[8,E,5]

        time.sleep(1.0)
        next_state, reward, done, our_entity_ids, actions_executed = env.step(all_actions, state, logger)

        done = (done == 1)

        state = next_state

        if step_count >= BUFFER_CAP:
            controller.simulating = False
            break

def main():
    env = SimulatedEnv()
    import time
    current_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = Log(name=current_time, log_dir="logs")

    controller = SignalListener(env, logger)

    maddpg = MADDPG(
        actor_types=ACTOR_TYPES,
        state_dim=256,
        action_feat_dim=128,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.95,
        tau=0.02,
        batch_size=1
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path=f"ckp/maddpg_model_last.pt"
    if os.path.exists(model_path):
        try:
            if not torch.cuda.is_available():
                load_checkpoint(maddpg, model_path, logger)
            else:
                load_checkpoint2(maddpg, model_path, logger)
        except Exception as e:
            print(e)
            load_checkpoint2(maddpg, model_path, logger)
    else:
        logger.info("no model find")

    maddpg.device = device
    maddpg.to(device) 

    threading.Thread(target=controller.simulate_signals, daemon=True).start()

    while True:
        # 等待 signal 设置 controller.simulating=True
        while not controller.simulating:
            time.sleep(0.2)

        # 开始执行主循环（可中途被 stop/restart）
        main_loop(env, controller, logger, maddpg, device)

        logger.info("[MAIN] Simulation ended. Waiting for new signal...")
 
    
def load_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.use_deterministic_algorithms(True)

    return seed

if __name__ == "__main__":
    load_seed(42)
    set_is_train(False)
    main()
