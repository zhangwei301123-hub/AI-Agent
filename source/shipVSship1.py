import numpy as np
from gym import spaces

from actor import * 
AIR_STATUS_MAP = {
    'AIRBORNE': 0,
    'TAXYING_TO_TAKE_OFF': 1,
    'TAKING_OFF': 2,
    'LANDING_POST_TOUCHDOWN': 3,
    'HOLDING_FOR_AVAILABLE_TRANSIT': 4,
    'HOLDING_ON_LANDING_QUEUE': 5,
    'PREPARING_TO_LAUNCH': 6,
    'REFUELLING': 7,
    'DEPLOYING_DIPPING_SONAR': 8,
    'TAXYING_TO_FLIGHT_DECK': 9,
    'BVRCRANK': 10,
    'TRANSFERRING_CARGO': 11,
    'PARKED': 12,
    'TAXYING_TO_PARK': 13,
    'LANDING_PRE_TOUCHDOWN': 14,
    'READYING': 15,
    'HOLDING_FOR_AVAILABLE_RUNWAY': 16,
    'RTB': 17,
    'MANOEUVERING_TO_REFUEL': 18,
    'OFFLOADING_FUEL': 19,
    'EMERGENCY_LANDING': 20,
    'CONST_19': 21,  # 特殊占位符保留
    'DOGFIGHT': 22
}

unit_status_mapping = {
    'UNASSIGNED': 0,
    'ENGAGED_OFFENSIVE': 1,
    'ON_ATTACK_RUN': 2,
    'RTB': 3,
    'FORMING_UP': 4,
    'ON_SUPPORT_MISSION': 5,
    'HEADING_TO_REFUEL_POINT': 6,
    'RTB_MISSION_OVER': 7,
    'RTB_GROUP': 8,
    'ON_PLOTTED_COURSE': 9,
    'ENGAGED_DEFENSIVE': 10,
    'ON_PATROL': 11,
    'TASKED': 12,
    'RTB_MANUAL': 13,
    'ON_FERRY_MISSION': 14,
    'REFUELLING': 15,
    'GROUP_LEAD_SLOWING_TO_ALLOW_FORM_UP': 16,
    'RTB_CALLED_OFF': 17
}




class TacticalController(nn.Module):
    """战术指令协同决策系统"""
    def __init__(self, config):
        super().__init__()
        self.obs_encoder = EntityEncoder()
        
        # 指令Actor集群
        self.takeoff_actor = AircraftTakeOffActor()
        self.rtb_actor = ReturnToBaseActor()
        self.move_actor = WaypointMoveActor()
        self.mobility_actor = MobilityActor()
        self.judge_actor = AttackJudgeActor()
        self.strike_actor = StrikeActor()
        self.sensor_actor = SensorControlActor()

        # 战术记忆模块
        self.memory_cell = nn.GRUCell(256, 256)

        
        # 冲突解决模块
        self.conflict_resolver = nn.Sequential(
            nn.Linear(256*3, 512),
            nn.ReLU(),
            nn.Linear(512, 3),  # 起飞/返航/移动的权重分配
            nn.Softmax(dim=-1)
        )

    def forward(self, raw_obs):
        # 状态编码
        encoded_state = self.obs_encoder(raw_obs)
        
        # 并行生成指令草案
        takeoff_cmd = self.takeoff_actor(encoded_state)
        rtb_cmd = self.rtb_actor(encoded_state)
        move_cmd = self.move_actor(encoded_state)
        
        # 指令协同优化
        conflict_input = torch.cat([
            takeoff_cmd['takeoff_signal'],
            rtb_cmd['rtb_priority'],
            move_cmd[0].mean(dim=-1)
        ], dim=-1)
        weights = self.conflict_resolver(conflict_input)
        
        # 生成最终指令
        return {
            'takeoff': {
                'waypoints': takeoff_cmd['waypoints'],
                'active': weights[:, 0] > 0.5
            },
            'rtb': {
                'priority': rtb_cmd['rtb_priority'] * weights[:, 1],
                'fuel_level': rtb_cmd['fuel_reserve']
            },
            'movement': (move_cmd[0] * weights[:, 2].unsqueeze(-1), 
                        move_cmd[1])
        }





    
# def build_observation_space(max_entities=50):
#     """构建包含多维战场信息的观察空间"""
#     return spaces.Dict({  # 使用字典空间组合不同类型的信息
#         # 全局战场信息 --------------------------------------------------
#         'resources': spaces.Box(0, 1, shape=(3,)),  
#         # 油料/弹药/电力（归一化到0-1的3维向量）

#         # 实体信息（动态填充）-------------------------------------------
#         'entities': spaces.Dict({  # 嵌套字典空间描述实体信息
#             'attributes': spaces.Box(-1, 1, shape=(max_entities, 16)),
#             # 实体属性矩阵：50个实体×16维特征（包含坐标/血量/速度等归一化值）
            
#             'mask': spaces.MultiBinary(max_entities)  
#             # 实体有效性掩码：50维二进制向量（1表示有效实体，0为填充空位）
#         }),
        
#         # 态势感知信息 --------------------------------------------------
#         'threat_level': spaces.Box(0, 1, shape=(4,)),  
#         # 海陆空天四维度威胁指数（每个维度0-1）
        
#         'strategic_map': spaces.Box(0, 1, shape=(64, 64))  
#         # 64×64的战略热力图（每个网格0-1表示战略价值）
#     })

# def build_action_space(max_units=20):
#     """
#     构建包含多种作战指令的动作空间
#     返回值：gym.spaces.Dict 对象，包含多个子动作空间
#     """
    
#     return spaces.Dict({
#         # 攻击指令（Attack）
#         'attack': spaces.Dict({
#             'target_type': spaces.Discrete(3),  # 0:陆军 1:海军 2:空军 离散3选1
#             'priority': spaces.Box(0, 1, shape=(1,)), #0-1优先级
#             'units': spaces.MultiBinary(max_units)  #20位二进制掩码 
#         }),
#         # 巡逻指令（Patrol）
#         'patrol': spaces.Dict({
#             'area_center': spaces.Box(0, 1, shape=(2,)), #2D坐标
#             'radius': spaces.Box(0.1, 0.5, shape=(1,)),  #范围半径
#             'pattern': spaces.Discrete(3) # 3种模式
#         }),
#         # 移动指令（Move）
#         'move': spaces.Dict({  
#             'destination': spaces.Box(0, 1, shape=(3,)),  # x,y,z 
#             'formation': spaces.Discrete(5),
#             'speed': spaces.Box(0.1, 1.0, shape=(1,))
#         }),
#         # 部署浮标指令（Deploy）
#         'deploy': spaces.Dict({
#             'buoy_type': spaces.Discrete(3),
#             'position': spaces.Box(0, 1, shape=(2,)),
#             'quantity': spaces.Box(1, 5, shape=(1,))
#         })
#     })


import numpy as np

class EntityEncoder:
    def __init__(self, max_entities=50):
        self.max_entities = max_entities
        self.feature_dim = 28 # 每个实体的特征维度

    def encode(self, raw_entities):
        encoded = np.zeros((self.max_entities, self.feature_dim))  # 确保feature_dim足够大（当前需要28）
        mask = np.zeros(self.max_entities)
        entity_count = 0

        for entity in raw_entities:
            if entity_count >= self.max_entities:
                break

            # 特征索引映射表 --------------------------------------------------------
            # [0] 阵营信息（原过滤条件改为特征）
            forceSide = entity.get('forceSide', [1])[0] if 'forceSide' in entity else -1
            encoded[entity_count, 0] = forceSide

            # [1] 血量
            activeLvl = entity.get('activeLvl', [0])[0] if 'activeLvl' in entity else -1
            encoded[entity_count, 1] = activeLvl

            # [2-4] 姿态角（俯仰/翻滚/偏航）
            attitude = entity.get('attitude', {})
            encoded[entity_count, 2] = attitude.get('pitch', [0])[0] if 'pitch' in attitude else -1
            encoded[entity_count, 3] = attitude.get('roll', [0])[0] if 'roll' in attitude else -1
            encoded[entity_count, 4] = attitude.get('yaw', [0])[0] if 'yaw' in attitude else -1

            # [5-7] 空间坐标（高度/纬度/经度）
            spatial_coord = entity.get('entitySpatialCoord', {})
            encoded[entity_count, 5] = spatial_coord.get('altitude', [0])[0] if 'altitude' in spatial_coord else -1
            encoded[entity_count, 6] = spatial_coord.get('latitude', [0])[0] if 'latitude' in spatial_coord else -1
            encoded[entity_count, 7] = spatial_coord.get('longitude', [0])[0] if 'longitude' in spatial_coord else -1

            # [8] 基地关联
            attrMap = entity.get('attrMap', {})
            encoded[entity_count, 8] = 1 if 'AirBase' in attrMap else -1  #他若有值就证明这个实体是飞机

            # [9] 油量
            logistic = entity.get('logisticStates', {}).get('0', [0])[0] if 'logisticStates' in entity else -1
            encoded[entity_count, 9] = logistic

            # [10-12] 速度分量（vx/vy/vz）
            velocity = entity.get('velocity', {})
            encoded[entity_count, 10] = velocity.get('vx', [0])[0] if 'vx' in velocity else -1
            encoded[entity_count, 11] = velocity.get('vy', [0])[0] if 'vy' in velocity else -1
            encoded[entity_count, 12] = velocity.get('vz', [0])[0] if 'vz' in velocity else -1

            # [13] 实体类型（0=飞机，1=舰船，2=其他）
            mdl_type = entity.get('mdlType', [])
            encoded[entity_count, 13] = 0 if 'Aircraft' in mdl_type else 1 if 'Ship' in mdl_type else 2

            # [14] 干扰状态
            inner = entity.get('innerstates', {})
            encoded[entity_count, 14] = 1 if 'IsJamReaction' in inner else -1

            # [15] 目标丢失时间
            encoded[entity_count, 15] = inner.get('lostTime', [0])[0] if 'lostTime' in inner else -1

            # [16-17] 武器挂载
            load_map = entity.get('loadMap', {})
            encoded[entity_count, 16] = load_map.get('GUIDED_WEAPON', [0])[0] if 'GUIDED_WEAPON' in load_map else -1
            encoded[entity_count, 17] = load_map.get('SONOBUY', [0])[0] if 'SONOBUY' in load_map else -1

            # [18] 实体序列ID（原entity_count） 执行命令的实体id
            encoded[entity_count, 18] = entity_count 

            # [19] 报告时间戳
            reportTime = entity.get('reportTime', [0])[0] if 'reportTime' in entity else -1
            encoded[entity_count, 19] = reportTime

            # [20] 燃油消耗率
            state_map = entity.get('stateMap', {})
            encoded[entity_count, 20] = state_map.get('FuelBurnRate', [0])[0] if 'FuelBurnRate' in state_map else -1

            # [21] 剩余航程
            encoded[entity_count, 21] = state_map.get('RemainDistance', [0])[0] if 'RemainDistance' in state_map else -1

            # [22] 单位状态（UnitStatus）
            UnitStatus = state_map.get('UnitStatus', [''])[0]
            encoded[entity_count, 22] = unit_status_mapping.get(UnitStatus, -1)

            # [23] 空中状态（AirStatus）
            air_status = state_map.get('AirStatus', [''])[0]
            encoded[entity_count, 23] = AIR_STATUS_MAP.get(air_status, -1)

            # [24-27] 传感器状态 
            encoded[entity_count, 24] = state_map.get('EcmStatus', [0])[0] if 'EcmStatus' in state_map else -1
            encoded[entity_count, 25] = state_map.get('RadarStatus', [0])[0] if 'RadarStatus' in state_map else -1
            encoded[entity_count, 26] = state_map.get('SonarStatus', [0])[0] if 'SonarStatus' in state_map else -1
            encoded[entity_count, 27] = state_map.get('IdentifyStatus', [0])[0] if 'IdentifyStatus' in state_map else -1

            # 有效性标记
            mask[entity_count] = 1
            entity_count += 1

        return encoded, mask


class GlobalStateEncoder(nn.Module):
    """全局状态编码器"""
    def __init__(self, entity_dim=28, hidden_dim=256):
        super().__init__()
        # 实体级编码
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU()
        )
        
        # 注意力聚合层
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        
        # 全局特征提取
        self.global_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, entities, mask):
        # 编码每个实体特征 [batch, max_ent, 28] -> [batch, max_ent, 128]
        ent_feat = self.entity_encoder(entities)
        
        # 注意力聚合 (考虑实体有效性)
        attn_output, _ = self.attention(
            query=ent_feat,
            key=ent_feat,
            value=ent_feat,
            key_padding_mask=~mask
        )
        
        # 全局特征 [batch, hidden_dim]
        global_feat = self.global_encoder(attn_output.mean(dim=1))
        return global_feat

import torch
import torch.nn as nn
import torch.nn.functional as F

# 五、MADDPG 网络架构

class BattleActor(nn.Module):
    """策略网络：生成作战指令参数"""
    def __init__(self, obs_dim=256, action_dims=None):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 指令分支
        self.attack_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, action_dims['attack'])
        )
        
        self.patrol_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, action_dims['patrol'])
        )
        
        # 其他指令分支类似...

    def forward(self, state):
        x = self.obs_encoder(state)
        return {
            'attack': torch.sigmoid(self.attack_net(x)),
            'patrol': self.patrol_net(x),
            # 其他指令...
        }

class BattleCritic(nn.Module):
    """价值网络：评估联合动作价值"""
    def __init__(self, obs_dim=256, action_dims=None):
        super().__init__()
        total_actions = sum(action_dims.values())
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim + total_actions, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state, actions):
        x = torch.cat([state, torch.cat(actions.values(), dim=-1)], dim=-1)
        return self.q_net(x)



# 六、训练框架实现
class MADDPGTrainer:
    def __init__(self, env, config):
        self.env = env
        self.agents = {
            'attack': DDPGAgent(),
            'patrol': DDPGAgent(),
            'move': DDPGAgent(),
            'deploy': DDPGAgent()
        }
        self.buffer = ReplayBuffer(config.buffer_size)
        self.encoder = EntityEncoder()

    def train_episode(self):
        state = self.env.reset()
        total_rewards = {k:0 for k in self.agents}
        
        while not done:
            Entity_number=get_env_entity_number(state)
            # 1. 编码观察空间
            encoded_state = self._encode_state(state)
            
            # 2. 获取各agent动作
            actions = {}

            for name, agent in self.agents.items():
                actions[name] = agent.act(encoded_state)
            
            # 3. 环境交互
            next_state, rewards, done, _ = self.env.step(actions)
            
            # 4. 存储经验
            self.buffer.add(encoded_state, actions, rewards, 
                          self._encode_state(next_state), done)
            
            # 5. 更新网络
            if len(self.buffer) > config.batch_size:
                for name in self.agents:
                    self._update_agent(name)
            
            # 更新状态
            state = next_state
            for k in total_rewards:
                total_rewards[k] += rewards[k]
        
        return total_rewards

    def _encode_state(self, raw_state):
        """编码原始游戏状态"""
        entities = self.encoder.encode(raw_state['entities'])
        return {
            'resources': np.array(raw_state['resources']),
            'entities': entities,
            'threat_level': raw_state['threat_level'],
            'strategic_map': raw_state['strategic_map']
        }

    def _update_agent(self, agent_name):
        """更新指定agent的网络"""
        batch = self.buffer.sample(config.batch_size)
        
        # 计算目标Q值
        with torch.no_grad():
            target_actions = self.agents[agent_name].target_actor(batch.next_states)
            q_next = self.agents[agent_name].target_critic(batch.next_states, target_actions)
            q_target = batch.rewards + config.gamma * (1 - batch.dones) * q_next
        
        # 计算当前Q值
        current_q = self.agents[agent_name].critic(batch.states, batch.actions)
        
        # 更新Critic
        critic_loss = F.mse_loss(current_q, q_target)
        self.agents[agent_name].critic_optim.zero_grad()
        critic_loss.backward()
        self.agents[agent_name].critic_optim.step()
        
        # 更新Actor
        policy_loss = -self.agents[agent_name].critic(batch.states, 
                                                     self.agents[agent_name].actor(batch.states))
        self.agents[agent_name].actor_optim.zero_grad()
        policy_loss.mean().backward()
        self.agents[agent_name].actor_optim.step()
        
        # 软更新目标网络
        self.agents[agent_name].soft_update()


# 七、奖励函数设计
class RewardCalculator:
    def __init__(self):
        self.prev_state = None
        
    def calculate(self, curr_state, actions):
        """计算复合奖励"""
        rewards = {}
        
        # 攻击奖励
        attack_r = self._attack_reward(curr_state)
        # 巡逻奖励
        patrol_r = self._patrol_coverage(curr_state)
        # 移动奖励
        move_r = self._movement_efficiency(curr_state)
        # 部署奖励
        deploy_r = self._deploy_effectiveness(curr_state)
        
        # 组合各指令奖励
        rewards['attack'] = attack_r * 0.6 + patrol_r * 0.1 + move_r * 0.3
        rewards['patrol'] = patrol_r * 0.7 + attack_r * 0.3
        rewards['move'] = move_r * 0.5 + attack_r * 0.3 + patrol_r * 0.2
        rewards['deploy'] = deploy_r * 0.8 + attack_r * 0.2
        
        # 资源惩罚项
        resource_penalty = np.clip(1 - curr_state['resources'].mean(), 0, 1)
        for k in rewards:
            rewards[k] -= resource_penalty * 0.2
            
        return rewards

    def _attack_reward(self, state):
        """计算攻击效果奖励"""
        enemy_damage = self.prev_state['enemy_force'] - state['enemy_force']
        return enemy_damage / 100  # 假设每100单位伤害得1分
    
    def _patrol_coverage(self, state):
        """计算巡逻区域覆盖率"""
        # 需要接入地图数据
        return np.clip(state['patrol_coverage'], 0, 1)
    
    def _movement_efficiency(self, state):
        """计算移动效率奖励"""
        speed = state['average_speed']
        return np.clip((speed - 0.5) * 2, 0, 1)  # 0.5为基准速度
    
    def _deploy_effectiveness(self, state):
        """计算部署有效性"""
        detected_enemies = state['detected_count'] - self.prev_state['detected_count']
        return detected_enemies / 10  # 每发现10个单位得1分



class TrainingConfig:
    # 网络参数
    actor_lr = 3e-4
    critic_lr = 1e-3
    hidden_dim = 512
    gamma = 0.95
    
    # 训练参数
    batch_size = 1024
    buffer_size = 1e6
    warmup_steps = 5000
    target_update = 0.01
    
    # 环境参数
    max_entities = 100
    map_size = (1000, 1000)  # 单位：公里
    episode_length = 300  # 每局最大步数
    
    # 探索参数
    noise_scale = 0.3
    noise_decay = 0.995
    min_noise = 0.1


main()