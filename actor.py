import numpy as np
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.ma.core import max_filler

from entity import  AIR_STATUS_MAP, MAX_ACTION_ENTITIES, MAX_TARGETS
# 所有actor指令输出横向拼接 最大维度为5


# ACTOR_TYPES_str = [
#     AircraftTakeOffActor,0
#     ReturnToBaseActor, 1
#     WayPointMoveActor,2
#     MobilityActor, 3 
#     AttackTargetActor,4 
#     SensorControlActor, 5
#     DeploySonobuoyActor, 6 
#     CancelAttackActor 7
# ]
class AircraftTakeOffActor(nn.Module):
    """起飞指令生成器：控制飞机起飞流程和航线规划"""
    def __init__(self, obs_dim=512):
        super().__init__()
        self.takeoff_decider = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),   
            nn.Linear(128, 1) ,
            nn.Sigmoid()      # 起飞决策概率 [0, 1]
        )

    def forward(self, state):

        takeoff_prob = self.takeoff_decider(state)
        return {
            "takeoff_prob": takeoff_prob.squeeze(-1)
        } # 1
    



class ReturnToBaseActor(nn.Module):
    """返航决策生成器：智能返航判断"""
    def __init__(self, obs_dim=512):
        super().__init__()
        # 返航决策网络
        self.return_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),  
            nn.Sigmoid()# 输出返航决策概率
        )

    def forward(self, state):
        """
        :param state: 实体状态特征 [batch_size, obs_dim]
        :return: 返航触发概率
        """
        return_prob = self.return_net(state)
        return {
            'return_prob': return_prob.squeeze(-1)  # 输出维度 [batch_size]
        } # 1


# class WayPointMoveActor(nn.Module):
#     """路径机动指令生成器：为单个单位设置机动路径"""
#     def __init__(self, obs_dim=512):
#         super().__init__()
#         # 航点生成网络（输出 [-1, 1]）
#         self.waypoint_net = nn.Sequential(
#             nn.Linear(obs_dim, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, 4),  # 输出4个参数: lon, lat, alt, vel
#             nn.Tanh()
#         )
        
#         # 执行决策网络
#         self.return_decider = nn.Sequential(
#             nn.Linear(obs_dim, 256),
#             nn.ReLU(),         # 推荐用 ReLU
#             nn.Linear(256, 1),
#             nn.Sigmoid()       # 输出概率
#         )

#         # 经纬度物理边界（只在 forward 中使用）
#         self.lon_min = -180.0
#         self.lon_max = 180.0
#         self.lat_min = -90.0
#         self.lat_max = 90.0

#     def forward(self, state):
#         # 生成原始航点参数
#         waypoints = self.waypoint_net(state).view(-1, 4)
        
#         longitude = 0.5 * (waypoints[:, 0] + 1.0) * (self.lon_max - self.lon_min) + self.lon_min
#         latitude = 0.5 * (waypoints[:, 1] + 1.0) * (self.lat_max - self.lat_min) + self.lat_min

#         # 高度 & 速度保持原始 [-1, 1] 输出（也可以后续 clamp 或缩放）
#         altitude = waypoints[:, 2]
#         velocity = waypoints[:, 3]

        
#         # 生成决策概率
#         return_prob = self.return_decider(state)
        
#         return {
#             "wayPoint": {
#                 "longitude": longitude,
#                 "latitude":  latitude,
#                 "altitude":  altitude,
#                 "velocity":  velocity
#             },
#             "return_prob": return_prob.squeeze(-1)
#         } # 5
    
class WayPointMoveActor(nn.Module):
    """
    路径机动指令生成器（离散方向版本）
    DIRECTION_MAP = {
            0: (0, +3),   # 前
            1: (+3, +3),  # 右前
            2: (+3, 0),   # 右
            3: (+3, -3),  # 右后
            4: (0, -3),   # 后
            5: (-3, -3),  # 左后
            6: (-3, 0),   # 左
            7: (-3, +3),  # 左前
            8： 漏斗状巡航
            9:  弓形巡航
            10: 8字状巡航（随机巡航）
            11：螺旋状搜索
        }
"""
    def __init__(self, obs_dim=512):
        super().__init__()

        # 方向分类：8个方向 +4 = 12种搜索
        self.direction_classifier = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 12)  # 输出12个方向的 logits 剩下的探索只能
        )

        # 高度 & 速度输出
        self.alt_vel_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh()  # 输出 ∈ [-1, 1]，后续量化
        )

        # 执行概率
        self.return_decider = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, state):

        batch_size, max_entity_len, obs_dim = state.shape
        flat_state = state.view(-1, obs_dim)  # [B * E, 512]

        direction_logits = self.direction_classifier(flat_state)
        bias_weights = torch.tensor([0.225, 0.036, 0.036, 0.036, 0.225, 0.036, 0.27, 0.036 ,0.025,0.025,0.025,0.025], device=direction_logits.device)
        log_bias = torch.log(bias_weights + 1e-6)  # 避免 log(0)
        biased_logits = direction_logits + log_bias
        # softmax + 采样
        direction_probs = torch.softmax(biased_logits, dim=-1)
        direction_index = torch.multinomial(direction_probs, num_samples=1).view(batch_size, max_entity_len)

        # # 原始方向 logits
        # direction_logits = self.direction_classifier(flat_state)
        # direction_probs = torch.softmax(direction_logits, dim=-1)
        # direction_index = torch.multinomial(direction_probs, num_samples=1).view(batch_size, max_entity_len)
        

        # 连续量
        alt_vel = self.alt_vel_net(flat_state)  # [B * E, 2]
        altitude = alt_vel[:, 0].view(batch_size, max_entity_len)
        velocity = alt_vel[:, 1].view(batch_size, max_entity_len)

        # 占位量
        ZERO = torch.zeros_like(altitude)

        # 执行概率
        return_prob = self.return_decider(flat_state).view(batch_size, max_entity_len)

        return {
            "wayPoint": {
                "direction_index": direction_index,
                "ZERO": ZERO,
                "altitude": altitude,
                "velocity": velocity
            },
            "return_prob": return_prob
        }

    
class MobilityActor(nn.Module):
    """机动指令生成器：智能速度/高度调整"""
    def __init__(self, obs_dim=512):
        super().__init__()
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # 速度控制分支
        self.speed_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(), 
            nn.Linear(128, 1),  
            nn.Sigmoid() # 限定输出 ∈ (0,1)
        )
        
        # 高度控制分支
        self.alt_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid() # 限定输出 ∈ (0,1)
        )
        
        # 执行概率预测
        self.mobility_net = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )



    def forward(self, state):
        # 特征提取
        feat = self.feature_extractor(state)
        
        # 速度调整量（相对当前速度的百分比）
        speed_ratio = self.speed_net(feat) 
        
        # 高度调整量（自动处理符号）
        altitude = self.alt_net(feat) 
        
        # 执行概率
        mobility_prob = self.mobility_net(feat)

        return {
            "speed": speed_ratio.squeeze(-1),     # [batch]
            "altitude": altitude.squeeze(-1),     # [batch]
            "mobility_prob": mobility_prob.squeeze(-1)  # [batch]
        } # 3

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(torch.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    return c * 6371  # 地球半径6371km

class AttackTargetActor(nn.Module):
    
    def __init__(self, 
                 self_obs_dim=512,    # 攻击方自身状态维度
                 target_obs_dim=256): # 目标状态特征维度
        super().__init__()
        
        # 攻击方自身状态编码器
        self.self_encoder = nn.Sequential(
            nn.Linear(self_obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # 目标特征适配器（当目标特征维度需要调整时）
        self.target_adapter = nn.Sequential(
            nn.Linear(target_obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # 攻击决策网络
        self.attack_net = nn.Sequential(
            nn.Linear(256 + 256 + 1 + 1, 128),  # 增加距离特征  增加速度特征
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.perform_net = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state, target_features, target_mask=None, self_mask=None, target_coords=None,self_coords=None,target_speeds=None):
        # 计算球面距离
        # self_coords [batch, max_entity_len, 2]
        # target_coords [batch,target_max_entity_len, 2]
        B, E = state.shape[:2]
        T     = target_features.size(1)
        # 编码自身状态（每个实体）
        self_feat = self.self_encoder(state)  # [batch, max_entity_len, 256]

        # 扩展广播形状 [B, E, T, 2]
        self_coords_exp = self_coords.unsqueeze(2)  # [B,E,1,2]
        target_coords_exp = target_coords.unsqueeze(1)  # [B,1,T,2]

        # 计算球面距离
        distance = haversine(
            self_coords_exp[...,0],
            self_coords_exp[...,1],
            target_coords_exp[...,0],
            target_coords_exp[...,1]
        ).to(self_feat.dtype)   # [B,E,T]

        if target_mask is not None:
            target_mask = target_mask.unsqueeze(1).expand_as(distance)
            distance = distance.masked_fill(~target_mask, 0)

        if self_mask is not None:
            self_mask = self_mask.unsqueeze(-1).expand_as(distance)
            distance = distance.masked_fill(~self_mask, 0)

        # distance = distance / distance.sum(dim=-1, keepdims=True)
        dist_sum = distance.sum(dim=-1, keepdims=True) + 1e-6  # ε 避免除 0
        distance  = distance / dist_sum                        # [B,E,T]

        # k = min(10, T)
        # _, topk_idx = distance.topk(k, dim=2, largest=False)  # [B, E, k]
        # topk_mask = torch.zeros_like(distance, dtype=torch.bool)
        # topk_mask.scatter_(2, topk_idx, True)
        target_speeds = target_speeds.unsqueeze(-1)
        

        speed_exp = target_speeds.unsqueeze(1).expand(-1, E, -1, -1)     # [B, E, T, 2]
        # speed_exp = speed_exp.masked_fill(~topk_mask.unsqueeze(-1), 0.)  # 仅最近 k
        speed = speed_exp.squeeze(-1)
        speed_sum = speed.sum(dim=-1, keepdims=True) + 1e-6
        speed = speed / speed_sum
        speed_exp_penalty = 0.35 * speed

        # 距离惩罚系数，可调节强度（越大越偏向近处） 
        distance_penalty = -0.4 * distance  # [B, E, T]
        # else:
        #     distance = torch.zeros(state.shape[0], MAX_ACTION_ENTITIES, MAX_TARGETS).to(self_feat.dtype)
        #     distance_penalty = 0 * distance


        
        # 处理目标特征
        target_feat = self.target_adapter(target_features)  # [batch, max_targets, 256]
        
        # 特征融合（实体与目标交叉组合）
        self_exp = self_feat.unsqueeze(2).expand(-1,-1,target_feat.size(1),-1)  # [B,E,T,256]
        target_exp = target_feat.unsqueeze(1).expand(-1,self_feat.size(1),-1,-1)  # [B,E,T,256]
        
        # 加入距离特征
        fused = torch.cat([
            self_exp, 
            target_exp,
            distance.unsqueeze(-1),  # [B,E,T,1]
            speed_exp
        ], dim=-1)  # [B,E,T,513]
        
        # 计算攻击概率
        # attack_logits = self.attack_net(fused).squeeze(-1)  # [B,E,T]
        # attack_probs = torch.softmax(attack_logits + distance_penalty, dim=-1)
        attack_logits = self.attack_net(fused).squeeze(-1)  # [B,E,T]
        if target_mask is not None:
            masked_logits = attack_logits.masked_fill(~target_mask, -1e9)
        else:
            masked_logits = attack_logits
        attack_probs = torch.softmax(masked_logits + distance_penalty + speed_exp_penalty, dim=-1)
        
        # 应用双重掩码
        if target_mask is not None:
            # target_mask = target_mask.unsqueeze(1).expand_as(attack_probs)
            attack_probs = attack_probs.masked_fill(~target_mask, 0)
        if self_mask is not None:
            # self_mask = self_mask.unsqueeze(-1).expand_as(attack_probs)
            attack_probs = attack_probs.masked_fill(~self_mask, 0)
        
        # 执行概率
        perform_probs = self.perform_net(self_feat).squeeze(-1)  # [B,E]
        
        return {
            "selected_index": torch.argmax(attack_probs, dim=-1),
            "attack_prob": perform_probs
        }


class SensorControlActor(nn.Module):
    """多模态传感器控制网络"""
    def __init__(self, obs_dim=512):
        super().__init__()
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # 多传感器联合控制分支（并行输出三类传感器状态）
        self.sensor_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 输出radar/sonar/ecm的独立控制信号
            nn.Sigmoid()
        )
        
        # 控制执行概率分支
        self.control_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        # 特征提取
        feat = self.feature_extractor(state)
        
        # 获取三类传感器的控制概率 [batch, 3]
        sensor_probs = self.sensor_net(feat)
        
        # 控制指令执行概率
        control_prob = self.control_net(feat).squeeze(-1)
        
        return {
            "radarOperationStatus": sensor_probs[:, :, 0],  # 雷达操作概率
            "sonarOperationStatus": sensor_probs[:, :, 1],  # 声纳操作概率
            "ecmOperationStatus": sensor_probs[:, :, 2],    # 电子对抗操作概率
            "SensorControl_prob": control_prob           # 指令执行概率
        } # 4



class DeploySonobuoyActor(nn.Module):
    """声呐浮标部署指令生成器：控制反潜单位部署声呐浮标的类型和位置"""
    def __init__(self, obs_dim=512):
        super().__init__()
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # 浮标类型选择分支（被动/主动）
        self.type_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 部署深度选择分支（浅/深）
        self.depth_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 映射到0-1范围
        )
        
        # 部署决策分支
        self.deploy_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        # 特征提取
        feat = self.feature_extractor(state)
        
        # 浮标类型选择概率（主动为True）
        type_prob = self.type_net(feat).squeeze(-1)
        
        # 深度选择（0-1映射到实际深度）
        depth_prob = self.depth_net(feat).squeeze(-1)
        
        # 部署决策概率
        deploy_prob = self.deploy_net(feat).squeeze(-1)
        
        return {
            "passiveOrActive": type_prob,    # 被动(false)/主动(true)的概率
            "shallowOrDeep": depth_prob,     # 深度选择概率（0-1） 
            "deploy_prob": deploy_prob       # 执行部署的概率
        } # 3

from entity import AIR_STATUS_MAP


class CancelAttackActor(nn.Module):
    """放弃打击指令生成器：控制作战单元取消所有已计划的攻击任务"""
    def __init__(self, obs_dim=512):
        super().__init__()
        # 共享特征提取层（与其它Actor架构统一）
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # 决策核心网络（双层MLP）
        self.decision_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出范围约束在[0,1]
        )

    def forward(self, state):
        # 特征抽象
        feat = self.feature_extractor(state)
        
        # 生成取消打击概率
        cancel_prob = self.decision_net(feat).squeeze(-1)
        
        return {
            "CancelAttack_prob": cancel_prob  # [batch_size]
        } # 1


def quantize_speed(speed_tensor):
    """
    将 [0, 1] 范围内的 speed 张量映射为 0~4 的等级
    """
    quantized = torch.zeros_like(speed_tensor, dtype=torch.long)

    quantized = torch.where(speed_tensor < 0.2, torch.full_like(quantized, 1), quantized)
    quantized = torch.where((speed_tensor >= 0.2) & (speed_tensor < 0.4), torch.full_like(quantized, 2), quantized)
    quantized = torch.where((speed_tensor >= 0.4) & (speed_tensor < 0.6), torch.full_like(quantized, 3), quantized)
    quantized = torch.where((speed_tensor >= 0.6) & (speed_tensor < 0.8), torch.full_like(quantized, 4), quantized)
    quantized = torch.where(speed_tensor >= 0.8, torch.full_like(quantized, 5), quantized)

    return quantized


def quantize_altitude(alt_tensor):
    """
    将 [0,1] 范围内的 altitude 映射为 6 个等级。
    支持 shape: [batch, num_entities] 或更高维
    """
    quantized = torch.zeros_like(alt_tensor, dtype=torch.long)

    quantized = torch.where(alt_tensor <= 0.1, torch.full_like(quantized, 1), quantized)
    quantized = torch.where((alt_tensor > 0.1) & (alt_tensor <= 0.25), torch.full_like(quantized, 2), quantized)
    quantized = torch.where((alt_tensor > 0.25) & (alt_tensor <= 0.4), torch.full_like(quantized, 3), quantized)
    quantized = torch.where((alt_tensor > 0.4) & (alt_tensor <= 0.6), torch.full_like(quantized, 4), quantized)
    quantized = torch.where((alt_tensor > 0.6) & (alt_tensor <= 0.8), torch.full_like(quantized, 5), quantized)
    quantized = torch.where(alt_tensor > 0.8, torch.full_like(quantized, 6), quantized)

    return quantized


# 将actor的输出映射到动作空间 字典=》tensor列表
# 输入是actor类的return (长度不一) [takeoff_prob，return_prob...]
# 输出是[batch,number_actor,5] 5代表5个参数的指令
# 第一个参数代表某个actor命令执行概率，2-4代表其具体参数
def actor_output_to_action(actor_output, max_entity_len=None):
    action_parts = []
    
    # 判断是否是多维输入
    is_multi_dim = max_entity_len is not None
    
    # AircraftTakeOffActor
    if 'takeoff_prob' in actor_output and 'wayPoint' not in actor_output:
        takeoff_prob = actor_output['takeoff_prob']
        zeros = torch.zeros_like(takeoff_prob)
        if is_multi_dim:
            # 多维逻辑
            # 确保所有张量的形状为 [batch_size, max_entity_len]
            takeoff_prob = takeoff_prob.view(-1, max_entity_len)
            zeros = zeros.view(-1, max_entity_len)
            action_parts = [takeoff_prob, zeros, zeros, zeros, zeros]
        else:
            # 单维逻辑
            action_parts = [actor_output['takeoff_prob'],zeros, zeros, zeros, zeros]
    
    # ReturnToBaseActor
    elif 'return_prob' in actor_output and 'wayPoint' not in actor_output:
        prob = actor_output['return_prob']
        zeros = torch.zeros_like(prob)
        if is_multi_dim:
            # 多维逻辑
            prob = prob.view(-1, max_entity_len)
            zeros = zeros.view(-1, max_entity_len)
            action_parts = [prob, zeros, zeros, zeros, zeros]

        else:
            # 单维逻辑
            action_parts = [prob, zeros, zeros, zeros, zeros]
    
    # WayPointMoveActor
    # elif 'return_prob' in actor_output and 'wayPoint' in actor_output:
    #     wp = actor_output['wayPoint']
    #     if is_multi_dim:
    #         # 多维逻辑
    #         return_prob = actor_output['return_prob'].view(-1, max_entity_len)
    #         longitude = wp['longitude'].view(-1, max_entity_len)
    #         latitude = wp['latitude'].view(-1, max_entity_len)
    #         # 对高度的参数进行量化
    #         altitude = wp['altitude'].view(-1, max_entity_len)
    #         altitude=quantize_altitude(altitude)
    #         # 对速度的参数进行量化
    #         velocity = wp['velocity'].view(-1, max_entity_len)
    #         velocity=quantize_speed(velocity)
    #         action_parts = [return_prob, longitude, latitude, altitude, velocity]

    #     else:
    #         # 单维逻辑
    #         action_parts = [
    #             actor_output['return_prob'],
    #             wp['longitude'], 
    #             wp['latitude'], 
    #             quantize_altitude(wp['altitude']), 
    #             quantize_speed(wp['velocity'])
    #         ]
    elif 'return_prob' in actor_output and 'wayPoint' in actor_output and 'direction_index' in actor_output['wayPoint']:
        wp = actor_output['wayPoint']

        if is_multi_dim:
            return_prob = actor_output['return_prob'].view(-1, max_entity_len)
            direction = wp['direction_index'].float().view(-1, max_entity_len)
            zero = wp['ZERO'].view(-1, max_entity_len)
            altitude = quantize_altitude(wp['altitude'].view(-1, max_entity_len))
            velocity = quantize_speed(wp['velocity'].view(-1, max_entity_len))
        else:
            return_prob = actor_output['return_prob']
            direction = wp['direction_index'].float()
            zero = wp['ZERO']
            altitude = quantize_altitude(wp['altitude'])
            velocity = quantize_speed(wp['velocity'])
        action_parts = [return_prob, direction, zero, altitude, velocity]


    # MobilityActor
    elif 'mobility_prob' in actor_output:
        
        speed = actor_output['speed']
        altitude = actor_output['altitude']
        prob = actor_output['mobility_prob']
        zeros = torch.zeros_like(speed)

        if is_multi_dim:
            # 多维逻辑
            prob = prob.view(-1, max_entity_len)
            speed = speed.view(-1, max_entity_len)
            # 将speed映射到0-4级   完全停止 游荡 巡航  满速  全速
            speed= quantize_speed(speed) 

            #将altitude映射到0-5级  0-5级
            # 飞机高度  
            # 最低海拔高度 
            # 低海拔高度(1000英尺) 305米
            # 低海拔高度(2000英尺) 610米
            # 中海拔高度(12000英尺) 3657米
            # 高海拔高度(25000英尺) 7620米
            # 最高海拔高度


            #潜艇深度
            # 水面
            # 潜望深度, 指潜艇潜望镜（其他升降装置）顶部高出水面0.5～1.0米的下潜深度， 常规潜艇艇体离水面大约为7—10米，核潜艇为9—15米
            # 浅潜望深度
            # 水温跃变层之上
            # 水温跃变层之下
            # 最大深度
            altitude = altitude.view(-1, max_entity_len)
            altitude=quantize_altitude(altitude)
            zeros = zeros.view(-1, max_entity_len)

            action_parts = [prob, speed, altitude, zeros, zeros]

        else:
            # 单维逻辑
            action_parts = [prob, quantize_speed(speed), quantize_altitude(altitude), zeros, zeros]
    
    # AttackTargetActor
    elif 'attack_prob' in actor_output:
        target_id = actor_output['selected_index'].float()  # 转换为float类型
        prob = actor_output['attack_prob']
        zeros = torch.zeros_like(prob)
        if is_multi_dim:
            # 多维逻辑
            prob = prob.view(-1, max_entity_len)
            target_id = target_id.view(-1, max_entity_len)
            zeros = zeros.view(-1, max_entity_len)

            action_parts = [prob, target_id, zeros, zeros, zeros]
        else:
            # 单维逻辑
            action_parts = [prob, target_id, zeros, zeros, zeros]
    
    # SensorControlActor
    elif 'SensorControl_prob' in actor_output:
        radar = actor_output['radarOperationStatus']
        sonar = actor_output['sonarOperationStatus']
        ecm = actor_output['ecmOperationStatus']
        prob = actor_output['SensorControl_prob']
        zero = torch.zeros_like(radar)
        
        if is_multi_dim:
            # 多维逻辑
            prob = prob.view(-1, max_entity_len)
            radar = radar.view(-1, max_entity_len)
            sonar = sonar.view(-1, max_entity_len)
            ecm = ecm.view(-1, max_entity_len)
            zero = zero.view(-1, max_entity_len)

            action_parts = [prob, radar, sonar, ecm, zero]
        else:
            # 单维逻辑
            action_parts = [prob, radar, sonar, ecm, zero]
    
    # DeploySonobuoyActor
    elif 'deploy_prob' in actor_output:
        type_p = actor_output['passiveOrActive']
        depth_p = actor_output['shallowOrDeep']
        prob = actor_output['deploy_prob']
        zero = torch.zeros_like(type_p)

        if is_multi_dim:
            # 多维逻辑
            prob = prob.view(-1, max_entity_len)
            type_p = type_p.view(-1, max_entity_len)
            depth_p = depth_p.view(-1, max_entity_len)
            zero = zero.view(-1, max_entity_len)
            action_parts = [prob, type_p, depth_p, zero, zero]
        else:
            # 单维逻辑
            action_parts = [prob, type_p, depth_p, zero, zero]
    
    # CancelAttackActor
    elif 'CancelAttack_prob' in actor_output:
        prob = actor_output['CancelAttack_prob']
        zeros = torch.zeros_like(prob)
        if is_multi_dim:
            # 多维逻辑
            prob = prob.view(-1, max_entity_len)
            zeros = zeros.view(-1, max_entity_len)
            action_parts = [prob, zeros, zeros, zeros, zeros]
        else:
            # 单维逻辑
            action_parts = [prob, zeros, zeros, zeros, zeros]
    
    else:
        raise ValueError("Unsupported actor output format")
    
    # 如果 action_parts 不为空，返回堆叠后的张量
    if action_parts:
        if is_multi_dim:
            # 多维逻辑：将 action_parts 堆叠到最后一个维度

            return torch.stack(action_parts, dim=-1)  # 形状为 [batch_size, max_entity_len, 5]
        else:
            # 单维逻辑：将 action_parts 堆叠到第二维度
            return torch.stack(action_parts, dim=1)  # 形状为 [batch_size, 5]
    else:
        return None


import  pdb


def nanmean(tensor, dim=None, keepdim=False):
    # 将 NaN 替换为 0，并计算有效值的数量
    mask = ~torch.isnan(tensor)
    tensor_masked = torch.where(mask, tensor, torch.zeros_like(tensor))
    # count = mask.sum(dim=dim, keepdim=keepdim)
    count = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1)
    # 把所有小于 1 的位置强行设为 1；

    
    # 计算均值
    mean = tensor_masked.sum(dim=dim, keepdim=keepdim) / count
    return mean

def nanstd(tensor, dim=None, keepdim=False):
    # 计算均值
    mean = nanmean(tensor, dim=dim, keepdim=True)
    
    # 计算方差
    mask = ~torch.isnan(tensor)
    tensor_masked = torch.where(mask, tensor, torch.zeros_like(tensor))
    variance = nanmean((tensor_masked - mean) ** 2, dim=dim, keepdim=keepdim)
    variance = torch.clamp(variance, min=0.0)

    # 计算标准差
    std = torch.sqrt(variance)
    return std

def feature_zscore(matrix, mask):
    """
    对矩阵进行 Z-score 归一化，排除无意义的实体。
    
    Args:
        matrix (torch.Tensor): 输入数据，形状为 (batch_size, num_entities, num_features)。
        mask (torch.Tensor): 掩码，形状为 (batch_size, num_entities)，指示哪些实体是有效的。
    
    Returns:
        torch.Tensor: 归一化后的矩阵，形状与输入相同。
    """
    if len(matrix.shape) != 3 or len(mask.shape) != 2:
        raise ValueError("Invalid input shape: matrix must be 3D and mask must be 2D")
    
    # 将无效实体的特征值设置为 NaN，以便在计算均值和标准差时忽略它们
    masked_matrix = torch.where(mask.unsqueeze(-1).expand_as(matrix), matrix, torch.full_like(matrix, float('nan')))
    
    # 计算均值和标准差，忽略 NaN 值
    mean = nanmean(masked_matrix, dim=1, keepdim=True)  # 沿实体维度计算均值
    std = nanstd(masked_matrix, dim=1, keepdim=True)    # 沿实体维度计算标准差
    
    # 处理标准差为 0 的情况，避免除以 0
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    
    # 归一化
    normalized_matrix = (matrix - mean) / std
    
    # 将无效实体的归一化结果置为 0（或其他默认值）
    normalized_matrix = torch.where(mask.unsqueeze(-1).expand_as(matrix), normalized_matrix, torch.zeros_like(matrix))
    
    return normalized_matrix

# 将可执行指令的实体通过actor的输出转化为动作空间向量后，按行stack，再特征归一化，最后通过action_encoder编码成[batch_size, 128]
# 主要作用是将不同数量实体的动作序列编码为固定维度的特征向量。适用于处理战场单位、机器人集群等场景中数量不定的实体动作数据。
# 初始化(ax_entities=5, action_dim=5)
# input 是action_data:[batch,entity_number(可以操作的实体),dim=5], action_data_mask[entity,1]
# out 是 [batch,128]
class ActionEncoder(nn.Module):
    def __init__(self, 
                 max_entity_len=50,
                 action_dim=5,
                 hidden_dim=128,
                 nhead=4,
                 num_layers=2):
        super().__init__()
        self.max_entity_len = max_entity_len
        
        self.norm = feature_zscore
        
        self.base_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.position_embed = nn.Embedding(max_entity_len, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, action_matrix, action_matrix_mask):
        """
        :param action_matrix: [batch_size, max_entity_len, action_dim]
        :param action_matrix_mask: [batch_size, max_entity_len]
        """
        batch_size, _, _ = action_matrix.shape

        normalized = self.norm(action_matrix, action_matrix_mask)
        
        padded_matrix, padding_mask = self._pad_with_mask(normalized)
        expanded_action_matrix_mask = torch.zeros_like(padding_mask, dtype=torch.bool)  
        expanded_action_matrix_mask[:, :action_matrix_mask.size(1)] = action_matrix_mask 
        inverted_action_matrix_mask = ~expanded_action_matrix_mask
        padding_mask = inverted_action_matrix_mask | padding_mask
                
        encoded = self.base_encoder(padded_matrix)
        
        positions = torch.arange(self.max_entity_len, 
                               device=action_matrix.device).expand(batch_size, -1)
        pos_emb = self.position_embed(positions)
        
        transformer_out = self.transformer_encoder(
            src=encoded + pos_emb,
            src_key_padding_mask=padding_mask
        )
        
        pooled = self._masked_pool(transformer_out, padding_mask)
        
        return self.final_proj(pooled)

    def _pad_with_mask(self, x):
        batch_size, num_ent, feat_dim = x.shape
        pad_size = self.max_entity_len - num_ent
        
        if pad_size > 0:
            padding = torch.zeros(batch_size, pad_size, feat_dim,
                                device=x.device, dtype=x.dtype)
            padded = torch.cat([x, padding], dim=1)
            
            mask = torch.zeros(batch_size, self.max_entity_len, 
                             dtype=torch.bool, device=x.device)
            mask[:, num_ent:] = True
        else:
            padded = x[:, :self.max_entity_len, :]
            mask = torch.zeros(batch_size, self.max_entity_len,
                             dtype=torch.bool, device=x.device)
            
        return padded, mask

    def _masked_pool(self, x, mask):
        weights = (~mask).unsqueeze(-1).to(x.dtype)  # [batch, max_ent, 1]
        
        weighted_sum = torch.sum(x * weights, dim=1)  # [batch, hidden_dim]
        
        # 如果整条序列全是 Padding（例如某一帧我方没有可操作实体），weights 全 0，
        # valid_counts==0；虽然后面 clamp(min=1.0)，但 weighted_sum 仍全 0，
        # 导致梯度永远 0，LayerNorm 里分母变 0 以后历史梯度里会出现 NaN。
        #(~mask)[i,j] = True 表示第 i 个样本的第 j 个实体是有效的（不是 Padding）。
        # (~mask).sum(1) 每个元素表示该样本“有效实体”的个数。
        # 每个元素表示该样本“有效实体”的个数。
        if (~mask).sum(1).eq(0).any():   
            x = x.clone()
            x[mask.all(-1)] = 0
        valid_counts = torch.sum(weights, dim=1)     # [batch, 1]

        valid_counts = torch.clamp(valid_counts, min=1.0)
        
        return weighted_sum / valid_counts


































# 测试代码
def test():
    # 正确输入格式：[batch_size=1, max_entity_len=5, action_dim=5]
    # 这个要改成每个batch有 不同数量的实体 的输入
    action_data = torch.tensor([[
        [0.5, 0.2, 8, 0.0, 0.0],   # 实体1
        [0.8, 0.9, 11, 0.0, 0.0],  # 实体2
        [0.9, 0.5, 18, 0.0, 0.0],  # 实体3
        [0.9, 0.5, 18, 0.0, 0.0]

    ],
    [
        [0.5, 0.2, 7, 0.0, 0.0],   # 实体1
        [0.8, 0.9, 11, 0.0, 0.0],  # 实体2
        [0.9, 0.5, 10, 0.0, 0.0],  # 实体3
        [0, 0, 0, 0, 0]

    ]], dtype=torch.float32)

    action_data_mask = torch.tensor([
        [1, 1, 1, 1],  # 实体1, 实体2, 实体3, 实体4
        [1, 1, 1, 0]   # 实体1, 实体2, 实体3, 实体4
    ], dtype=torch.bool)


    
    encoder = ActionEncoder(max_entity_len=5, action_dim=5)
    output = encoder(action_data, action_data_mask)
    print(f"编码结果维度: {output.shape}")  # 正确输出 torch.Size([1, 128])

# TEST
# [batch_size, max_entity_len, dim]
def test_actor(actor_class, input_shape, max_entity_len, extra_input=None):
    import random
    batch_size = 2
    actor = actor_class()
    
    # 生成模拟输入，形状为 [batch_size, max_entity_len, dim]
    state = torch.randn(batch_size, max_entity_len, *input_shape)

    # 特殊处理AttackTargetActor的extra_input
    if actor_class == AttackTargetActor:
        # 保留原始extra_input的形状，但动态生成有效掩码
        _, num_targets, feat_dim = extra_input.shape  # 解包原始形状

        # rint = random.randint(1, num_targets)
        rint = 2
        # 生成动态目标掩码（每帧场景随机保留1~num_targets个有效目标）
        target_masks = torch.stack([
            torch.cat([
                torch.ones(rint, dtype=torch.bool),
                torch.zeros(num_targets - rint, dtype=torch.bool)
            ]) 
            for _ in range(batch_size)
        ]).view(batch_size, num_targets)
        
        # 对extra_input应用动态掩码
        masked_input = extra_input * target_masks.unsqueeze(-1)
        inputs = (state, masked_input, target_masks)
    else:
        inputs = (state, extra_input) if extra_input is not None else (state,)
    

    # 执行前向传播
    output = actor(*inputs)
    # 转换动作

    action = actor_output_to_action(output, max_entity_len=max_entity_len)
    pdb.set_trace()
    
    # 验证维度，期望输出形状为 [batch_size, max_entity_len, action_dim]
    assert action.shape == (batch_size, max_entity_len, 5), f"维度错误: {action.shape}"
    print(f"{actor_class.__name__} 测试通过，输出维度: {action.shape}")

def test1():
    max_entity_len = 10  # 假设最大实体数量为10
    # AircraftTakeOffActor测试
    # test_actor(AircraftTakeOffActor, (512,), max_entity_len)

    # ReturnToBaseActor测试
    # test_actor(ReturnToBaseActor, (512,), max_entity_len)

    # WayPointMoveActor测试
    test_actor(WayPointMoveActor, (512,), max_entity_len)

    # MobilityActor测试
    # test_actor(MobilityActor, (512,), max_entity_len)

    # AttackTargetActor测试（需要额外输入）
    # test_actor(AttackTargetActor, (512,), max_entity_len, extra_input=torch.randn(2, 7, 256))

    # SensorControlActor测试
    # test_actor(SensorControlActor, (512,), max_entity_len)

    # DeploySonobuoyActor测试
    # test_actor(DeploySonobuoyActor, (512,), max_entity_len)


    # CancelAttackActor测试
    # test_actor(CancelAttackActor, (512,), max_entity_len)


if __name__ == "__main__":

    test1()
