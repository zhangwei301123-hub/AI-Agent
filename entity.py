import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import json
from typing import List, Dict

from execute import *

# 打算按照reportTime.json 保存json文件
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
Aircraft_type={
     "Aircraft_Fighter" : 0 ,# ;//	战斗机
     "Aircraft_Multirole" : 1,# ;//	多用途（战斗/攻击）
     "Aircraft_ASAT" : 2,# ;//	反卫星武器 （反卫星武器）
     "Aircraft_AirborneLaserPlatform" : 3,# ;//	机载激光平台
     "Aircraft_Attack" : 4,# ;//	攻击机
     "Aircraft_WildWeasel" : 5,# ;//	野鼬鼠 （防空压制）
     "Aircraft_Bomber" : 6,# ;//	轰炸机
     "Aircraft_CAS" : 7,# ;//	战场空中拦截（BAI/ CAS）
     "Aircraft_OECM" : 8,# ;//	电子战
     "Aircraft_AEW" : 9,# ;//	早期空中预警 （AEW）
     "Aircraft_AirborneCP" : 10,# ;//	机载指挥站 （ACP）
     "Aircraft_SAR" : 11,# ;//	搜索及救援 （SAR）
     "Aircraft_MCM" : 12,# ;//	反水雷舰 （MCM）
     "Aircraft_ASW" : 13,# ;//	反潜作战 （ASW）
     "Aircraft_MPA" : 14,# ;//	海上巡逻机 （MPA）
     "Aircraft_ForwardObserver" : 15,# ;//	前进观察员
     "Aircraft_AreaSurveillance" : 16,# ;//	区域监视
     "Aircraft_Recon" : 17,# ;//	侦查
     "Aircraft_ELINT" : 18,# ;//	电子情报收集 （ELINT）
     "Aircraft_SIGINT" : 19,# ;//	信号情报收集 （SIGINT）
     "Aircraft_Transport" : 20,# ;//	运输机
     "Aircraft_Cargo" : 21,# ;//	货机
     "Aircraft_Commercial" : 22,# ;//	商业飞机
     "Aircraft_Civilian" : 23,# ;//	民用飞机
     "Aircraft_Utility" : 24,# ;//	通用直升机
     "Aircraft_Utility_Naval" : 25,# ;//	海军通用直升机
     "Aircraft_Tanker" : 26,# ;//	加油机 （空中加油）
     "Aircraft_Trainer" : 27,# ;//	教练机
     "Aircraft_TargetTowing" : 28,# ;//	牵引机
     "Aircraft_TargetDrone" : 29,# ;//	靶机
     "Aircraft_UAV" : 30,# ;//	无人机（UAV）
     "Aircraft_UCAV" : 31,# ;//	无人作战飞行器（UCAV）
     "Aircraft_AirShip" : 32,# ;//	飞艇
     "Aircraft_Aerostat" : 33,# ;//	航空器
     "Aircraft_IMGSAT" : 34,# ;//
     "Aircraft_RORSAT" : 35,# ;//
     "Aircraft_EORSAT" : 36,
 }
OIL_STATUS={
    'IS_BINGO': 1,
    'IS_JOKER': 2,
    'NONE': 3,
}
    # Aircraft_Fighter = 0 ,# ;//	战斗机
    # Aircraft_Multirole = 1;//	多用途（战斗/攻击）
    # Aircraft_ASAT = 2;//	反卫星武器 （反卫星武器）
    # Aircraft_AirborneLaserPlatform = 3;//	机载激光平台
    # Aircraft_Attack = 4;//	攻击机
    # Aircraft_WildWeasel = 5;//	野鼬鼠 （防空压制）
    # Aircraft_Bomber = 6;//	轰炸机
    # Aircraft_CAS = 7;//	战场空中拦截（BAI/ CAS）
    # Aircraft_OECM = 8;//	电子战
    # Aircraft_AEW = 9;//	早期空中预警 （AEW）
    # Aircraft_AirborneCP = 10;//	机载指挥站 （ACP）
    # Aircraft_SAR = 11;//	搜索及救援 （SAR）
    # Aircraft_MCM = 12;//	反水雷舰 （MCM）
    # Aircraft_ASW = 13;//	反潜作战 （ASW）
    # Aircraft_MPA = 14;//	海上巡逻机 （MPA）
    # Aircraft_ForwardObserver = 15;//	前进观察员
    # Aircraft_AreaSurveillance = 16;//	区域监视
    # Aircraft_Recon = 17;//	侦查
    # Aircraft_ELINT = 18;//	电子情报收集 （ELINT）
    # Aircraft_SIGINT = 19;//	信号情报收集 （SIGINT）
    # Aircraft_Transport = 20;//	运输机
    # Aircraft_Cargo = 21;//	货机
    # Aircraft_Commercial = 22;//	商业飞机
    # Aircraft_Civilian = 23;//	民用飞机
    # Aircraft_Utility = 24;//	通用直升机
    # Aircraft_Utility_Naval = 25;//	海军通用直升机
    # Aircraft_Tanker = 26;//	加油机 （空中加油）
    # Aircraft_Trainer = 27;//	教练机
    # Aircraft_TargetTowing = 28;//	牵引机
    # Aircraft_TargetDrone = 29;//	靶机
    # Aircraft_UAV = 30;//	无人机（UAV）
    # Aircraft_UCAV = 31;//	无人作战飞行器（UCAV）
    # Aircraft_AirShip = 32;//	飞艇
    # Aircraft_Aerostat = 33;//	航空器
    # Aircraft_IMGSAT = 34;//
    # Aircraft_RORSAT = 35;//
    # Aircraft_EORSAT = 36;


ATTACK_CAPABLE_AIRCRAFT_TYPES = {0, 1, 2, 4, 6, 7, 13, 14}

MAX_ENTITIES = 700

MAX_ACTION_ENTITIES = 400

MAX_TARGETS = 300

#对读取到的json 进行编码 获得 encoded,mask
# encoded.shape()= [know_entities,38]
# mask.shape()=[know_entities] [1,1,1,1] 有几个实体，就在该位置上设为1

def normalize_entity_type(mdl_type):
    """将不同态势接口的实体类型名称统一为 MADDPG 使用的类型编号。"""
    value = str(mdl_type or '').strip().upper()

    # 导弹名称可能同时包含 AIR/SURFACE，必须优先于平台类型判断。
    if any(alias in value for alias in ('WEAPON', 'MISSILE', 'TORPEDO', 'MUNITION')) \
            or any(alias in value for alias in ('导弹', '鱼雷', '武器')):
        return 4
    # SUBSURFACE 包含 SURFACE，必须先判断水下平台。
    if any(alias in value for alias in ('SUBMARINE', 'SUBSURFACE')) \
            or any(alias in value for alias in ('潜艇', '水下舰艇')):
        return 2
    if value == 'AIR' or 'AIRCRAFT' in value \
            or any(alias in value for alias in ('飞机', '航空器')):
        return 0
    if any(alias in value for alias in ('SHIP', 'SURFACE', 'VESSEL', 'CARRIER')) \
            or any(alias in value for alias in ('水面舰艇', '舰船', '军舰', '航母')):
        return 1
    if value == 'LAND' or 'FACILITY' in value \
            or any(alias in value for alias in ('地面设施', '设施')):
        return 3
    return 5


class EntityEncoder:
    '''
    对单个实体进行编码
    '''
    
    def __init__(self, max_entities=MAX_ENTITIES):
        self.max_entities = max_entities
        # 按需求添加 modify
        self.feature_dim = 38  # 保持38个特征


    # 相当与setup()
    def encode(self, raw_entities):
        '''
        对单个实体进行编码
        得到 encoded, mask
        encoded.shape()=[max_entities, feature_dim]  本实验用38个特征表示
        mask.shape()=[max_entities,]
'''
        encoded = np.full((self.max_entities, self.feature_dim), -1, dtype=np.float32)
        mask = np.zeros(self.max_entities)
        entity_count = 0
        for entity in raw_entities:
            if entity_count >= self.max_entities:
                break

            # [0] 阵营信息
            string_forceSide = entity.get('forceSide')
            if string_forceSide == '蓝方':
                encoded[entity_count, 0] = 0
            elif string_forceSide == '红方':
                encoded[entity_count, 0] = 1
            elif string_forceSide == 'c':
                encoded[entity_count, 0] = 2
            else:
                encoded[entity_count, 0] = -1

            # [1] 血量
            encoded[entity_count, 1] = entity.get('activeLvl', -1)

            # [2-4] 姿态角（俯仰/翻滚/偏航）
            attitude = entity.get('attitude', {})
            encoded[entity_count, 2] = attitude.get('pitch', -1)
            encoded[entity_count, 3] = attitude.get('roll', -1)
            encoded[entity_count, 4] = attitude.get('yaw', -1)

            # [5-7] 空间坐标（高度/纬度/经度）
            spatial_coord = entity.get('entitySpatialCoord', {})
            encoded[entity_count, 5] = spatial_coord.get('altitude', -1)
            encoded[entity_count, 6] = spatial_coord.get('latitude', -1)
            encoded[entity_count, 7] = spatial_coord.get('longitude', -1)

            # [8] 基地关联
            encoded[entity_count, 8] = 1 if 'AirBase' in entity.get('attrMap', {}) else -1

            # [9] 燃油状态
            logistic = entity.get('logisticStates', {}).get('oilStatus', -1)
            if logistic in OIL_STATUS.keys():
                encoded[entity_count, 9] = OIL_STATUS[logistic]

            # [10-12] 速度分量（vx/vy/vz）
            velocity = entity.get('velocity', {})
            encoded[entity_count, 10] = velocity.get('vx', -1)
            encoded[entity_count, 11] = velocity.get('vy', -1)
            encoded[entity_count, 12] = velocity.get('vz', -1)

            # [13] 实体类型（0=飞机，1=舰船，2=潜艇，3=设施，4=武器，5=其他）
            encoded[entity_count, 13] = normalize_entity_type(entity.get('mdlType'))

            # [14] 干扰状态
            inner = entity.get('innerstates', {})
            encoded[entity_count, 14] = 1 if inner.get('IsJamReaction', False) else -1

            # [15] 目标丢失时间
            encoded[entity_count, 15] = inner.get('lostTime', -1)

            # [16-17] 武器挂载 给一个有攻击性的 和无攻击性的武器数量 
            # 按需求添加 modify
            load_map = entity.get('loadMap', {})
            encoded[entity_count, 16] = load_map.get('offensive', 0)
            encoded[entity_count, 17] = load_map.get('offenseless', 0)

            # [18] 实体序列ID
            encoded[entity_count, 18] = entity_count 

            # [19] 报告时间戳
            encoded[entity_count, 19] = entity.get('reportTime', -1)

            # [20-21] 状态信息
            state_map = entity.get('stateMap', {})
            encoded[entity_count, 20] = float(state_map.get('FuelBurnRate', -1))
            encoded[entity_count, 21] = float(state_map.get('RemainDistance', -1))

            # [22-23] 单位状态
            encoded[entity_count, 22] = unit_status_mapping.get(state_map.get('UnitStatus', ''), -1)
            encoded[entity_count, 23] = AIR_STATUS_MAP.get(state_map.get('AirStatus', ''), -1)

            # [24-27] 传感器状态（直接获取整数值）
            encoded[entity_count, 24] = int(state_map.get('EcmStatus', -1))
            encoded[entity_count, 25] = int(state_map.get('RadarStatus', -1))
            encoded[entity_count, 26] = int(state_map.get('SonarStatus', -1))
            encoded[entity_count, 27] = int(state_map.get('IdentifyStatus', -1))
            
            # [28] 如果是敌方实体 该值为是否被攻击 1为被攻击 0为未攻击 -1为无意义
            encoded[entity_count, 28] = int(state_map.get('IsUnderAttack', -1))

            # [29-32] 最大攻击距离
            max_range = entity.get('maxRange', {})
            encoded[entity_count, 29] = max_range.get('maxAir', -1)
            encoded[entity_count, 30] = max_range.get('maxSubsurface', -1)
            encoded[entity_count, 31] = max_range.get('maxSurface', -1)
            encoded[entity_count, 32] = max_range.get('maxLand', -1)

            # 飞机 战斗机  巡逻机 预警机
            # 船 巡洋舰 航母 驱逐舰
            # 具体的实施细节见proto的enum UnitCategory
            # 有攻击的飞机型号：0，1，2,4,6，7,13,
            # 打雷达的飞机型号：5 
            # 无攻击的飞机型号：8,9,10,14,17,18,19,20,21,22,
            # [33] 实体具体类型
            if encoded[entity_count, 13] == 0 :
                if entity.get('unitCategory') in Aircraft_type.keys():
                    encoded[entity_count, 33] = int(Aircraft_type[entity.get('unitCategory')])

            wpn_info = entity.get('weaponNumber', {})
            wpn_air_left = wpn_info.get('airNum', 0)
            wpn_sub_left = wpn_info.get('subNum', 0)
            wpn_ship_left = wpn_info.get('shipNum', 0)

            encoded[entity_count, 34] = int(wpn_air_left)
            encoded[entity_count, 35] = int(wpn_sub_left)
            encoded[entity_count, 36] = int(wpn_ship_left)

            # [34] 是否可以操纵
            if not get_is_train():
                encoded[entity_count, 37] = int(entity.get('isCanManaged',True))
            else:
                encoded[entity_count, 37] = 1

            mask[entity_count] = 1
            entity_count += 1

        return encoded, mask


# 编码每个全局特征  相当于整个的readout
# [batch, max_ent, 30] -> [batch, 256]
class GlobalStateEncoder(nn.Module):

    """全局状态编码器
    编码每个全局特征  相当于整个的readout
 [batch, max_ent, 37] -> [batch, 256]
    """
    def __init__(self, entity_dim=37, hidden_dim=256, max_entities=MAX_ENTITIES):
        super().__init__()
        # 实体级编码
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1)
        )
        
        # 注意力聚合层
        # self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4,batch_first=True,dropout=0.1)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4,batch_first=True, dropout=0.1)
        
        # 全局特征提取
        self.global_encoder = nn.Sequential(
            nn.Linear(max_entities*128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化防止梯度爆炸"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, entities, mask):
        batch_size = entities.shape[0]
        # 编码每个实体特征 [batch, max_ent, 30] -> [batch, max_ent, 128]
        ent_feat = self.entity_encoder(entities)
        # 注意力聚合[batch, max_ent, 128]
        attn_output, _ = self.attention(
            query=ent_feat,
            key=ent_feat,
            value=ent_feat,
            key_padding_mask=~mask
        )
        # attn_output = torch.nan_to_num(attn_output, nan=0)
        # 全局特征 [batch, hidden_dim]
        global_feat = self.global_encoder(attn_output.reshape(batch_size, -1))
        return attn_output, global_feat # 从attn_output中拿出指定的实体特征（根据顺序）
    
# 编码每个entity特征  相当于整个的readout
# [batch, 256]
class SelfStateEncoder(nn.Module):
    """自身状态编码器
    # 编码每个entity特征  相当于整个的readout
    # [batch, 256]
    """

    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()

        self.input_dim = input_dim

        self.self_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, from_global_feat):
        return self.self_encoder(from_global_feat) # [batch, hidden_dim]


def is_can_attack(
    encoded_data,
    mask,
    target_side=1,
    allowed_types=[0, 1],
    require_manageable=None,
):
    """
    返回【本帧】允许执行 AttackTargetActor 的实体索引列表
    - 只有目标阵营 (target_side) 的实体才会被考虑
    - 大类型必须在 allowed_types (第 13 列) 中
    - 飞机还需满足“具体类型”(第 33 列) ∈ ATTACK_CAPABLE_AIRCRAFT_TYPES
    - 必须携带进攻性武器 (第 16 列 > 0)
    """
        
    if allowed_types is None:
        allowed_types = [0, 1, 2]

    if require_manageable is None:
        # 兼容旧调用：旧工程默认控制蓝方(0)。新实时链路会显式传参。
        require_manageable = target_side == 0

    can_attack = []
    for i in range(encoded_data.shape[0]):
        if mask[i] == 1:
            if require_manageable and encoded_data[i, -1] == 0:
                continue
            force_side = int(encoded_data[i, 0])
            entity_type = int(encoded_data[i, 13])  # 大类型
            attack_weapon_num = int(encoded_data[i, 16])

            if force_side == target_side and entity_type in allowed_types and attack_weapon_num > 0:
                # if entity_type == 0:                     # 0 = Aircraft
                #     spec_type = int(encoded_data[i, 33])
                    # if spec_type not in ATTACK_CAPABLE_AIRCRAFT_TYPES:
                    #     continue  # 无攻击能力 -> 跳过
                can_attack.append(i)
    
    return can_attack


def get_env_entity_ids(
    encoded_data,
    mask,
    target_side=1,
    allowed_types=[0, 1],
    think_hp=False,
    require_manageable=None,
):
    '''
     获取指定阵营、类型的实体ID列表 还得限制血量为正数
     将EntityEncoder的30个数据（EntityEncoder） 获得A阵营的所有实体的索引
    '''
    if allowed_types is None:
        allowed_types = [0, 1]
    if require_manageable is None:
        # 兼容旧调用：旧工程默认控制蓝方(0)。新实时链路会显式传参。
        require_manageable = target_side == 0

    filtered_index = []
    for i in range(encoded_data.shape[0]):
        if mask[i] == 1:
            if require_manageable and encoded_data[i, -1] == 0:
                continue
            force_side = int(encoded_data[i, 0])
            hp = encoded_data[i, 1]
            entity_type = int(encoded_data[i, 13])
            if think_hp == True:
                if force_side == target_side and entity_type in allowed_types and hp > 0:
                    filtered_index.append(i)
            elif  think_hp == False:
                if force_side == target_side and entity_type in allowed_types:
                    filtered_index.append(i)
    
    
    return filtered_index


def get_velocity_from_encoded_data(encoded_data,
                                   mask,
                                   target_side=1,
                                   allowed_types=(0, 1, 2, 4),
                                   idxs=(10, 11, 12)):
    """
    提取指定阵营 + 指定类型实体的速度模长列表。
    
    参数
    ----
    encoded_data : np.ndarray | torch.Tensor
        实体特征矩阵，形状 [N, D]
    mask : np.ndarray | torch.Tensor
        实体有效掩码，形状 [N]，1 表有效
    target_side : int
        目标阵营（0=我方，1=敌方）
    allowed_types : Tuple[int, ...]
        允许的实体类型编号
    idxs : Tuple[int, int, int]
        vx, vy, vz 在特征向量中的列索引
        
    返回
    ----
    speeds : List[float]
        满足过滤条件的实体速度模长
    """
    # 若传入 torch.Tensor，先转为 CPU，保持与 numpy 使用一致
    if hasattr(encoded_data, 'cpu'):
        encoded_np = encoded_data.cpu().detach().numpy()
        mask_np    = mask.cpu().detach().numpy()
    else:
        encoded_np = encoded_data
        mask_np    = mask

    speeds = []
    for i in range(encoded_np.shape[0]):
        if mask_np[i] != 1:
            continue

        force_side  = int(encoded_np[i, 0])    # 阵营
        entity_type = int(encoded_np[i, 13])   # 类型

        if force_side == target_side and entity_type in allowed_types:
            vx, vy, vz = (float(encoded_np[i, idx]) for idx in idxs)
            speed = (vx**2 + vy**2 + vz**2) ** 0.5
            speeds.append(speed)

    return speeds


def get_coordinate_from_encoded_data(
    encoded_data,
    mask,
    target_side=1,
    allowed_types=[0, 1],
    require_manageable=None,
):
    """
    获取指定阵营、指定类型、血量>0 的实体的坐标列表
    输入:
        encoded_data: [N, D]，每个实体的特征编码
        mask: [N]，是否有效实体（1表示有效）
        target_side: 目标阵营（默认1表示敌方）
        allowed_types: 实体类型过滤
    返回:
        coords: List[Tuple]，实体坐标 (lon, lat)
    """
    if require_manageable is None:
        # 兼容旧调用：旧工程默认控制蓝方(0)。新实时链路会显式传参。
        require_manageable = target_side == 0

    coords = []
    for i in range(encoded_data.shape[0]):
        if mask[i] == 1:
            if require_manageable and encoded_data[i, -1] == 0:
                continue
            force_side = int(encoded_data[i, 0])     # 阵营
            entity_type = int(encoded_data[i, 13])   # 类型
            if force_side == target_side and entity_type in allowed_types:
                lat = encoded_data[i, 6].item()
                lon = encoded_data[i, 7].item()
                coords.append((lon, lat))
    return coords


def get_unitCategory(encoded_data, mask, target_side=1, allowed_types=[0]): # 0=飞机，1=舰船
    """
    获取指定阵营、指定类型、血量>0 的实体的坐标列表
    输入:
        encoded_data: [N, D]，每个实体的特征编码
        mask: [N]，是否有效实体（1表示有效）
        target_side: 目标阵营（默认1表示敌方）
        allowed_types: 实体类型过滤
    返回:
        coords: List[Tuple]，实体坐标 (lon, lat)
    """
    unitCategory = []
    for i in range(encoded_data.shape[0]):
        if mask[i] == 1:
            force_side = int(encoded_data[i, 0])     # 阵营
            entity_type = int(encoded_data[i, 13])   # 类型
            if force_side == target_side and entity_type in allowed_types:
                unitCategory.append(int(encoded_data[i, 33]))
    return unitCategory

import math

def direction_bucket(lon_src, lat_src, lon_tgt, lat_tgt):
    """返回 0‑7 的方向象限索引：0=北, 1=东北 ... 7=西北"""
    ang = (math.degrees(math.atan2(lat_tgt - lat_src, lon_tgt - lon_src)) + 360) % 360
    return int((ang + 22.5) // 45)




# import pdb
# def test1():
#     pdb.set_trace()
#     file_path = 'json.json'
#     with open(file_path, 'r', encoding='utf-8') as f:
#         json_data = json.load(f)

#     encoder = EntityEncoder(max_entities=MAX_ENTITIES)
#     encoded_data, mask = encoder.encode(json_data)

# # 将selfstate和globalstate进行拼接，作为actor的输入
# def test2():
#     file_path = 'json.json'
#     with open(file_path, 'r', encoding='utf-8') as f:
#         json_data = json.load(f)

#     encoder = EntityEncoder(max_entities=MAX_ENTITIES)
#     encoded_np, mask_np = encoder.encode(json_data)
#     pdb.set_trace()
#     encoded_data = torch.FloatTensor(encoded_np).unsqueeze(0)
#     mask = torch.BoolTensor(mask_np).unsqueeze(0)

#     global_state_encoder = GlobalStateEncoder(entity_dim=30, hidden_dim=256, max_entities=MAX_ENTITIES)
#     self_state_encoder = SelfStateEncoder(input_dim=128, hidden_dim=256)

#     filtered_index = get_env_entity_ids(encoded_np, mask_np, target_side=1, allowed_types=[0, 1])
#     attn_output, global_feat = global_state_encoder(encoded_data, mask)

#     for index in filtered_index:
#         self_state = self_state_encoder(attn_output[0][index]) # 没有batchsize不用[0]
#         fusion_result = torch.cat([self_state, global_feat[0]], dim=0) # 没有batchsize不用[0]
#         print(fusion_result)
#         pdb.set_trace()

# if __name__ == '__main__':
#     test1()
