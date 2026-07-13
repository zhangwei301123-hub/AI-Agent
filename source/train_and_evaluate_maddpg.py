# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# from maddpg_ import MADDPG, ACTOR_TYPES  # 假设你主程序叫 maddpg_.py


# # # 日志目录
# # log_dir = f'logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
# # os.makedirs(log_dir, exist_ok=True)

# # # 超参数
# batch_size = 4
# MAX_ENTITIES = 70
# MAX_ACTION_ENTITIES = 50
# MAX_TARGETS = 20
# NUM_OURS = 20
# NUM_ENEMIES = 10
# NUM_PADDING = MAX_ENTITIES - NUM_OURS - NUM_ENEMIES
# STATE_DIM = 29
# ACTION_DIM = 5
# NUM_AGENTS = len(ACTOR_TYPES)

# # # 初始化 MADDPG
# # maddpg = MADDPG(actor_types=ACTOR_TYPES, batch_size=batch_size)
# # loss_logs = {i: {'actor': [], 'critic': []} for i in range(NUM_AGENTS)}

# from entity import EntityEncoder  # 确保你已经实现 EntityEncoder 类
# import random

# def create_fake_state():
#     encoder = EntityEncoder(max_entities=MAX_ENTITIES)
#     raw_entities = []

#     for _ in range(NUM_ENEMIES):
#         raw_entities.append(create_fake_entity(_, "红方", "SHIP"))
#     for _ in range(NUM_OURS):
#         raw_entities.append(create_fake_entity(_, "蓝方", "AIRCRAFT"))
#     for _ in range(NUM_PADDING):
#         raw_entities.append(create_fake_entity(_, "c", "UNKNOWN"))  # padding

#     encoded, mask = encoder.encode(raw_entities)
#     return {
#         "encoded_data": torch.tensor(encoded, dtype=torch.float32),
#         "mask": torch.tensor(mask, dtype=torch.bool),
#     }

# # def create_fake_state():
# #     encoder = EntityEncoder(max_entities=MAX_ENTITIES)
# #     raw_entities = []

# #     # 敌方实体
# #     for _ in range(NUM_ENEMIES):
# #         raw_entities.append(create_fake_entity(entity_id=_, force_side="红方", mdl_type="SHIP"))

# #     # 我方实体
# #     for _ in range(NUM_OURS):
# #         raw_entities.append(create_fake_entity(entity_id=_, force_side="蓝方", mdl_type="AIRCRAFT"))

# #     # Padding
# #     for _ in range(NUM_PADDING):
# #         raw_entities.append(create_fake_entity(entity_id=_, force_side="c", mdl_type="UNKNOWN"))

# #     # 编码
# #     encoded, mask = encoder.encode(raw_entities)

# #     return {
# #         "encoded_data": torch.tensor(encoded, dtype=torch.float32),
# #         "mask": torch.tensor(mask, dtype=torch.bool),
# #     }


# # # # 主循环
# # # for step in range(1000):
# # #     for _ in range(batch_size * 2):
# # #         fake_state = create_fake_state()
# # #         fake_next_state = create_fake_state()
# # #         fake_actions = np.random.randn(NUM_AGENTS, MAX_ACTION_ENTITIES, ACTION_DIM).astype(np.float32)
# # #         fake_actions_mask = np.random.randint(0, 2, size=(NUM_AGENTS, MAX_ACTION_ENTITIES)).astype(bool)
# # #         fake_rewards = np.random.randn(NUM_AGENTS)
# # #         fake_dones = np.random.randint(0, 2)
# # #         fake_entity_ids = [f"ent_{i}" for i in range(MAX_ACTION_ENTITIES)]
# # #         fake_executed = np.random.randint(0, 2, size=(MAX_ACTION_ENTITIES, NUM_AGENTS))

# # #         maddpg.store_experience(
# # #             states=fake_state,
# # #             actions=fake_actions,
# # #             actions_mask=fake_actions_mask,
# # #             rewards=fake_rewards,
# # #             next_states=fake_next_state,
# # #             dones=fake_dones,
# # #             step=step,
# # #             action_entity_id=fake_entity_ids,
# # #             actions_executed=fake_executed
# # #         )

# # #     maddpg.update()

# # #     for idx, loss in enumerate(maddpg.agent_losses):
# # #         loss_logs[idx]['actor'].append(loss['actor'][0])
# # #         loss_logs[idx]['critic'].append(loss['critic'][0])

# # #     if step % 50 == 0 and step > 0:
# # #         print(f"\n✅ Step {step}: 保存模型与 loss 图...")
# # #         torch.save(maddpg, os.path.join(log_dir, f"maddpg_model_step{step}.pt"))
# # #         torch.save(maddpg, os.path.join(log_dir, f"maddpg_model_last.pt"))


# # import torch
# # import random
# # from actor import WayPointMoveActor  # 你可以替换为任意 actor
# # from maddpg_ import MADDPGAgent      # 确保这个类你已经定义好

# # # 设置超参数
# # MAX_ENTITY_LEN = 50
# # STATE_DIM = 512  # 你 actor 的输入 dim，通常是 encoder 输出维度
# # ACTION_DIM = 5

# # # 创建一个 agent（用 WayPointMoveActor 为例）
# # agent = MADDPGAgent(actor_type=WayPointMoveActor)

# # # 构造假的输入状态 [batch=1, max_entity_len=50, dim=512]
# # fake_state = create_fake_state()

# # # 调用 get_action 并触发 exploration
# # action = agent.get_action(state=fake_state, training=True)
# # print("动作 shape：", action.shape)

# import torch
# from maddpg_ import MADDPG, ACTOR_TYPES, MADDPGAgent
# from actor import AttackTargetActor



# def test_agent_action_single_step(agent: MADDPGAgent, maddpg: MADDPG, batch_size=1):
#     """
#     测试一个 agent 的 get_action 输出是否正常工作（包括探索）

#     参数:
#         agent: 某一个 MADDPGAgent
#         maddpg: 包含 encoder 的完整 MADDPG 实例
#         batch_size: 测试用的批量大小，默认 1
#     返回:
#         action: 输出的动作张量 [batch, max_entity_len, 5]
#     """
#     print(f"\n📦 正在测试 agent: {agent.actor.__class__.__name__}")

#     # 1. 构造假状态并编码为 actor 输入
#     fake_states = [create_fake_entity() for _ in range(batch_size)]
#     prepared = maddpg.prepare_state(fake_states)

#     actor_input, _, _, actor_mask, target_feat, target_mask = maddpg._encode_global_and_self_state(prepared)

#     # 2. 根据 actor 类型选择输入方式
#     if isinstance(agent.actor, AttackTargetActor):
#         action = agent.get_action(
#             (actor_input, target_feat, target_mask),
#             training=True
#         )
#     else:
#         action = agent.get_action(actor_input, training=True)

#     # 3. 打印动作结果
#     print(f"✅ 动作 shape: {action.shape}")  # [batch, max_entity_len, 5]
#     print("🎯 前几个动作:")
#     print(action[0, :5])  # 第一个样本的前5个动作

#     return action


# def test_all_agents():
#     """
#     测试所有 agent 的动作输出，确保 get_action 接口无误
#     """
#     # 初始化整体系统
#     maddpg = MADDPG(actor_types=ACTOR_TYPES, batch_size=1)

#     # 循环每个 agent 进行动作测试
#     for idx, agent in enumerate(maddpg.agents):
#         test_agent_action_single_step(agent, maddpg)


# if __name__ == "__main__":
#     test_all_agents()

import os
import torch
import numpy as np
import random
from datetime import datetime
from maddpg_ import MADDPG, ACTOR_TYPES  # 替换为你的主模型文件路径
from entity import EntityEncoder  # 替换为你的 EntityEncoder 类路径

# ---------------- 参数设置 ----------------
MAX_ENTITIES = 70
MAX_ACTION_ENTITIES = 50
MAX_TARGETS = 20
NUM_OURS = 20
NUM_ENEMIES = 10
NUM_PADDING = MAX_ENTITIES - NUM_OURS - NUM_ENEMIES
STATE_DIM = 30
ACTION_DIM = 5
BATCH_SIZE = 4
NUM_AGENTS = len(ACTOR_TYPES)

# ---------------- 模拟实体生成 ----------------
def create_fake_entity(entity_id, force_side, mdl_type):
    return {
        "forceSide": force_side,
        "activeLvl": random.uniform(0, 100),
        "attitude": {
            "pitch": random.uniform(-90, 90),
            "roll": random.uniform(-180, 180),
            "yaw": random.uniform(0, 360),
        },
        "entitySpatialCoord": {
            "altitude": random.uniform(0, 10000),
            "latitude": random.uniform(-90, 90),
            "longitude": random.uniform(-180, 180),
        },
        "attrMap": {"AirBase": True} if mdl_type == "AIRCRAFT" else {},
        "logisticStates": {
            "oil": random.uniform(0, 1000),
        },
        "velocity": {
            "vx": random.uniform(-1000, 1000),
            "vy": random.uniform(-1000, 1000),
            "vz": random.uniform(-1000, 1000),
        },
        "mdlType": mdl_type,
        "innerstates": {
            "IsJamReaction": random.choice([True, False]),
            "lostTime": random.uniform(0, 300),
        },
        "loadMap": {
            "offensive": random.randint(0, 5),
            "offenseless": random.randint(0, 5),
        },
        "reportTime": random.randint(0, 10000),
        "stateMap": {
            "FuelBurnRate": random.uniform(0, 10),
            "RemainDistance": random.uniform(0, 1000),
            "UnitStatus": "Normal",
            "AirStatus": "Flying",
            "EcmStatus": 1,
            "RadarStatus": 1,
            "SonarStatus": 0,
            "IdentifyStatus": 2,
            "IsUnderAttack": random.choice([0, 1]),
        }
    }

def create_fake_state():
    encoder = EntityEncoder(max_entities=MAX_ENTITIES)
    raw_entities = []

    for _ in range(NUM_ENEMIES):
        raw_entities.append(create_fake_entity(_, "红方", "SHIP"))
    for _ in range(NUM_OURS):
        raw_entities.append(create_fake_entity(_, "蓝方", "AIRCRAFT"))
    for _ in range(NUM_PADDING):
        raw_entities.append(create_fake_entity(_, "c", "UNKNOWN"))

    encoded, mask = encoder.encode(raw_entities,current_step=raw_entities[0].get('reportTime'))
    return {
        "encoded_data": torch.tensor(encoded, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.bool),
    }
import pdb  
# ---------------- 主测试函数 ----------------
def test_maddpg_action():
    maddpg = MADDPG(actor_types=ACTOR_TYPES, batch_size=BATCH_SIZE)
    print("✅ MADDPG 初始化成功")

    fake_state = create_fake_state()

    state_batch = {
        'encoded_data': fake_state['encoded_data'].unsqueeze(0),  # [1, max_entities, dim]
        'mask': fake_state['mask'].unsqueeze(0)
    }
    actors_input, _, _, actors_mask, target_features, target_mask = maddpg._encode_global_and_self_state(state_batch)

    for i, agent in enumerate(maddpg.agents):

        if isinstance(agent.actor, ACTOR_TYPES[4]):  # AttackTargetActor
            action = agent.get_action((actors_input, target_features, target_mask), training=True)

        else:
            action = agent.get_action(actors_input, training=True)


        print(f"Agent {i} 动作 shape：", action.shape)
        pdb.set_trace()

if __name__ == "__main__":
    test_maddpg_action()
