import torch
from maddpg_ import MADDPG, ACTOR_TYPES   # 根据你的包结构调整 import

def main():
    # ——> 构造模型所需的超参数保持跟训练脚本一致
    model = MADDPG(
        actor_types=ACTOR_TYPES,
        state_dim=256,
        action_feat_dim=128,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.95,
        tau=0.02,
        batch_size=64
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total      = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,}")
    print(f"All params       : {total:,}")

if __name__ == "__main__":
    main()
