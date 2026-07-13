from entity import *
import copy
from actor import AircraftTakeOffActor, ReturnToBaseActor
from actor import ActionEncoder,actor_output_to_action
# 合并全局状态和所有智能体动作特征
class Critic(nn.Module):
    """集中式Critic网络, 评估联合动作价值"""
    def __init__(self, 
                 state_dim=256, #全局状态
                 action_feat_dim=128,# 动作特征
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=3):
        super().__init__()
        
        # 特征交叉层：融合全局状态和动作特征
        self.cross_feat = nn.Sequential(
            nn.Linear(state_dim + action_feat_dim, hidden_dim*2),  # 扩展维度
            nn.GELU(),  # 平滑的非线性激活
            nn.Linear(hidden_dim*2, hidden_dim),  # 压缩回隐藏维度
            nn.LayerNorm(hidden_dim)  # 稳定训练过程
        )
        
        # 分层Transformer编码器：捕获多智能体交互
        self.attention_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # 多尺度特征融合：整合各层抽象特征
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim*(num_layers+1), hidden_dim),  # 拼接所有层输出
            nn.SiLU(),  # 引入非线性
            nn.LayerNorm(hidden_dim)  # 归一化
        )
        
        # 价值预估头：输出Q值估计
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),  # 最终抽象层
            nn.SiLU(),  # 保持特征平滑性
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(256, 1)  # 输出标量Q值
        )

    def forward(self, global_state, encoded_actions):
        # 特征交叉：合并全局状态和所有智能体动作特征
        joint_input = torch.cat([global_state, encoded_actions], dim=-1)
        x = self.cross_feat(joint_input)
        
        # 分层特征提取：保存各层输出
        features = [x]  # 包含初始交叉特征
        for layer in self.attention_layers:
            x = layer(x)
            features.append(x)  # 保存每层输出
        
        # 特征融合：拼接各层特征形成多尺度表示
        fused = self.fusion_net(torch.cat(features, dim=-1))
        
        # 价值估计：输出最终Q值
        return self.value_head(fused)


# """带有残差连接的Transformer模块"""
class TransformerBlock(nn.Module):
    """带有残差连接的Transformer模块"""
    def __init__(self, dim, num_heads):
        super().__init__()
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),  # 扩展维度
            nn.GELU(),              # 引入非线性
            nn.Linear(dim*4, dim),  # 压缩回原维度
            nn.Dropout(0.1)         # 正则化
        )
        # 归一化层
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # 残差连接+自注意力
        attn_output, _ = self.attention(x, x, x)  # 自注意力计算
        x = self.norm1(x + attn_output)           # 残差连接+归一化
        
        # 残差连接+前馈网络
        mlp_output = self.mlp(x)
        return self.norm2(x + mlp_output)         # 再次残差连接



