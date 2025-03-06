#!/usr/bin/env python3
"""
GAT-DRL算法实现
结合图注意力网络的深度强化学习算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # 权重矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力参数
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # Leaky ReLU激活
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, input, adj):
        """前向传播"""
        h = torch.mm(input, self.W)  # [N, out_features]
        N = h.size()[0]
        
        # 准备用于计算注意力系数的数据
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # 掩码和标准化注意力系数
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, h)
        
        return h_prime

class GATDRL(nn.Module):
    """GAT-DRL实现（结合图注意力网络的深度强化学习）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_heads=4):
        super(GATDRL, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # 特征提取层
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 图注意力层
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim//n_heads)
            for _ in range(n_heads)
        ])
        
        # 策略网络（动作概率）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 价值网络
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, adj_matrix):
        """前向传播，计算动作概率和状态价值"""
        # 特征提取
        x = self.feature_extraction(state)
        
        # 图注意力机制
        gat_outputs = [gat(x, adj_matrix) for gat in self.gat_layers]
        x = torch.cat(gat_outputs, dim=1)
        
        # 计算动作概率和状态价值
        action_probs = F.softmax(self.policy_head(x), dim=-1)
        state_value = self.value_head(x)
        
        return action_probs, state_value
    
    def select_action(self, state, adj_matrix, deterministic=False):
        """选择动作"""
        # 转换为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0)
        
        # 获取动作概率
        with torch.no_grad():
            action_probs, _ = self.forward(state_tensor, adj_tensor)
            action_probs = action_probs.squeeze(0).numpy()
        
        # 确定性策略或随机抽样
        if deterministic:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(self.action_dim, p=action_probs)
        
        return action, action_probs
    
    def evaluate(self, state, adj_matrix, action):
        """评估动作，返回log概率、熵和状态价值"""
        action_probs, state_value = self.forward(state, adj_matrix)
        
        # 计算选定动作的log概率
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_log_probs, dist_entropy, state_value
    
    def save(self, path):
        """保存模型"""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """加载模型"""
        self.load_state_dict(torch.load(path))

# 用于训练GAT-DRL的函数
def train_gatdrl(model, optimizer, state, adj_matrix, action, reward, next_state, next_adj_matrix, done,
               gamma=0.99, entropy_coef=0.01, value_coef=0.5):
    """训练GAT-DRL模型的单步更新"""
    # 计算当前状态的动作log概率、熵和价值
    action_log_probs, entropy, state_value = model.evaluate(state, adj_matrix, action)
    
    # 计算下一状态的价值
    with torch.no_grad():
        _, next_state_value = model(next_state, next_adj_matrix)
        next_state_value = next_state_value.squeeze(-1)
        # 如果是终止状态，则下一状态的价值为0
        next_state_value = next_state_value * (1 - done)
    
    # 计算目标值（TD目标）
    target_value = reward + gamma * next_state_value
    
    # 计算价值损失（MSE）
    value_loss = F.mse_loss(state_value.squeeze(-1), target_value)
    
    # 计算策略损失（负的动作log概率 * 优势）
    advantage = target_value - state_value.squeeze(-1).detach()
    policy_loss = -(action_log_probs * advantage).mean()
    
    # 计算总损失
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
    
    # 优化器更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), policy_loss.item(), value_loss.item(), entropy.mean().item()

def build_adjacency_matrix(waypoints, current_position, obstacles, max_distance=50.0):
    """
    构建邻接矩阵
    
    参数:
        waypoints: 必经点列表
        current_position: 当前无人机位置
        obstacles: 障碍物列表
        max_distance: 两点之间的最大连接距离
    
    返回:
        邻接矩阵
    """
    # 所有点（包括当前位置和所有必经点）
    all_points = [current_position] + waypoints
    n = len(all_points)
    
    # 初始化邻接矩阵
    adj_matrix = np.zeros((n, n))
    
    # 填充邻接矩阵
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # 计算两点之间的距离
            p1 = all_points[i]
            p2 = all_points[j]
            distance = np.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
            
            # 如果距离在阈值内，检查连线是否与障碍物相交
            if distance <= max_distance:
                collision = False
                
                # 简化的碰撞检测：检查连线上的采样点是否与障碍物碰撞
                for t in np.linspace(0, 1, num=10):
                    point = [p1[0] + t * (p2[0] - p1[0]),
                            p1[1] + t * (p2[1] - p1[1]),
                            p1[2] + t * (p2[2] - p1[2])]
                    
                    # 检查点是否在任何障碍物内
                    for obstacle in obstacles:
                        obs_pos = obstacle['position']
                        obs_size = obstacle['size']
                        
                        if (abs(point[0] - obs_pos[0]) < obs_size[0]/2 and
                            abs(point[1] - obs_pos[1]) < obs_size[1]/2 and
                            abs(point[2] - obs_pos[2]) < obs_size[2]/2):
                            collision = True
                            break
                    
                    if collision:
                        break
                
                # 如果没有碰撞，则设置邻接矩阵
                if not collision:
                    # 距离越近，连接权重越大
                    adj_matrix[i, j] = 1.0 / (1.0 + distance)
    
    return adj_matrix 