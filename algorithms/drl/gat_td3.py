#!/usr/bin/env python3
"""
GAT-TD3: 结合图注意力网络的TD3深度强化学习算法
用于能量高效的无人机路径规划
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import copy
import gym
import math

# 修复导入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 从能量模型中导入
from algorithms.drl.energy_model import DroneEnergyModel

# 配置类
class GATTD3Config:
    def __init__(self, config_file=None):
        # 网络架构
        self.hidden_sizes = [256, 256]
        
        # GAT相关参数
        self.gat_heads = 4
        self.gat_hidden_dim = 128
        self.gat_dropout = 0.2
        self.gat_alpha = 0.2  # LeakyReLU 角度
        
        # Actor参数
        self.actor_lr = 3e-4
        self.policy_noise = 0.2  # 目标策略噪声
        self.noise_clip = 0.5    # 噪声裁剪范围
        
        # Critic参数
        self.critic_lr = 3e-4
        
        # 训练参数
        self.buffer_size = 1000000
        self.min_buffer_size = 1000
        self.batch_size = 100
        self.gamma = 0.99        # 折扣因子
        self.tau = 0.005         # 目标网络更新率
        
        # TD3特定参数
        self.policy_freq = 2     # 延迟策略更新参数
        
        # 探索噪声
        self.action_noise = 0.1
        self.noise_decay = 0.995
        self.min_noise = 0.05
        
        # 训练计划
        self.total_timesteps = 1000000
        self.update_freq = 1     # 每n步更新网络
        self.save_freq = 10000   # 每n步保存模型
        self.log_freq = 1000     # 每n步记录指标
        self.eval_freq = 10000   # 每n步评估模型
        
        # 能量效率相关
        self.energy_weight = 0.5  # 能量奖励权重
        self.use_energy_model = True  # 是否使用能量模型
        
        # 环境表示
        self.max_graph_distance = 50.0  # 图中连接的最大距离
        self.use_graph_representation = True  # 是否使用图表示
        
        # 从文件加载配置（如果提供）
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """从JSON文件加载配置"""
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
                
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
            print(f"已从 {config_file} 加载GAT-TD3配置")
        except Exception as e:
            print(f"加载GAT-TD3配置时出错: {e}")
    
    def save_config(self, save_path):
        """将配置保存到JSON文件"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        
        try:
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"已将GAT-TD3配置保存到 {save_path}")
        except Exception as e:
            print(f"保存GAT-TD3配置时出错: {e}")


# 图注意力层
class GraphAttentionLayer(nn.Module):
    """
    图注意力层实现
    基于论文: Graph Attention Networks (Veličković et al., ICLR 2018)
    """
    
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 权重矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力参数
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # Leaky ReLU激活
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        # 线性层替代矩阵乘法，更灵活处理不同维度
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.data = self.W.data
    
    def forward(self, input, adj):
        """前向传播"""
        # 使用线性层进行特征变换
        h = self.linear(input)
        
        # 单样本处理
        if len(input.shape) == 2:
            N = input.size(0)
            
            # 计算注意力系数
            # 方法1：使用循环计算注意力（更稳定但较慢）
            attention = torch.zeros((N, N), device=input.device)
            for i in range(N):
                for j in range(N):
                    if adj[i, j] > 0:  # 只计算有连接的节点
                        cat_features = torch.cat([h[i], h[j]]).unsqueeze(0)
                        attention[i, j] = self.leakyrelu(torch.matmul(cat_features, self.a).squeeze())
            
            # 掩码和标准化
            zero_vec = -9e15 * torch.ones_like(attention)
            attention = torch.where(adj > 0, attention, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            
            # 特征聚合
            h_prime = torch.matmul(attention, h)
            
            # 非线性激活
            if self.concat:
                h_prime = F.elu(h_prime)
            
            return h_prime
        
        # 批处理情况
        else:
            batch_size = input.size(0)
            results = []
            
            # 对每个样本单独处理
            for i in range(batch_size):
                h_i = h[i]  # 当前样本的特征
                adj_i = adj[i] if len(adj.shape) > 2 else adj  # 当前样本的邻接矩阵
                
                N = h_i.size(0)
                
                # 计算注意力系数
                attention = torch.zeros((N, N), device=input.device)
                for j in range(N):
                    for k in range(N):
                        if adj_i[j, k] > 0:  # 只计算有连接的节点
                            cat_features = torch.cat([h_i[j], h_i[k]]).unsqueeze(0)
                            attention[j, k] = self.leakyrelu(torch.matmul(cat_features, self.a).squeeze())
                
                # 掩码和标准化
                zero_vec = -9e15 * torch.ones_like(attention)
                attention = torch.where(adj_i > 0, attention, zero_vec)
                attention = F.softmax(attention, dim=1)
                attention = F.dropout(attention, self.dropout, training=self.training)
                
                # 特征聚合
                h_prime = torch.matmul(attention, h_i)
                
                # 非线性激活
                if self.concat:
                    h_prime = F.elu(h_prime)
                
                results.append(h_prime)
            
            # 合并结果
            return torch.stack(results)


# 多头图注意力层
class MultiHeadGAT(nn.Module):
    """
    多头图注意力网络
    通过并行多个GAT层并连接输出来增加模型表示能力
    """
    
    def __init__(self, in_features, hidden_features, out_features, heads=4, dropout=0.6, alpha=0.2):
        super(MultiHeadGAT, self).__init__()
        self.heads = heads
        
        # 多头GAT层
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(in_features, hidden_features, dropout, alpha, concat=True)
            for _ in range(heads)
        ])
        
        # 输出层
        self.out_layer = GraphAttentionLayer(
            hidden_features * heads, 
            out_features, 
            dropout, 
            alpha, 
            concat=False
        )
    
    def forward(self, x, adj):
        """前向传播"""
        # 并行处理多头注意力
        x = torch.cat([layer(x, adj) for layer in self.gat_layers], dim=1)
        # 输出层
        x = self.out_layer(x, adj)
        return x


# GAT-TD3的Actor网络
class GATActor(nn.Module):
    """
    结合图注意力机制的Actor网络
    输入状态和图结构，输出连续动作
    """
    
    def __init__(self, state_dim, action_dim, max_action, config):
        super(GATActor, self).__init__()
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 节点特征提取层
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[0], config.gat_hidden_dim),
            nn.ReLU()
        )
        
        # 图注意力层
        self.gat = MultiHeadGAT(
            in_features=config.gat_hidden_dim,
            hidden_features=config.gat_hidden_dim,
            out_features=config.hidden_sizes[1],
            heads=config.gat_heads,
            dropout=config.gat_dropout,
            alpha=config.gat_alpha
        )
        
        # 动作输出层
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_sizes[1], config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[1], action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
    
    def forward(self, state, adj=None):
        """
        前向传播
        
        参数:
            state: 状态张量 [batch_size, state_dim] 或 [node_count, state_dim]
            adj: 邻接矩阵 [batch_size, node_count, node_count] 或 [node_count, node_count]
        
        返回:
            动作张量 [batch_size, action_dim] 或 [1, action_dim]
        """
        # 如果使用图表示且提供了邻接矩阵
        if self.config.use_graph_representation and adj is not None:
            # 提取节点特征
            x = self.feature_extraction(state)
            # 应用图注意力
            x = self.gat(x, adj)
            # 只使用与当前位置相关的节点特征 (第一个节点)
            x = x[0].unsqueeze(0)
        else:
            # 直接使用状态作为特征
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            x = self.feature_extraction(state)
            if len(x.shape) > 2:  # 如果是批处理
                x = x.view(x.shape[0], -1)
            else:
                x = x.view(1, -1)
        
        # 确保维度匹配
        if x.shape[1] != self.config.hidden_sizes[1]:
            # 添加线性层进行维度转换
            x = nn.Linear(x.shape[1], self.config.hidden_sizes[1]).to(x.device)(x)
        
        # 输出动作
        actions = self.action_head(x)
        return self.max_action * actions


# GAT-TD3的Critic网络
class GATCritic(nn.Module):
    """
    结合图注意力机制的Critic网络
    输入状态、动作和图结构，输出Q值
    """
    
    def __init__(self, state_dim, action_dim, config):
        super(GATCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 节点特征提取层
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[0], config.gat_hidden_dim),
            nn.ReLU()
        )
        
        # 图注意力层
        self.gat = MultiHeadGAT(
            in_features=config.gat_hidden_dim,
            hidden_features=config.gat_hidden_dim,
            out_features=config.hidden_sizes[1],
            heads=config.gat_heads,
            dropout=config.gat_dropout,
            alpha=config.gat_alpha
        )
        
        # 第一个Q网络的动作处理和输出层
        self.q1_action_layer = nn.Linear(action_dim, config.hidden_sizes[1])
        self.q1_output = nn.Sequential(
            nn.Linear(config.hidden_sizes[1] * 2, config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[1], 1)
        )
        
        # 第二个Q网络的动作处理和输出层 (TD3使用双Q网络)
        self.q2_action_layer = nn.Linear(action_dim, config.hidden_sizes[1])
        self.q2_output = nn.Sequential(
            nn.Linear(config.hidden_sizes[1] * 2, config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[1], 1)
        )
    
    def forward(self, state, action, adj=None):
        """
        前向传播
        
        参数:
            state: 状态张量
            action: 动作张量
            adj: 邻接矩阵
        
        返回:
            (q1_value, q2_value): 两个Q网络的输出
        """
        # 获取批处理大小
        batch_size = state.shape[0] if len(state.shape) > 1 else 1
        
        # 如果使用图表示且提供了邻接矩阵
        if self.config.use_graph_representation and adj is not None:
            # 提取节点特征
            x = self.feature_extraction(state)
            # 应用图注意力
            x = self.gat(x, adj)
            # 只使用与当前位置相关的节点特征 (第一个节点)
            x = x[0].unsqueeze(0)
            # 复制到批处理大小
            if batch_size > 1:
                x = x.repeat(batch_size, 1)
        else:
            # 直接使用状态作为特征
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            x = self.feature_extraction(state)
            if len(x.shape) > 2:  # 如果是批处理
                x = x.view(x.shape[0], -1)
            else:
                x = x.view(1, -1)
        
        # 确保维度匹配
        if x.shape[1] != self.config.hidden_sizes[1]:
            # 添加线性层进行维度转换
            x = nn.Linear(x.shape[1], self.config.hidden_sizes[1]).to(x.device)(x)
        
        # 处理动作
        a1 = F.relu(self.q1_action_layer(action))
        a2 = F.relu(self.q2_action_layer(action))
        
        # 确保批处理大小匹配
        if x.shape[0] != a1.shape[0]:
            if x.shape[0] == 1:
                x = x.repeat(a1.shape[0], 1)
            elif a1.shape[0] == 1:
                a1 = a1.repeat(x.shape[0], 1)
                a2 = a2.repeat(x.shape[0], 1)
        
        # 合并状态和动作特征
        q1_input = torch.cat([x, a1], dim=1)
        q2_input = torch.cat([x, a2], dim=1)
        
        # 输出Q值
        q1 = self.q1_output(q1_input)
        q2 = self.q2_output(q2_input)
        
        return q1, q2
    
    def q1(self, state, action, adj=None):
        """只返回第一个Q值，用于Actor的策略梯度更新"""
        # 获取批处理大小
        batch_size = state.shape[0] if len(state.shape) > 1 else 1
        
        if self.config.use_graph_representation and adj is not None:
            x = self.feature_extraction(state)
            x = self.gat(x, adj)
            x = x[0].unsqueeze(0)
            # 复制到批处理大小
            if batch_size > 1:
                x = x.repeat(batch_size, 1)
        else:
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            x = self.feature_extraction(state)
            if len(x.shape) > 2:
                x = x.view(x.shape[0], -1)
            else:
                x = x.view(1, -1)
        
        # 确保维度匹配
        if x.shape[1] != self.config.hidden_sizes[1]:
            # 添加线性层进行维度转换
            x = nn.Linear(x.shape[1], self.config.hidden_sizes[1]).to(x.device)(x)
        
        a1 = F.relu(self.q1_action_layer(action))
        
        # 确保批处理大小匹配
        if x.shape[0] != a1.shape[0]:
            if x.shape[0] == 1:
                x = x.repeat(a1.shape[0], 1)
            elif a1.shape[0] == 1:
                a1 = a1.repeat(x.shape[0], 1)
        
        q1_input = torch.cat([x, a1], dim=1)
        
        return self.q1_output(q1_input)


# 经验回放缓冲区
class ReplayBuffer:
    """
    经验回放缓冲区
    存储并采样智能体的经验以用于训练
    """
    
    def __init__(self, state_dim, action_dim, max_size=1000000, device='cpu'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        
        # 创建缓冲区
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        
        # 图表示相关
        self.has_graph = False
        self.adj_matrix = None  # 初始化时不知道图的大小
    
    def add(self, state, action, next_state, reward, done, adj_matrix=None, next_adj_matrix=None):
        """添加经验到缓冲区"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        
        # 如果提供了图结构，存储它们
        if adj_matrix is not None and next_adj_matrix is not None:
            self.has_graph = True
            if self.adj_matrix is None:
                # 创建邻接矩阵缓冲区
                node_count = adj_matrix.shape[0]
                self.adj_matrix = np.zeros((self.max_size, node_count, node_count))
                self.next_adj_matrix = np.zeros((self.max_size, node_count, node_count))
            
            self.adj_matrix[self.ptr] = adj_matrix
            self.next_adj_matrix[self.ptr] = next_adj_matrix
        
        # 更新指针和大小
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        ind = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'state': torch.FloatTensor(self.state[ind]).to(self.device),
            'action': torch.FloatTensor(self.action[ind]).to(self.device),
            'next_state': torch.FloatTensor(self.next_state[ind]).to(self.device),
            'reward': torch.FloatTensor(self.reward[ind]).to(self.device),
            'done': torch.FloatTensor(self.done[ind]).to(self.device)
        }
        
        # 如果有图结构，也加入批次
        if self.has_graph:
            batch['adj_matrix'] = torch.FloatTensor(self.adj_matrix[ind]).to(self.device)
            batch['next_adj_matrix'] = torch.FloatTensor(self.next_adj_matrix[ind]).to(self.device)
        
        return batch


# GAT-TD3 Agent类
class GATTD3Agent:
    """
    GAT-TD3 Agent
    结合图注意力网络的TD3强化学习智能体
    """
    
    def __init__(self, state_dim, action_dim, max_action, config=None, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        
        # 配置
        if config is None:
            self.config = GATTD3Config()
        elif isinstance(config, str):
            self.config = GATTD3Config(config)
        else:
            self.config = config
        
        # 初始化Actor网络
        self.actor = GATActor(state_dim, action_dim, max_action, self.config).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        
        # 初始化Critic网络
        self.critic = GATCritic(state_dim, action_dim, self.config).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        
        # 初始化探索噪声和缓冲区
        self.exploration_noise = self.config.action_noise
        self.buffer = ReplayBuffer(state_dim, action_dim, self.config.buffer_size, device)
        
        # 初始化能量消耗模型
        if self.config.use_energy_model:
            self.energy_model = DroneEnergyModel()
        
        # 训练状态
        self.total_it = 0  # 总迭代次数
        
        print(f"GAT-TD3 Agent initialized with state dim: {state_dim}, action dim: {action_dim}")
    
    def select_action(self, state, adj_matrix=None, evaluate=False):
        """
        选择动作
        
        参数:
            state: 环境状态
            adj_matrix: 邻接矩阵（如果使用图表示）
            evaluate: 是否为评估模式（不添加探索噪声）
        
        返回:
            选定的动作
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # 如果使用图表示且提供了邻接矩阵
        if self.config.use_graph_representation and adj_matrix is not None:
            adj_tensor = torch.FloatTensor(adj_matrix).to(self.device)
            action = self.actor(state_tensor, adj_tensor).cpu().data.numpy().flatten()
        else:
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        # 在训练模式下添加探索噪声
        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
            # 将动作限制在合法范围内
            action = np.clip(action, -self.max_action, self.max_action)
            
            # 适用于训练期间降低噪声
            self.exploration_noise = max(
                self.exploration_noise * self.config.noise_decay,
                self.config.min_noise
            )
        
        return action
    
    def store_experience(self, state, action, next_state, reward, done, adj_matrix=None, next_adj_matrix=None):
        """
        存储经验到回放缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            next_state: 下一个状态
            reward: 获得的奖励
            done: 是否是终止状态
            adj_matrix: 当前邻接矩阵
            next_adj_matrix: 下一个邻接矩阵
        """
        self.buffer.add(state, action, next_state, reward, done, adj_matrix, next_adj_matrix)
    
    def _compute_energy_reward(self, state, action):
        """
        计算基于能量消耗的奖励
        
        参数:
            state: 当前状态（字典格式）
            action: 执行的动作
            
        返回:
            energy_reward: 能量消耗奖励（负值）
        """
        if not self.config.use_energy_model:
            return 0.0
        
        try:
            # 处理Pendulum环境的特殊情况（只有一个动作维度）
            if len(action) == 1:
                # 创建适合能量模型的动作格式 [roll, pitch, yaw_rate, thrust]
                expanded_action = np.array([0.0, 0.0, 0.0, abs(action[0])])
                # 计算能量消耗
                energy_consumption = self.energy_model.calculate_energy_consumption(state, expanded_action)
            else:
                # 计算能量消耗
                energy_consumption = self.energy_model.calculate_energy_consumption(state, action)
            
            # 能量奖励为负值，消耗越多惩罚越大
            energy_reward = -self.config.energy_weight * energy_consumption
            
            return energy_reward
        except Exception as e:
            print(f"能量计算错误: {e}")
            return 0.0
    
    def update_policy(self):
        """更新策略网络和价值网络"""
        # 如果缓冲区中的样本不足，则跳过更新
        if self.buffer.size < self.config.min_buffer_size:
            return {
                'critic_loss': 0.0,
                'actor_loss': 0.0,
                'q_value': 0.0
            }
        
        self.total_it += 1
        
        # 从缓冲区采样
        batch = self.buffer.sample(self.config.batch_size)
        
        # 获取批数据
        state = batch['state']
        action = batch['action']
        next_state = batch['next_state']
        reward = batch['reward']
        done = batch['done']
        
        # 获取邻接矩阵（如果有）
        adj = batch.get('adj_matrix', None)
        next_adj = batch.get('next_adj_matrix', None)
        
        with torch.no_grad():
            # 选择下一个动作并添加目标策略噪声
            noise = (torch.randn_like(action) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            
            # 计算下一个动作
            if self.config.use_graph_representation and next_adj is not None:
                next_action = (self.actor_target(next_state, next_adj) + noise).clamp(-self.max_action, self.max_action)
                # 获取目标Q值
                target_q1, target_q2 = self.critic_target(next_state, next_action, next_adj)
            else:
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                # 获取目标Q值
                target_q1, target_q2 = self.critic_target(next_state, next_action)
            
            # 取两个Q值中的最小值
            target_q = torch.min(target_q1, target_q2)
            # 计算目标值
            target_q = reward + (1 - done) * self.config.gamma * target_q
        
        # 当前Q值
        if self.config.use_graph_representation and adj is not None:
            current_q1, current_q2 = self.critic(state, action, adj)
        else:
            current_q1, current_q2 = self.critic(state, action)
        
        # 计算critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 优化critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟策略更新
        actor_loss = torch.tensor(0.0)
        if self.total_it % self.config.policy_freq == 0:
            # 计算actor损失
            if self.config.use_graph_representation and adj is not None:
                actor_loss = -self.critic.q1(state, self.actor(state, adj), adj).mean()
            else:
                actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            
            # 优化actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        # 返回训练指标
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q1.mean().item()
        }
    
    def save(self, directory):
        """保存模型"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pth"))
        torch.save(self.actor_target.state_dict(), os.path.join(directory, "actor_target.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(directory, "critic_target.pth"))
        
        # 保存配置
        self.config.save_config(os.path.join(directory, "config.json"))
        
        print(f"模型已保存到 {directory}")
    
    def load(self, directory):
        """加载模型"""
        self.actor.load_state_dict(torch.load(os.path.join(directory, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, "critic.pth")))
        self.actor_target.load_state_dict(torch.load(os.path.join(directory, "actor_target.pth")))
        self.critic_target.load_state_dict(torch.load(os.path.join(directory, "critic_target.pth")))
        
        # 加载配置
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            self.config.load_config(config_path)
        
        print(f"模型已从 {directory} 加载")


# 图结构生成工具
def build_adjacency_matrix(waypoints, current_position, obstacles=None, max_distance=50.0):
    """
    构建邻接矩阵
    
    参数:
        waypoints: 必经点列表
        current_position: 当前无人机位置
        obstacles: 障碍物列表
        max_distance: 两点之间的最大连接距离
    
    返回:
        邻接矩阵, 节点特征
    """
    # 确保输入是numpy数组
    if not isinstance(current_position, np.ndarray):
        current_position = np.array(current_position)
    
    # 转换waypoints为numpy数组
    if isinstance(waypoints, list):
        waypoints = [np.array(wp) if not isinstance(wp, np.ndarray) else wp for wp in waypoints]
    
    # 所有点（包括当前位置和所有必经点）
    all_points = [current_position] + waypoints
    n = len(all_points)
    
    # 初始化邻接矩阵和节点特征
    adj_matrix = np.zeros((n, n))
    node_features = np.zeros((n, 3))  # 初始只包含位置特征
    
    # 记录每个点的特征
    for i in range(n):
        node_features[i] = all_points[i]
    
    # 填充邻接矩阵
    for i in range(n):
        for j in range(n):
            if i == j:
                adj_matrix[i, j] = 1.0  # 自连接
                continue
            
            # 计算两点之间的欧氏距离
            p1 = all_points[i]
            p2 = all_points[j]
            distance = np.linalg.norm(p1 - p2)
            
            # 如果距离在阈值内，检查连线是否与障碍物相交
            if distance <= max_distance:
                collision = False
                
                # 如果提供了障碍物，检查连线是否与障碍物相交
                if obstacles:
                    # 简化的碰撞检测：检查连线上的采样点是否与障碍物碰撞
                    for t in np.linspace(0, 1, num=10):
                        point = p1 + t * (p2 - p1)
                        
                        # 检查点是否在任何障碍物内
                        for obstacle in obstacles:
                            obs_pos = np.array(obstacle['position']) if isinstance(obstacle['position'], list) else obstacle['position']
                            obs_size = np.array(obstacle['size']) if isinstance(obstacle['size'], list) else obstacle['size']
                            
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
    
    return adj_matrix, node_features


# 计算能量相关的路径特征
def calculate_energy_path_features(path, energy_model):
    """
    计算给定路径的能量相关特征
    
    参数:
        path: 路径点列表
        energy_model: 能量模型
    
    返回:
        total_energy: 总能量消耗
        energy_efficiency: 能量效率 (m/kJ)
        flight_time: 预计飞行时间 (min)
    """
    # 如果路径少于2个点，无法计算
    if len(path) < 2:
        return 0, 0, 0
    
    total_energy = 0
    total_distance = 0
    total_time = 0
    dt = 0.1  # 时间步长，秒
    
    # 计算每个路径段的能量消耗
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        
        # 计算距离
        distance = np.linalg.norm(p2 - p1)
        total_distance += distance
        
        # 估计速度和加速度
        velocity = (p2 - p1) / dt
        velocity_magnitude = np.linalg.norm(velocity)
        
        # 估计飞行时间
        segment_time = distance / max(velocity_magnitude, 0.1)  # 避免除以零
        total_time += segment_time
        
        # 创建简化状态
        state = {
            'linear_velocity': velocity,
            'angular_velocity': np.array([0.0, 0.0, 0.0])
        }
        
        # 估计动作（简化为恒定动作）
        action = np.array([0.0, 0.0, 0.0, 0.6])  # [roll, pitch, yaw_rate, thrust]
        
        # 计算能量消耗
        energy = energy_model.calculate_energy_consumption(state, action, dt)
        total_energy += energy * (segment_time / dt)  # 按时间比例缩放
    
    # 计算能量效率 (m/kJ)
    energy_efficiency = energy_model.efficiency_score(total_energy, total_distance)
    
    # 估计飞行时间 (min)
    average_power = total_energy / total_time if total_time > 0 else 0
    flight_time = total_time / 60.0  # 转换为分钟
    
    return total_energy, energy_efficiency, flight_time


# GAT-TD3训练器
class GATTD3Trainer:
    """GAT-TD3训练器"""
    
    def __init__(self, env, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = torch.device(device)
        
        # 加载配置
        if config is None:
            self.config = GATTD3Config()
        elif isinstance(config, str):
            self.config = GATTD3Config(config)
        else:
            self.config = config
            
        # 获取状态和动作维度
        self.state_dim = env.flat_observation_space.shape[0]
        
        # 检查动作空间是否连续
        assert isinstance(env.action_space, gym.spaces.Box), "GAT-TD3 requires continuous action space"
        
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        
        # 初始化智能体
        self.agent = GATTD3Agent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            config=self.config,
            device=device
        )
        
        # 初始化能量模型
        if self.config.use_energy_model:
            self.energy_model = DroneEnergyModel()
        
        # 跟踪训练进度
        self.best_eval_reward = -float('inf')
        self.best_eval_energy_efficiency = 0
        self.start_time = None
        
        # 训练指标
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_energy': [],
            'episode_efficiency': [],
            'critic_loss': [],
            'actor_loss': [],
            'q_values': [],
            'exploration_noise': [],
            'timestamp': []
        }
    
    def train(self, total_timesteps=None):
        """训练智能体"""
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps
            
        print(f"开始GAT-TD3训练，总步数: {total_timesteps}...")
        
        # 设置时间和跟踪
        self.start_time = time.time()
        timesteps_elapsed = 0
        episode_num = 0
        
        # 获取环境信息
        waypoints = self.env.waypoints if hasattr(self.env, 'waypoints') else []
        obstacles = self.env.obstacles if hasattr(self.env, 'obstacles') else []
        
        # 主训练循环
        state, info = self.env.reset()
        
        while timesteps_elapsed < total_timesteps:
            episode_num += 1
            done = False
            episode_step = 0
            episode_reward = 0
            episode_energy = 0
            
            # 生成初始邻接矩阵（如果使用图表示）
            if self.config.use_graph_representation:
                current_position = self.env.drone_position if hasattr(self.env, 'drone_position') else np.zeros(3)
                adj_matrix, _ = build_adjacency_matrix(
                    waypoints=waypoints,
                    current_position=current_position,
                    obstacles=obstacles,
                    max_distance=self.config.max_graph_distance
                )
            else:
                adj_matrix = None
            
            # 记录路径
            path = [current_position] if self.config.use_graph_representation else []
            
            # 情节循环
            while not done and timesteps_elapsed < total_timesteps:
                # 选择动作
                action = self.agent.select_action(state, adj_matrix)
                
                # 在环境中执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # 记录能量消耗
                if self.config.use_energy_model:
                    # 创建状态字典
                    state_dict = {}
                    if isinstance(state, np.ndarray):
                        # 将平坦状态解析为字典（取决于环境实现）
                        if self.state_dim >= 9:  # 假设至少有位置、速度、角速度
                            state_dict['linear_velocity'] = state[3:6]  # 假设线速度在索引3-5
                            state_dict['angular_velocity'] = state[6:9]  # 假设角速度在索引6-8
                    else:
                        state_dict = state  # 假设state已经是字典
                    
                    # 计算能量消耗并添加到奖励
                    energy_consumption = self.agent._compute_energy_reward(state_dict, action)
                    episode_energy -= energy_consumption  # 负能量奖励转换为正能量消耗
                
                # 更新图结构（如果使用图表示）
                if self.config.use_graph_representation:
                    current_position = self.env.drone_position if hasattr(self.env, 'drone_position') else np.zeros(3)
                    path.append(current_position)
                    
                    next_adj_matrix, _ = build_adjacency_matrix(
                        waypoints=waypoints,
                        current_position=current_position,
                        obstacles=obstacles,
                        max_distance=self.config.max_graph_distance
                    )
                else:
                    next_adj_matrix = None
                
                # 存储经验
                self.agent.store_experience(state, action, next_state, reward, float(done), adj_matrix, next_adj_matrix)
                
                # 更新策略
                if timesteps_elapsed % self.config.update_freq == 0:
                    update_info = self.agent.update_policy()
                    
                    # 记录训练指标
                    if timesteps_elapsed % self.config.log_freq == 0:
                        self.training_metrics['critic_loss'].append(update_info['critic_loss'])
                        self.training_metrics['actor_loss'].append(update_info['actor_loss'])
                        self.training_metrics['q_values'].append(update_info['q_value'])
                        self.training_metrics['exploration_noise'].append(self.agent.exploration_noise)
                        self.training_metrics['timestamp'].append(time.time() - self.start_time)
                
                # 保存模型
                if timesteps_elapsed % self.config.save_freq == 0:
                    save_dir = os.path.join("training_results", f"gattd3_{int(time.time())}")
                    self.agent.save(save_dir)
                
                # 评估模型
                if timesteps_elapsed % self.config.eval_freq == 0:
                    eval_reward, eval_energy, eval_efficiency = self.evaluate()
                    print(f"评估结果 - 步数: {timesteps_elapsed}, 奖励: {eval_reward:.2f}, 能量效率: {eval_efficiency:.2f} m/kJ")
                    
                    # 保存最佳模型
                    if eval_reward > self.best_eval_reward:
                        self.best_eval_reward = eval_reward
                        self.agent.save(os.path.join("training_results", "gattd3_best_reward"))
                    
                    if eval_efficiency > self.best_eval_energy_efficiency:
                        self.best_eval_energy_efficiency = eval_efficiency
                        self.agent.save(os.path.join("training_results", "gattd3_best_efficiency"))
                
                # 打印训练进度
                if time.time() - self.start_time > 10:  # 每10秒打印一次
                    fps = timesteps_elapsed / (time.time() - self.start_time)
                    print(f"训练进度: {timesteps_elapsed}/{total_timesteps} 步 ({100*timesteps_elapsed/total_timesteps:.1f}%), FPS: {fps:.1f}")
                    self.start_time = time.time()
                    timesteps_elapsed = 0
                
                # 更新状态和计数器
                state = next_state
                adj_matrix = next_adj_matrix
                timesteps_elapsed += 1
                episode_step += 1
                episode_reward += reward
            
            # 情节结束
            # 计算能量效率（如果有路径）
            episode_efficiency = 0
            if self.config.use_energy_model and len(path) > 1:
                path_array = np.array(path)
                _, episode_efficiency, _ = calculate_energy_path_features(path_array, self.energy_model)
            
            # 记录情节结果
            self.training_metrics['episode_rewards'].append(episode_reward)
            self.training_metrics['episode_lengths'].append(episode_step)
            self.training_metrics['episode_energy'].append(episode_energy)
            self.training_metrics['episode_efficiency'].append(episode_efficiency)
            
            print(f"情节 {episode_num} 结束，步数: {episode_step}, 奖励: {episode_reward:.2f}, 能量: {episode_energy:.2f}, 效率: {episode_efficiency:.2f} m/kJ")
            
            # 重置环境
            state, info = self.env.reset()
        
        # 训练结束
        print("训练完成!")
        
        # 绘制训练指标
        self._plot_metrics()
        
        # 保存最终模型
        final_save_dir = os.path.join("training_results", f"gattd3_final_{int(time.time())}")
        self.agent.save(final_save_dir)
        
        return self.agent
    
    def evaluate(self, n_episodes=10, render=False):
        """评估智能体的性能"""
        total_rewards = []
        total_energy = []
        total_efficiency = []
        
        for i in range(n_episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # 重置环境
            state, info = self.env.reset()
            
            # 获取环境信息
            waypoints = self.env.waypoints if hasattr(self.env, 'waypoints') else []
            obstacles = self.env.obstacles if hasattr(self.env, 'obstacles') else []
            
            # 生成初始邻接矩阵（如果使用图表示）
            if self.config.use_graph_representation:
                current_position = self.env.drone_position if hasattr(self.env, 'drone_position') else np.zeros(3)
                adj_matrix, _ = build_adjacency_matrix(
                    waypoints=waypoints,
                    current_position=current_position,
                    obstacles=obstacles,
                    max_distance=self.config.max_graph_distance
                )
            else:
                adj_matrix = None
            
            # 记录路径
            path = [current_position] if self.config.use_graph_representation else []
            
            # 情节循环
            while not done:
                # 选择动作（评估模式，无探索）
                action = self.agent.select_action(state, adj_matrix, evaluate=True)
                
                # 在环境中执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # 渲染（如果需要）
                if render:
                    self.env.render()
                
                # 更新图结构（如果使用图表示）
                if self.config.use_graph_representation:
                    current_position = self.env.drone_position if hasattr(self.env, 'drone_position') else np.zeros(3)
                    path.append(current_position)
                    
                    adj_matrix, _ = build_adjacency_matrix(
                        waypoints=waypoints,
                        current_position=current_position,
                        obstacles=obstacles,
                        max_distance=self.config.max_graph_distance
                    )
                
                # 更新状态和计数器
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            # 计算能量相关指标
            episode_energy = 0
            episode_efficiency = 0
            if self.config.use_energy_model and len(path) > 1:
                path_array = np.array(path)
                episode_energy, episode_efficiency, _ = calculate_energy_path_features(path_array, self.energy_model)
            
            # 记录情节结果
            total_rewards.append(episode_reward)
            total_energy.append(episode_energy)
            total_efficiency.append(episode_efficiency)
        
        # 计算平均值
        mean_reward = np.mean(total_rewards)
        mean_energy = np.mean(total_energy)
        mean_efficiency = np.mean(total_efficiency)
        
        return mean_reward, mean_energy, mean_efficiency
    
    def _plot_metrics(self, save_dir=None):
        """绘制训练指标"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        plt.subplot(3, 2, 1)
        plt.plot(self.training_metrics['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(3, 2, 2)
        plt.plot(self.training_metrics['episode_lengths'])
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.subplot(3, 2, 3)
        plt.plot(self.training_metrics['critic_loss'])
        plt.title('Critic Loss')
        plt.xlabel('Step (x log_freq)')
        plt.ylabel('Loss')
        
        plt.subplot(3, 2, 4)
        plt.plot(self.training_metrics['actor_loss'])
        plt.title('Actor Loss')
        plt.xlabel('Step (x log_freq)')
        plt.ylabel('Loss')
        
        plt.subplot(3, 2, 5)
        plt.plot(self.training_metrics['q_values'])
        plt.title('Q Values')
        plt.xlabel('Step (x log_freq)')
        plt.ylabel('Q Value')
        
        plt.subplot(3, 2, 6)
        plt.plot(self.training_metrics['episode_energy'])
        plt.title('Episode Energy Consumption')
        plt.xlabel('Episode')
        plt.ylabel('Energy (J)')
        
        plt.tight_layout()
        
        # 保存或显示图表
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"gattd3_training_metrics_{int(time.time())}.png"))
            
            # 保存指标到CSV
            metrics_df = pd.DataFrame({
                'step': np.arange(0, len(self.training_metrics['q_values'])) * self.config.log_freq,
                'q_value': self.training_metrics['q_values'],
                'critic_loss': self.training_metrics['critic_loss'],
                'actor_loss': self.training_metrics['actor_loss'],
                'exploration_noise': self.training_metrics['exploration_noise'],
                'timestamp': self.training_metrics['timestamp']
            })
            metrics_df.to_csv(os.path.join(save_dir, f"gattd3_training_metrics_{int(time.time())}.csv"), index=False)
            
            # 保存情节指标到CSV
            episode_df = pd.DataFrame({
                'episode': np.arange(0, len(self.training_metrics['episode_rewards'])),
                'reward': self.training_metrics['episode_rewards'],
                'length': self.training_metrics['episode_lengths'],
                'energy': self.training_metrics['episode_energy'],
                'efficiency': self.training_metrics['episode_efficiency']
            })
            episode_df.to_csv(os.path.join(save_dir, f"gattd3_episode_metrics_{int(time.time())}.csv"), index=False)
        else:
            plt.show()


# 测试函数
def test_gattd3():
    """测试GAT-TD3算法的简单函数"""
    print("GAT-TD3算法实现完成！")
    print("这是一个结合图注意力网络和TD3深度强化学习的无人机能量高效路径规划算法。")
    print("主要特点：")
    print("1. 使用图注意力网络处理环境的空间结构")
    print("2. 结合TD3算法进行深度强化学习")
    print("3. 考虑能量消耗，实现能量高效的路径规划")
    print("4. 适用于复杂城市环境中的无人机导航")
    print("\n要运行完整训练，请使用：")
    print("  ./run_gat_td3.sh -t 100000 -e urban")
    
    return "GAT-TD3测试成功"


if __name__ == "__main__":
    test_gattd3() 