#!/usr/bin/env python3
"""
DQN（深度Q网络）算法实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNNetwork(nn.Module):
    """DQN网络架构"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q网络架构
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """前向传播，输出每个动作的Q值"""
        return self.network(state)

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样经验"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class DQN:
    """DQN算法实现"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3,
                gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                buffer_capacity=10000, batch_size=64, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 创建策略网络和目标网络
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不训练
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 探索参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 训练步数
        self.steps_done = 0
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        # 转换为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 贪婪策略
        if not deterministic and random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return action
    
    def update(self):
        """更新模型参数"""
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # 随机采样经验
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为tensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 优化器更新
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']

# 简化DQN训练的函数
def train_dqn_episode(env, dqn, max_steps=1000):
    """使用DQN训练单个回合"""
    state = env.reset()
    episode_reward = 0
    
    for t in range(max_steps):
        # 选择动作
        action = dqn.select_action(state)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 存储经验
        dqn.replay_buffer.push(state, action, reward, next_state, done)
        
        # 更新状态
        state = next_state
        episode_reward += reward
        
        # 更新模型
        loss = dqn.update()
        
        if done:
            break
    
    return episode_reward 