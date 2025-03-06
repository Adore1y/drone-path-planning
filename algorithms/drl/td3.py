#!/usr/bin/env python3
"""
TD3（双延迟深度确定性策略梯度）算法实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    """Actor网络架构，输出确定性动作"""
    
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        """前向传播，输出动作"""
        return self.max_action * self.network(state)

class Critic(nn.Module):
    """Critic网络架构，输出Q值"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 第一个Q网络
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 第二个Q网络
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """前向传播，输出两个Q值"""
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, state, action):
        """仅使用第一个Q网络"""
        x = torch.cat([state, action], dim=1)
        return self.q1(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样经验"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class TD3:
    """TD3算法实现"""
    
    def __init__(self, state_dim, action_dim, max_action=1.0, hidden_dim=256, lr=3e-4,
                gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                buffer_capacity=100000, batch_size=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        
        # 创建Actor网络
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # 创建Critic网络
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 训练步数
        self.total_it = 0
    
    def select_action(self, state, noise=0.1, deterministic=False):
        """选择动作"""
        # 转换为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            # 确定性动作
            action = self.actor(state_tensor).squeeze(0).numpy()
        
        if not deterministic:
            # 添加噪声
            action = action + np.random.normal(0, noise * self.max_action, size=self.action_dim)
            
            # 裁剪动作
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def update(self):
        """更新模型参数"""
        self.total_it += 1
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 随机采样经验
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为tensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        with torch.no_grad():
            # 选择下一个动作并添加目标策略平滑正则化的噪声
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)
            
            # 计算目标Q值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # 计算当前Q值
        current_q1, current_q2 = self.critic(states, actions)
        
        # 计算Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟更新Actor和目标网络
        if self.total_it % self.policy_delay == 0:
            # 计算Actor损失
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            # 优化Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
    
    def _soft_update(self, local_model, target_model):
        """软更新目标网络参数"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint['total_it']

# 简化TD3训练的函数
def train_td3_episode(env, td3, max_steps=1000):
    """使用TD3训练单个回合"""
    state = env.reset()
    episode_reward = 0
    
    for t in range(max_steps):
        # 选择动作
        action = td3.select_action(state)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 存储经验
        td3.replay_buffer.push(state, action, reward, next_state, done)
        
        # 更新状态
        state = next_state
        episode_reward += reward
        
        # 更新模型
        td3.update()
        
        if done:
            break
    
    return episode_reward 