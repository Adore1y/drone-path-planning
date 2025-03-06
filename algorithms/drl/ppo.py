#!/usr/bin/env python3
"""
PPO（近端策略优化）算法实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Categorical

class PPONetwork(nn.Module):
    """PPO网络架构"""
    
    def __init__(self, state_dim, action_dim, continuous=False, hidden_dim=128):
        super(PPONetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.hidden_dim = hidden_dim
        
        # 共享特征提取层
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略网络
        if continuous:
            # 连续动作空间
            self.action_mean = nn.Linear(hidden_dim, action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            # 离散动作空间
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """前向传播，输出动作概率分布和状态价值"""
        x = self.feature_extraction(state)
        
        # 策略输出
        if self.continuous:
            action_mean = self.action_mean(x)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # 返回正态分布
            return Normal(action_mean, action_std), self.critic(x)
        else:
            action_logits = self.actor(x)
            # 返回分类分布
            return Categorical(logits=action_logits), self.critic(x)

class PPO:
    """PPO算法实现"""
    
    def __init__(self, state_dim, action_dim, continuous=False, hidden_dim=128, lr=3e-4,
                gamma=0.99, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # 创建网络
        self.network = PPONetwork(state_dim, action_dim, continuous, hidden_dim)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 存储经验
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        # 转换为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取动作分布和价值
        with torch.no_grad():
            dist, value = self.network(state_tensor)
        
        # 根据deterministic选择动作
        if deterministic:
            if self.continuous:
                action = dist.mean
            else:
                action = torch.argmax(dist.probs).item()
        else:
            action = dist.sample()
        
        # 计算动作的log概率
        log_prob = dist.log_prob(action).sum(-1)
        
        # 转换为numpy
        if self.continuous:
            action_np = action.cpu().numpy().flatten()
        else:
            action_np = action.item() if isinstance(action, torch.Tensor) else action
        
        value_np = value.item()
        log_prob_np = log_prob.item()
        
        return action_np, log_prob_np, value_np
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self, next_value=0, epochs=10, batch_size=64):
        """更新模型参数"""
        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states))
        if self.continuous:
            actions = torch.FloatTensor(np.array(self.actions))
        else:
            actions = torch.LongTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        rewards = torch.FloatTensor(np.array(self.rewards))
        values = torch.FloatTensor(np.array(self.values))
        dones = torch.FloatTensor(np.array(self.dones))
        
        # 计算GAE（广义优势估计）
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # 规范化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新模型
        for _ in range(epochs):
            # 生成随机索引
            indices = np.random.permutation(len(states))
            
            # 分批更新
            for start_idx in range(0, len(states), batch_size):
                # 获取批次索引
                idx = indices[start_idx:start_idx + batch_size]
                
                # 获取批次数据
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                # 计算新的动作分布和价值
                dist, value = self.network(mb_states)
                
                # 计算新的log概率
                if self.continuous:
                    new_log_probs = dist.log_prob(mb_actions).sum(-1)
                else:
                    new_log_probs = dist.log_prob(mb_actions)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # 计算裁剪后的目标函数
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = F.mse_loss(value.squeeze(-1), mb_returns)
                
                # 计算熵损失
                entropy_loss = -dist.entropy().mean()
                
                # 计算总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 优化器更新
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
        
        # 清空存储的经验
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def save(self, path):
        """保存模型"""
        torch.save(self.network.state_dict(), path)
    
    def load(self, path):
        """加载模型"""
        self.network.load_state_dict(torch.load(path))

# 为简化PPO训练的函数
def train_ppo_episode(env, ppo, max_steps=1000):
    """使用PPO训练单个回合"""
    state = env.reset()
    episode_reward = 0
    
    for t in range(max_steps):
        # 选择动作
        action, log_prob, value = ppo.select_action(state)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 存储经验
        ppo.store_transition(state, action, log_prob, reward, value, done)
        
        # 更新状态
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    # 更新策略
    if len(ppo.states) > 0:
        ppo.update()
    
    return episode_reward 