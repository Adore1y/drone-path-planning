#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

# 添加父目录到Python路径，以便导入其他模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# 导入必要的模块
from utils.graph_utils import create_city_graph, convert_nx_graph_to_pyg_graph, find_nearest_node
from controllers.gat_drl.gat_model import GATEnvironmentEncoder

class DronePathPlanningEnv(gym.Env):
    """无人机路径规划强化学习环境"""
    
    def __init__(self, scenario='mixed', device='cpu'):
        """初始化环境"""
        super(DronePathPlanningEnv, self).__init__()
        
        # 加载环境数据
        self.buildings, self.boundaries = self._load_environment(scenario)
        
        # 创建城市图
        self.graph, self.node_positions = create_city_graph(self.buildings, self.boundaries)
        
        # 创建节点特征（默认为零向量）
        self.node_features = {}
        for node in self.graph.nodes():
            self.node_features[node] = [0.0, 0.0, 0.0]  # 默认特征
        
        # 创建PyG图（用于可视化和调试）
        self.pyg_graph = convert_nx_graph_to_pyg_graph(self.graph, self.node_positions, self.node_features)
        
        # 初始化GAT编码器
        self.device = torch.device(device)
        self.gat_encoder = GATEnvironmentEncoder(num_node_features=3, hidden_dim=64, output_dim=64, device=self.device)
        
        # 使用GAT编码器对环境进行编码
        self.node_embeddings = self.gat_encoder.encode_environment(self.graph, self.node_positions, self.node_features)
        
        # 初始化起点和终点
        self.start_pos, self.goal_pos = self._generate_random_positions()
        self.current_pos = self.start_pos.copy()
        
        # 找到最近的图节点
        self.start_node, _ = find_nearest_node(self.graph, self.node_positions, (self.start_pos[0], self.start_pos[2]))
        self.goal_node, _ = find_nearest_node(self.graph, self.node_positions, (self.goal_pos[0], self.goal_pos[2]))
        self.current_node = self.start_node
        
        # 定义动作空间（离散动作：上下左右前后）
        self.action_space = spaces.Discrete(6)
        
        # 定义观察空间
        # 包括：当前位置(3)、目标位置(3)、当前节点嵌入(64)、目标节点嵌入(64)
        obs_dim = 3 + 3 + 64 + 64
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # 初始化路径和步数
        self.path = [self.start_pos.copy()]
        self.steps = 0
        self.max_steps = 100
    
    def _load_environment(self, scenario):
        """加载环境数据"""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../controllers/supervisor'))
        
        # 现在使用完整路径导入
        from supervisor_controller import UAVSupervisor
        
        supervisor = UAVSupervisor()
        supervisor.load_buildings(scenario)
        
        return supervisor.buildings, supervisor.boundaries
    
    def reset(self, seed=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 生成新的起点和终点
        self.start_pos, self.goal_pos = self._generate_random_positions()
        self.current_pos = self.start_pos.copy()
        
        # 找到最近的图节点
        self.start_node, _ = find_nearest_node(self.graph, self.node_positions, (self.start_pos[0], self.start_pos[2]))
        self.goal_node, _ = find_nearest_node(self.graph, self.node_positions, (self.goal_pos[0], self.goal_pos[2]))
        self.current_node = self.start_node
        
        # 重置路径和步数
        self.path = [self.start_pos.copy()]
        self.steps = 0
        
        # 获取观察
        observation = self._get_observation()
        
        return observation, {}
    
    def _generate_random_positions(self):
        """生成随机起点和终点"""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../controllers/supervisor'))
        
        # 使用完整路径导入
        from supervisor_controller import UAVSupervisor
        
        supervisor = UAVSupervisor()
        supervisor.buildings = self.buildings
        supervisor.boundaries = self.boundaries
        
        # 生成随机位置
        start_pos = supervisor.generate_random_position(min_height=1.0, max_height=5.0)
        goal_pos = supervisor.generate_random_position(min_height=1.0, max_height=5.0)
        
        # 确保位置不在建筑物内
        while supervisor.is_collision_with_buildings(start_pos):
            start_pos = supervisor.generate_random_position(min_height=1.0, max_height=5.0)
        
        while supervisor.is_collision_with_buildings(goal_pos):
            goal_pos = supervisor.generate_random_position(min_height=1.0, max_height=5.0)
        
        return np.array(start_pos), np.array(goal_pos)
    
    def _get_observation(self):
        """获取当前观察"""
        # 当前位置
        current_pos = self.current_pos
        
        # 目标位置
        goal_pos = self.goal_pos
        
        # 当前节点嵌入
        current_node_embedding = self.node_embeddings[self.current_node]
        
        # 目标节点嵌入
        goal_node_embedding = self.node_embeddings[self.goal_node]
        
        # 组合观察
        observation = np.concatenate([
            current_pos,
            goal_pos,
            current_node_embedding,
            goal_node_embedding
        ]).astype(np.float32)
        
        return observation
    
    def step(self, action):
        """执行动作并返回新的状态"""
        # 增加步数
        self.steps += 1
        
        # 根据动作更新位置
        step_size = 1.0  # 每步移动的距离
        
        # 动作映射：0=前, 1=后, 2=左, 3=右, 4=上, 5=下
        if action == 0:  # 前
            self.current_pos[2] -= step_size
        elif action == 1:  # 后
            self.current_pos[2] += step_size
        elif action == 2:  # 左
            self.current_pos[0] -= step_size
        elif action == 3:  # 右
            self.current_pos[0] += step_size
        elif action == 4:  # 上
            self.current_pos[1] += step_size
        elif action == 5:  # 下
            self.current_pos[1] -= step_size
        
        # 确保高度不会太低
        self.current_pos[1] = max(0.5, self.current_pos[1])
        
        # 记录路径
        self.path.append(self.current_pos.copy())
        
        # 更新当前节点
        self.current_node, _ = find_nearest_node(self.graph, self.node_positions, (self.current_pos[0], self.current_pos[2]))
        
        # 检查是否到达目标
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal_pos)
        reached_goal = distance_to_goal < 2.0
        
        # 检查是否与建筑物碰撞
        collision = self._check_collision(self.current_pos)
        
        # 检查是否超出边界
        min_x, min_z, max_x, max_z = self.boundaries
        out_of_bounds = (self.current_pos[0] < min_x or 
                         self.current_pos[0] > max_x or 
                         self.current_pos[2] < min_z or 
                         self.current_pos[2] > max_z)
        
        # 确定是否结束
        done = reached_goal or collision or out_of_bounds or self.steps >= self.max_steps
        
        # 计算奖励
        reward = 0
        
        if reached_goal:
            reward += 100.0  # 到达目标的奖励
        elif collision:
            reward -= 100.0  # 碰撞惩罚
        elif out_of_bounds:
            reward -= 50.0   # 超出边界惩罚
        else:
            # 距离奖励：越接近目标，奖励越高
            prev_distance = np.linalg.norm(self.path[-2] - self.goal_pos)
            current_distance = distance_to_goal
            reward += (prev_distance - current_distance) * 10.0
            
            # 高度奖励：保持合理高度
            optimal_height = 5.0
            height_diff = abs(self.current_pos[1] - optimal_height)
            reward -= height_diff * 0.1
            
            # 时间惩罚：每步小惩罚，鼓励尽快完成
            reward -= 0.1
        
        # 获取新的观察
        observation = self._get_observation()
        
        # 准备信息字典
        info = {
            'distance_to_goal': distance_to_goal,
            'collision': collision,
            'out_of_bounds': out_of_bounds,
            'steps': self.steps,
            'path_length': self._calculate_path_length(),
            'energy': self._calculate_energy()
        }
        
        return observation, reward, done, False, info
    
    def _check_collision(self, position):
        """检查是否与建筑物碰撞"""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../controllers/supervisor'))
        
        # 使用完整路径导入
        from supervisor_controller import UAVSupervisor
        
        supervisor = UAVSupervisor()
        supervisor.buildings = self.buildings
        
        return supervisor.is_collision_with_buildings(position)
    
    def _calculate_path_length(self):
        """计算路径长度"""
        length = 0
        for i in range(1, len(self.path)):
            length += np.linalg.norm(self.path[i] - self.path[i-1])
        return length
    
    def _calculate_energy(self):
        """计算能量消耗"""
        from utils.evaluation_utils import calculate_energy_consumption
        return calculate_energy_consumption(self.path)

class GATDRLFeaturesExtractor(BaseFeaturesExtractor):
    """GAT-DRL特征提取器"""
    
    def __init__(self, observation_space, features_dim=128):
        """初始化特征提取器"""
        super(GATDRLFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # 观察空间维度
        obs_dim = observation_space.shape[0]
        
        # 假设观察是 [current_pos(3), goal_pos(3), current_node_embedding(64), goal_node_embedding(64)]
        pos_dim = 3
        embedding_dim = 64
        
        # 位置编码器
        self.position_encoder = nn.Sequential(
            nn.Linear(pos_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # 嵌入融合器
        self.embedding_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 最终特征融合
        self.final_fusion = nn.Sequential(
            nn.Linear(64 + 64, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        """处理每个观察分量"""
        # 分解观察
        current_pos = observations[:, :3]
        goal_pos = observations[:, 3:6]
        current_embedding = observations[:, 6:70]
        goal_embedding = observations[:, 70:]
        
        # 位置编码
        positions = torch.cat([current_pos, goal_pos], dim=1)
        position_features = self.position_encoder(positions)
        
        # 嵌入融合
        embeddings = torch.cat([current_embedding, goal_embedding], dim=1)
        embedding_features = self.embedding_fusion(embeddings)
        
        # 最终特征融合
        combined_features = torch.cat([position_features, embedding_features], dim=1)
        return self.final_fusion(combined_features)

def train_model(scenario='mixed', total_timesteps=100000, model_path='models/gat_drl_model.zip'):
    """训练GAT-DRL模型"""
    # 创建环境
    def make_env():
        return DronePathPlanningEnv(scenario=scenario)
    
    env = DummyVecEnv([make_env])
    
    # 创建特征提取器
    policy_kwargs = dict(
        features_extractor_class=GATDRLFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 64]  # 策略网络架构
    )
    
    # 创建PPO模型
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # 训练模型
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # 保存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"模型已保存到 {model_path}")
    
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser(description='训练GAT-DRL模型')
    parser.add_argument('--scenario', type=str, default='mixed', choices=['dense', 'sparse', 'mixed'],
                       help='训练场景 (dense, sparse, mixed)')
    parser.add_argument('--timesteps', type=int, default=100000, help='训练时间步数')
    parser.add_argument('--model-path', type=str, default='models/gat_drl_model.zip', help='模型保存路径')
    args = parser.parse_args()
    
    train_model(scenario=args.scenario, total_timesteps=args.timesteps, model_path=args.model_path)

if __name__ == "__main__":
    main()
