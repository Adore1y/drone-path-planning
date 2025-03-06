#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAT环境编码器
用于将建筑物和环境数据编码为适合GAT-DRL使用的表示形式
"""

import numpy as np
import networkx as nx

class GATEnvironmentEncoder:
    """GAT环境编码器类，用于将环境编码为适合GAT-DRL模型的格式"""
    
    def __init__(self):
        """初始化GAT环境编码器"""
        pass
    
    def encode_environment(self, graph, start_node, goal_node, obs_size=(30, 30, 1)):
        """将环境编码为观察空间
        
        Args:
            graph: NetworkX图对象
            start_node: 起始节点ID
            goal_node: 目标节点ID
            obs_size: 观察空间大小 (height, width, channels)
        
        Returns:
            observation: 适合输入GAT-DRL模型的观察
        """
        # 获取图的节点位置
        node_positions = nx.get_node_attributes(graph, 'position')
        
        # 获取起点和终点的位置
        start_pos = node_positions[start_node]
        goal_pos = node_positions[goal_node]
        
        # 创建观察空间
        observation = np.zeros(obs_size, dtype=np.float32)
        
        # 设置观察空间的中心为起点位置
        height, width, channels = obs_size
        center_h, center_w = height // 2, width // 2
        
        # 确定地图的缩放比例（假设环境大小约为200x200）
        scale = min(height, width) / 200.0
        
        # 将节点映射到观察空间
        for node, pos in node_positions.items():
            # 计算相对于起点的偏移
            rel_x = pos[0] - start_pos[0]
            rel_z = pos[2] - start_pos[2]
            
            # 缩放并转换到观察空间坐标
            map_h = int(center_h - rel_z * scale)
            map_w = int(center_w + rel_x * scale)
            
            # 检查是否在观察空间范围内
            if 0 <= map_h < height and 0 <= map_w < width:
                # 设置节点位置（值为0.5）
                observation[map_h, map_w, 0] = 0.5
        
        # 标记起点（值为1.0）
        observation[center_h, center_w, 0] = 1.0
        
        # 标记终点
        goal_rel_x = goal_pos[0] - start_pos[0]
        goal_rel_z = goal_pos[2] - start_pos[2]
        goal_h = int(center_h - goal_rel_z * scale)
        goal_w = int(center_w + goal_rel_x * scale)
        
        # 确保终点在观察空间范围内
        if 0 <= goal_h < height and 0 <= goal_w < width:
            observation[goal_h, goal_w, 0] = 0.8
        
        # 返回观察空间
        return observation 