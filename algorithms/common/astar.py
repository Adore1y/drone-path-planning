#!/usr/bin/env python3
"""
A*算法实现，用于无人机路径规划
"""

import heapq
import numpy as np
import math
from collections import defaultdict

class Node:
    """A*算法的节点"""
    
    def __init__(self, position, parent=None):
        self.position = position  # 节点位置 (x, y, z)
        self.parent = parent      # 父节点
        
        self.g = 0  # 从起点到当前节点的成本
        self.h = 0  # 从当前节点到终点的估计成本（启发式）
        self.f = 0  # 总成本 f = g + h
    
    def __eq__(self, other):
        """比较两个节点是否相同（位置相同）"""
        return self.position == other.position
    
    def __lt__(self, other):
        """比较两个节点的优先级"""
        return self.f < other.f
    
    def __hash__(self):
        """节点的哈希值，用于在集合中存储"""
        return hash(tuple(self.position))

def astar_path_planning(start, end, obstacles, grid_resolution=1.0, heuristic='euclidean', max_iterations=10000):
    """
    使用A*算法进行路径规划
    
    参数:
        start: 起点坐标 (x, y, z)
        end: 终点坐标 (x, y, z)
        obstacles: 障碍物列表，每个障碍物为字典 {'position': [x, y, z], 'size': [width, length, height]}
        grid_resolution: 网格分辨率，较小的值会生成更精细但计算量更大的路径
        heuristic: 启发式函数类型 ('euclidean', 'manhattan', 'diagonal')
        max_iterations: 最大迭代次数
    
    返回:
        path: 路径点列表，如果找不到路径则返回None
    """
    # 创建起点和终点节点
    start_node = Node(start)
    end_node = Node(end)
    
    # 初始化开启列表和关闭列表
    open_list = []
    closed_set = set()
    
    # 将起点加入开启列表
    heapq.heappush(open_list, start_node)
    
    # 定义移动方向（6个方向：上下左右前后）
    directions = [
        [grid_resolution, 0, 0],  # 右
        [-grid_resolution, 0, 0],  # 左
        [0, grid_resolution, 0],  # 前
        [0, -grid_resolution, 0],  # 后
        [0, 0, grid_resolution],  # 上
        [0, 0, -grid_resolution]   # 下
    ]
    
    # 添加对角线方向
    if heuristic == 'diagonal':
        # 12个对角线方向
        diagonals = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    if i == 0 and j == 0 and k == 0:
                        continue
                    if abs(i) + abs(j) + abs(k) == 1:
                        continue  # 跳过已包含的6个方向
                    diagonals.append([i * grid_resolution, j * grid_resolution, k * grid_resolution])
        
        directions.extend(diagonals)
    
    # 迭代计数
    iterations = 0
    
    # 主循环
    while open_list and iterations < max_iterations:
        iterations += 1
        
        # 获取开启列表中f值最小的节点
        current_node = heapq.heappop(open_list)
        
        # 将当前节点加入关闭列表
        closed_position = tuple(current_node.position)
        if closed_position in closed_set:
            continue
        
        closed_set.add(closed_position)
        
        # 检查是否到达终点（在一定范围内）
        if calculate_distance(current_node.position, end_node.position) < grid_resolution:
            # 重建路径
            path = []
            current = current_node
            while current:
                path.append(current.position)
                current = current.parent
            
            # 路径是从终点到起点的，需要反转
            return path[::-1]
        
        # 生成子节点
        for direction in directions:
            # 计算新位置
            new_position = [
                current_node.position[0] + direction[0],
                current_node.position[1] + direction[1],
                current_node.position[2] + direction[2]
            ]
            
            # 检查位置是否有效（不与障碍物碰撞）
            if check_collision(new_position, obstacles):
                continue
            
            # 创建新节点
            new_node = Node(new_position, current_node)
            
            # 计算成本
            # g值 = 父节点的g值 + 移动成本
            move_cost = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
            new_node.g = current_node.g + move_cost
            
            # h值 = 启发式函数值
            if heuristic == 'manhattan':
                new_node.h = (abs(new_node.position[0] - end_node.position[0]) +
                             abs(new_node.position[1] - end_node.position[1]) +
                             abs(new_node.position[2] - end_node.position[2]))
            elif heuristic == 'diagonal':
                dx = abs(new_node.position[0] - end_node.position[0])
                dy = abs(new_node.position[1] - end_node.position[1])
                dz = abs(new_node.position[2] - end_node.position[2])
                # 3D对角线距离
                diag = min(dx, dy, dz)
                straight = abs(dx - dy) + abs(dy - dz) + abs(dz - dx) - diag
                new_node.h = math.sqrt(3) * diag + straight
            else:  # 默认使用欧几里得距离
                new_node.h = calculate_distance(new_node.position, end_node.position)
            
            # f值 = g值 + h值
            new_node.f = new_node.g + new_node.h
            
            # 如果新节点位置已在关闭列表中，跳过
            if tuple(new_node.position) in closed_set:
                continue
            
            # 将新节点加入开启列表
            heapq.heappush(open_list, new_node)
    
    # 超过最大迭代次数或开启列表为空，表示没有找到路径
    return None

def calculate_distance(point1, point2):
    """计算两点之间的欧几里得距离"""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

def check_collision(position, obstacles, safety_margin=0.5):
    """检查位置是否与障碍物碰撞"""
    for obstacle in obstacles:
        obs_pos = obstacle['position']
        obs_size = obstacle['size']
        
        half_width = obs_size[0] / 2 + safety_margin
        half_length = obs_size[1] / 2 + safety_margin
        half_height = obs_size[2] / 2 + safety_margin
        
        # 检查是否在障碍物范围内
        if (abs(position[0] - obs_pos[0]) < half_width and
            abs(position[1] - obs_pos[1]) < half_length and
            abs(position[2] - obs_pos[2]) < half_height):
            return True
    
    return False

def smooth_path(path, obstacles, safety_margin=0.5, max_iterations=100):
    """
    平滑A*算法生成的路径
    
    参数:
        path: 路径点列表
        obstacles: 障碍物列表
        safety_margin: 安全距离
        max_iterations: 最大迭代次数
    
    返回:
        平滑后的路径
    """
    if not path or len(path) < 3:
        return path
    
    smoothed_path = path.copy()
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        
        # 标记是否有修改
        modified = False
        
        # 尝试移除中间点
        i = 1
        while i < len(smoothed_path) - 1:
            p1 = smoothed_path[i-1]
            p2 = smoothed_path[i+1]
            
            # 检查直接连接p1和p2是否无碰撞
            if check_path_segment(p1, p2, obstacles, safety_margin):
                # 可以移除中间点
                smoothed_path.pop(i)
                modified = True
            else:
                i += 1
        
        # 如果没有修改，则退出
        if not modified:
            break
    
    return smoothed_path

def check_path_segment(p1, p2, obstacles, safety_margin=0.5, num_checks=10):
    """检查路径段是否无碰撞"""
    for t in np.linspace(0, 1, num_checks):
        # 插值点
        point = [
            p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]),
            p1[2] + t * (p2[2] - p1[2])
        ]
        
        # 检查点是否与障碍物碰撞
        if check_collision(point, obstacles, safety_margin):
            return False
    
    return True 