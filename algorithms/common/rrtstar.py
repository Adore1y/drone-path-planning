#!/usr/bin/env python3
"""
RRT*（快速随机扩展树*）算法实现，用于无人机路径规划
"""

import numpy as np
import math
import random

class Node:
    """RRT*算法的节点"""
    
    def __init__(self, position):
        self.position = position  # 节点位置 (x, y, z)
        self.parent = None        # 父节点
        self.cost = 0.0           # 从起点到该节点的成本
        self.children = []        # 子节点列表

def rrtstar_path_planning(start, end, obstacles, boundary, max_iterations=5000, step_size=1.0, search_radius=5.0, goal_sample_rate=0.1):
    """
    使用RRT*算法进行路径规划
    
    参数:
        start: 起点坐标 (x, y, z)
        end: 终点坐标 (x, y, z)
        obstacles: 障碍物列表，每个障碍物为字典 {'position': [x, y, z], 'size': [width, length, height]}
        boundary: 边界 [min_x, max_x, min_y, max_y, min_z, max_z]
        max_iterations: 最大迭代次数
        step_size: 每一步的大小
        search_radius: 搜索半径
        goal_sample_rate: 目标采样率
    
    返回:
        path: 路径点列表，如果找不到路径则返回None
    """
    # 创建树的根节点
    root = Node(start)
    nodes = [root]
    
    # 创建目标节点
    goal_node = Node(end)
    
    # 设置最佳目标节点
    best_goal_node = None
    best_goal_distance = float('inf')
    
    # 主循环
    for i in range(max_iterations):
        # 根据概率选择随机点或目标点
        if random.random() < goal_sample_rate:
            random_point = end
        else:
            random_point = [
                random.uniform(boundary[0], boundary[1]),
                random.uniform(boundary[2], boundary[3]),
                random.uniform(boundary[4], boundary[5])
            ]
        
        # 找到最近的节点
        nearest_node = find_nearest_node(nodes, random_point)
        
        # 在方向上扩展一步
        new_position = steer(nearest_node.position, random_point, step_size)
        
        # 检查新位置是否有效
        if check_collision(new_position, obstacles):
            continue
        
        # 检查连接是否有效
        if not check_path_segment(nearest_node.position, new_position, obstacles):
            continue
        
        # 创建新节点
        new_node = Node(new_position)
        
        # 找到邻近节点
        nearby_nodes = find_nearby_nodes(nodes, new_position, search_radius)
        
        # 选择最佳父节点
        parent_node, min_cost = choose_parent(nearby_nodes, new_position, nearest_node)
        
        # 连接到新节点
        add_node_and_edge(nodes, new_node, parent_node)
        
        # 重新布线
        rewire(nodes, new_node, nearby_nodes)
        
        # 检查是否接近目标
        distance_to_goal = calculate_distance(new_node.position, end)
        
        if distance_to_goal < step_size:
            # 检查是否可以直接连接到目标
            if not check_path_segment(new_node.position, end, obstacles):
                continue
            
            # 计算到目标的总成本
            total_cost = new_node.cost + distance_to_goal
            
            # 如果这条路径更好，则更新最佳目标节点
            if best_goal_node is None or total_cost < best_goal_node.cost:
                goal_node.parent = new_node
                goal_node.cost = total_cost
                best_goal_node = goal_node
                best_goal_distance = distance_to_goal
    
    # 如果找到路径，则进行提取
    if best_goal_node is not None:
        path = []
        current = best_goal_node
        while current:
            path.append(current.position)
            current = current.parent
        
        # 路径是从终点到起点的，需要反转
        return path[::-1]
    
    # 没有找到路径
    return None

def find_nearest_node(nodes, point):
    """找到距离给定点最近的节点"""
    distances = [calculate_distance(node.position, point) for node in nodes]
    min_index = np.argmin(distances)
    return nodes[min_index]

def find_nearby_nodes(nodes, point, radius):
    """找到给定半径内的所有节点"""
    return [node for node in nodes if calculate_distance(node.position, point) <= radius]

def steer(from_point, to_point, step_size):
    """从一个点朝另一个点移动一步"""
    # 计算单位向量
    distance = calculate_distance(from_point, to_point)
    
    if distance == 0:
        return from_point
    
    # 计算每个维度的步长
    step_vector = [(to_point[i] - from_point[i]) * step_size / distance for i in range(3)]
    
    # 如果距离小于步长，则直接返回目标点
    if distance <= step_size:
        return to_point
    
    # 否则移动一步
    return [from_point[i] + step_vector[i] for i in range(3)]

def choose_parent(nearby_nodes, new_position, nearest_node):
    """选择最佳父节点"""
    if not nearby_nodes:
        return nearest_node, calculate_distance(nearest_node.position, new_position) + nearest_node.cost
    
    # 计算到每个邻近节点的成本
    costs = []
    for node in nearby_nodes:
        distance = calculate_distance(node.position, new_position)
        
        # 检查路径是否无碰撞
        if not check_path_segment(node.position, new_position, []):
            cost = float('inf')
        else:
            cost = node.cost + distance
        
        costs.append(cost)
    
    # 找到最小成本的节点
    min_cost_index = np.argmin(costs)
    min_cost = costs[min_cost_index]
    
    # 如果最近节点不是近邻节点之一，检查它的成本
    if nearest_node not in nearby_nodes:
        nearest_cost = nearest_node.cost + calculate_distance(nearest_node.position, new_position)
        if nearest_cost < min_cost:
            return nearest_node, nearest_cost
    
    return nearby_nodes[min_cost_index], min_cost

def add_node_and_edge(nodes, new_node, parent_node):
    """添加节点和边"""
    new_node.parent = parent_node
    new_node.cost = parent_node.cost + calculate_distance(parent_node.position, new_node.position)
    parent_node.children.append(new_node)
    nodes.append(new_node)

def rewire(nodes, new_node, nearby_nodes):
    """重新布线"""
    for node in nearby_nodes:
        # 如果节点已经是新节点的父节点，跳过
        if node == new_node.parent:
            continue
        
        # 计算通过新节点的成本
        distance = calculate_distance(new_node.position, node.position)
        new_cost = new_node.cost + distance
        
        # 如果通过新节点的成本更低，则重新布线
        if new_cost < node.cost and check_path_segment(new_node.position, node.position, []):
            # 更新父节点关系
            old_parent = node.parent
            if old_parent:
                old_parent.children.remove(node)
            
            node.parent = new_node
            node.cost = new_cost
            new_node.children.append(node)
            
            # 递归更新所有子节点的成本
            update_children_cost(node)

def update_children_cost(node):
    """更新子节点的成本"""
    for child in node.children:
        child.cost = node.cost + calculate_distance(node.position, child.position)
        update_children_cost(child)

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