#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路径规划工具函数
包含A*和RRT*算法的实现
"""

import numpy as np
import heapq
import random
import math
from collections import defaultdict

def astar_path(graph, start, goal, heuristic=None):
    """
    使用A*算法在图中寻找从起点到终点的最短路径
    
    参数:
        graph: 图对象，需要有graph.neighbors(node)方法返回节点的邻居
        start: 起始节点
        goal: 目标节点
        heuristic: 启发式函数，接受两个节点参数并返回估计距离
        
    返回:
        path: 路径节点列表
        path_length: 路径长度
    """
    if start == goal:
        return [start], 0
    
    if heuristic is None:
        # 默认启发式函数（假设所有节点距离为1）
        heuristic = lambda u, v: 1
    
    # 初始化开放列表和关闭列表
    open_set = []
    closed_set = set()
    
    # 记录从起点到每个节点的最短路径
    g_score = {start: 0}
    
    # 记录从起点经过每个节点到目标的估计总成本
    f_score = {start: heuristic(start, goal)}
    
    # 记录每个节点的前驱节点
    came_from = {}
    
    # 将起点加入开放列表
    heapq.heappush(open_set, (f_score[start], start))
    
    while open_set:
        # 获取f值最小的节点
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # 重建路径
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            
            # 计算路径长度
            path_length = g_score[goal]
            
            return path, path_length
        
        # 将当前节点加入关闭列表
        closed_set.add(current)
        
        # 检查所有邻居
        for neighbor in graph.neighbors(current):
            if neighbor in closed_set:
                continue
            
            # 计算从起点经过当前节点到邻居的距离
            edge_weight = 1  # 默认边权重为1
            for u, v, data in graph.edges(current, data=True):
                if v == neighbor:
                    edge_weight = data.get('weight', 1)
                    break
            
            tentative_g_score = g_score[current] + edge_weight
            
            # 如果邻居不在开放列表中，或者找到了更短的路径
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # 更新路径信息
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                
                # 将邻居加入开放列表
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # 如果没有找到路径
    return None, float('inf')

class RRTNode:
    """RRT树的节点"""
    
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z if z is not None else 0.0
        self.path_x = []
        self.path_y = []
        self.path_z = []
        self.parent = None
        self.cost = 0.0

class RRTStar:
    """RRT*算法实现"""
    
    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=5.0, goal_sample_rate=5, max_iter=500,
                 connect_circle_dist=50.0, search_until_max_iter=True):
        """
        初始化RRT*算法
        
        参数:
            start: 起始位置 [x, y, z]
            goal: 目标位置 [x, y, z]
            obstacle_list: 障碍物列表 [[x, y, width, height, length], ...]
            rand_area: 随机采样区域 [min_x, max_x, min_y, max_y]
            expand_dis: 扩展距离
            goal_sample_rate: 采样目标点的概率 (%)
            max_iter: 最大迭代次数
            connect_circle_dist: 连接圆的半径
            search_until_max_iter: 是否一直搜索到最大迭代次数
        """
        self.start = self._create_node(start[0], start[2], start[1])
        self.goal = self._create_node(goal[0], goal[2], goal[1])
        self.obstacle_list = obstacle_list
        self.min_x, self.max_x, self.min_y, self.max_y = rand_area
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist
        self.search_until_max_iter = search_until_max_iter
        
        self.node_list = [self.start]
        self.safe_height = 5.0  # 安全飞行高度
    
    def _create_node(self, x, y, z=None):
        """创建一个新节点"""
        return RRTNode(x, y, z)
    
    def planning(self):
        """
        执行RRT*路径规划
        
        返回:
            path: 路径点列表 [[x, y, z], ...]
            path_length: 路径长度
        """
        for i in range(self.max_iter):
            # 生成随机节点
            if random.randint(0, 100) > self.goal_sample_rate:
                rnd = self._create_node(
                    random.uniform(self.min_x, self.max_x),
                    random.uniform(self.min_y, self.max_y),
                    self.safe_height
                )
            else:
                rnd = self._create_node(self.goal.x, self.goal.y, self.goal.z)
            
            # 找到最近的节点
            nearest_ind = self._get_nearest_node_index(rnd)
            nearest_node = self.node_list[nearest_ind]
            
            # 朝随机点方向扩展
            new_node = self._steer(nearest_node, rnd, self.expand_dis)
            
            # 检查是否与障碍物碰撞
            if self._is_collision_free(new_node):
                # 找到附近的节点
                near_inds = self._find_near_nodes(new_node)
                
                # 从最近的节点连接到新节点
                new_node = self._choose_parent(new_node, near_inds)
                
                if new_node:
                    # 将新节点添加到树中
                    self.node_list.append(new_node)
                    
                    # 重新布线
                    self._rewire(new_node, near_inds)
                    
                    # 尝试连接到目标
                    if self._is_near_goal(new_node):
                        if self._is_path_collision_free(new_node, self.goal):
                            last_node = self._steer(new_node, self.goal, self.expand_dis)
                            if self._is_collision_free(last_node):
                                # 找到路径
                                path, path_length = self._generate_final_path(last_node)
                                if not self.search_until_max_iter:
                                    return path, path_length
        
        # 尝试找到最接近目标的路径
        min_dist = float('inf')
        best_goal_node = None
        
        for node in self.node_list:
            dist = self._calc_dist_to_goal(node)
            if dist < min_dist:
                min_dist = dist
                best_goal_node = node
        
        if best_goal_node and min_dist < self.expand_dis:
            path, path_length = self._generate_final_path(best_goal_node)
            return path, path_length
        
        return None, float('inf')
    
    def _steer(self, from_node, to_node, extend_length=float('inf')):
        """
        从一个节点朝另一个节点方向扩展
        
        参数:
            from_node: 起始节点
            to_node: 目标节点
            extend_length: 最大扩展距离
            
        返回:
            new_node: 新节点
        """
        new_node = self._create_node(from_node.x, from_node.y, from_node.z)
        
        # 计算方向向量
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        
        # 计算距离
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # 如果距离小于扩展距离，直接返回目标节点
        if dist < extend_length:
            new_node.x = to_node.x
            new_node.y = to_node.y
            new_node.z = to_node.z
        else:
            # 否则，朝目标方向扩展指定距离
            theta = math.atan2(dy, dx)
            phi = math.asin(dz / dist) if dist > 0 else 0
            
            new_node.x = from_node.x + extend_length * math.cos(theta) * math.cos(phi)
            new_node.y = from_node.y + extend_length * math.sin(theta) * math.cos(phi)
            new_node.z = from_node.z + extend_length * math.sin(phi)
        
        # 设置父节点和路径
        new_node.parent = from_node
        new_node.cost = from_node.cost + self._calc_distance(new_node, from_node)
        
        # 记录路径
        new_node.path_x = [from_node.x, new_node.x]
        new_node.path_y = [from_node.y, new_node.y]
        new_node.path_z = [from_node.z, new_node.z]
        
        return new_node
    
    def _is_collision_free(self, node):
        """
        检查节点是否与障碍物碰撞
        
        参数:
            node: 要检查的节点
            
        返回:
            is_safe: 是否安全（无碰撞）
        """
        if node is None:
            return False
        
        # 检查节点是否在障碍物内部
        for obstacle in self.obstacle_list:
            ox, oy, width, height, length = obstacle
            
            # 检查节点是否在障碍物的水平范围内
            if (ox - width/2 <= node.x <= ox + width/2) and (oy - length/2 <= node.y <= oy + length/2):
                # 检查节点是否在障碍物的垂直范围内
                if node.z <= height:
                    return False
        
        # 如果有父节点，检查从父节点到当前节点的路径是否安全
        if node.parent:
            # 简化：只检查路径的几个点
            for i in range(10):
                t = i / 10.0
                x = node.parent.x + t * (node.x - node.parent.x)
                y = node.parent.y + t * (node.y - node.parent.y)
                z = node.parent.z + t * (node.z - node.parent.z)
                
                # 检查该点是否在障碍物内部
                for obstacle in self.obstacle_list:
                    ox, oy, width, height, length = obstacle
                    
                    if (ox - width/2 <= x <= ox + width/2) and (oy - length/2 <= y <= oy + length/2):
                        if z <= height:
                            return False
        
        return True
    
    def _is_path_collision_free(self, from_node, to_node):
        """
        检查从一个节点到另一个节点的路径是否与障碍物碰撞
        
        参数:
            from_node: 起始节点
            to_node: 目标节点
            
        返回:
            is_safe: 是否安全（无碰撞）
        """
        # 简化：只检查路径的几个点
        for i in range(10):
            t = i / 10.0
            x = from_node.x + t * (to_node.x - from_node.x)
            y = from_node.y + t * (to_node.y - from_node.y)
            z = from_node.z + t * (to_node.z - from_node.z)
            
            # 检查该点是否在障碍物内部
            for obstacle in self.obstacle_list:
                ox, oy, width, height, length = obstacle
                
                if (ox - width/2 <= x <= ox + width/2) and (oy - length/2 <= y <= oy + length/2):
                    if z <= height:
                        return False
        
        return True
    
    def _get_nearest_node_index(self, node):
        """
        找到距离给定节点最近的节点的索引
        
        参数:
            node: 目标节点
            
        返回:
            min_index: 最近节点的索引
        """
        dlist = [(self._calc_distance(node, n)) for n in self.node_list]
        min_index = dlist.index(min(dlist))
        return min_index
    
    def _find_near_nodes(self, new_node):
        """
        找到距离新节点一定范围内的所有节点的索引
        
        参数:
            new_node: 新节点
            
        返回:
            near_inds: 附近节点的索引列表
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        r = min(r, self.expand_dis * 5.0)
        
        dlist = [(self._calc_distance(new_node, n)) for n in self.node_list]
        near_inds = [i for i, d in enumerate(dlist) if d <= r]
        return near_inds
    
    def _choose_parent(self, new_node, near_inds):
        """
        从附近的节点中选择最佳父节点
        
        参数:
            new_node: 新节点
            near_inds: 附近节点的索引列表
            
        返回:
            new_node: 更新后的新节点
        """
        if not near_inds:
            return None
        
        # 计算从每个附近节点到新节点的成本
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self._steer(near_node, new_node)
            if t_node and self._is_collision_free(t_node):
                costs.append(near_node.cost + self._calc_distance(near_node, t_node))
            else:
                costs.append(float('inf'))
        
        # 选择成本最小的节点作为父节点
        min_cost = min(costs)
        if min_cost == float('inf'):
            return None
        
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self._steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost
        
        return new_node
    
    def _rewire(self, new_node, near_inds):
        """
        重新布线，检查是否可以通过新节点改善附近节点的路径
        
        参数:
            new_node: 新节点
            near_inds: 附近节点的索引列表
        """
        for i in near_inds:
            near_node = self.node_list[i]
            
            # 计算通过新节点的成本
            edge_node = self._steer(new_node, near_node)
            if not edge_node:
                continue
            
            edge_node.cost = new_node.cost + self._calc_distance(new_node, edge_node)
            
            no_collision = self._is_collision_free(edge_node)
            improved_cost = near_node.cost > edge_node.cost
            
            # 如果通过新节点的路径更好，则重新布线
            if no_collision and improved_cost:
                self.node_list[i].parent = new_node
                self.node_list[i].cost = edge_node.cost
                
                # 更新路径
                self.node_list[i].path_x = [new_node.x, near_node.x]
                self.node_list[i].path_y = [new_node.y, near_node.y]
                self.node_list[i].path_z = [new_node.z, near_node.z]
    
    def _calc_distance(self, from_node, to_node):
        """
        计算两个节点之间的距离
        
        参数:
            from_node: 起始节点
            to_node: 目标节点
            
        返回:
            distance: 距离
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    def _calc_dist_to_goal(self, node):
        """
        计算节点到目标的距离
        
        参数:
            node: 节点
            
        返回:
            distance: 距离
        """
        return self._calc_distance(node, self.goal)
    
    def _is_near_goal(self, node):
        """
        检查节点是否接近目标
        
        参数:
            node: 节点
            
        返回:
            is_near: 是否接近
        """
        return self._calc_dist_to_goal(node) <= self.expand_dis
    
    def _generate_final_path(self, goal_node):
        """
        生成最终路径
        
        参数:
            goal_node: 目标节点
            
        返回:
            path: 路径点列表 [[x, y, z], ...]
            path_length: 路径长度
        """
        path = []
        node = goal_node
        
        # 从目标节点回溯到起始节点
        while node.parent:
            path.append([node.x, node.z, node.y])  # 注意：y和z坐标交换
            node = node.parent
        
        path.append([self.start.x, self.start.z, self.start.y])
        path.reverse()
        
        # 计算路径长度
        path_length = goal_node.cost
        
        return path, path_length

if __name__ == "__main__":
    # 测试代码
    pass 