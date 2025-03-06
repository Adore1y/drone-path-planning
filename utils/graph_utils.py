#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图形工具函数
用于创建和操作城市环境的图形表示
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import math

def create_city_graph(buildings, boundaries, resolution=5.0):
    """
    根据建筑物和边界创建城市环境的图形表示
    
    参数:
        buildings: 建筑物列表 [[x, y, width, height, length], ...]
        boundaries: 环境边界 [min_x, min_z, max_x, max_z]
        resolution: 图网格分辨率
        
    返回:
        G: NetworkX图对象
        node_positions: 节点位置字典 {node_id: (x, y), ...}
    """
    # 创建无向图
    G = nx.Graph()
    
    # 解析边界
    min_x, min_z, max_x, max_z = boundaries
    
    # 计算网格尺寸
    x_size = int((max_x - min_x) / resolution) + 1
    z_size = int((max_z - min_z) / resolution) + 1
    
    # 创建节点位置字典
    node_positions = {}
    
    # 创建网格节点
    node_id = 0
    for i in range(x_size):
        for j in range(z_size):
            x = min_x + i * resolution
            z = min_z + j * resolution
            
            # 检查节点是否在建筑物内部
            if not is_point_in_buildings(x, z, buildings):
                G.add_node(node_id)
                node_positions[node_id] = (x, z)
                node_id += 1
    
    # 创建边
    for u in G.nodes():
        x1, z1 = node_positions[u]
        
        for v in G.nodes():
            if u == v:
                continue
                
            x2, z2 = node_positions[v]
            
            # 计算距离
            dist = math.sqrt((x2 - x1)**2 + (z2 - z1)**2)
            
            # 只连接相邻节点（距离小于等于分辨率的√2倍）
            if dist <= resolution * math.sqrt(2) + 0.01:  # 添加小的容差
                # 检查边是否穿过建筑物
                if not is_line_intersect_buildings(x1, z1, x2, z2, buildings):
                    G.add_edge(u, v, weight=dist)
    
    return G, node_positions

def is_point_in_buildings(x, z, buildings):
    """
    检查点是否在建筑物内部
    
    参数:
        x, z: 点的坐标
        buildings: 建筑物列表
        
    返回:
        is_inside: 是否在建筑物内部
    """
    for building in buildings:
        bx, bz, width, height, length = building
        
        # 检查点是否在建筑物的水平范围内
        if (bx - width/2 <= x <= bx + width/2) and (bz - length/2 <= z <= bz + length/2):
            return True
    
    return False

def is_line_intersect_buildings(x1, z1, x2, z2, buildings):
    """
    检查线段是否与建筑物相交
    
    参数:
        x1, z1: 线段起点
        x2, z2: 线段终点
        buildings: 建筑物列表
        
    返回:
        is_intersect: 是否相交
    """
    for building in buildings:
        bx, bz, width, height, length = building
        
        # 建筑物的四个角点
        corners = [
            (bx - width/2, bz - length/2),
            (bx + width/2, bz - length/2),
            (bx + width/2, bz + length/2),
            (bx - width/2, bz + length/2)
        ]
        
        # 建筑物的四条边
        edges = [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0])
        ]
        
        # 检查线段是否与建筑物的任意一条边相交
        for edge in edges:
            if line_segments_intersect((x1, z1), (x2, z2), edge[0], edge[1]):
                return True
        
        # 检查线段的端点是否在建筑物内部
        if is_point_in_rect(x1, z1, bx, bz, width, length) or is_point_in_rect(x2, z2, bx, bz, width, length):
            return True
    
    return False

def is_point_in_rect(x, z, rect_x, rect_z, width, length):
    """
    检查点是否在矩形内部
    
    参数:
        x, z: 点的坐标
        rect_x, rect_z: 矩形中心坐标
        width, length: 矩形宽度和长度
        
    返回:
        is_inside: 是否在矩形内部
    """
    return (rect_x - width/2 <= x <= rect_x + width/2) and (rect_z - length/2 <= z <= rect_z + length/2)

def line_segments_intersect(p1, p2, p3, p4):
    """
    检查两条线段是否相交
    
    参数:
        p1, p2: 第一条线段的端点
        p3, p4: 第二条线段的端点
        
    返回:
        is_intersect: 是否相交
    """
    # 计算方向
    def direction(p1, p2, p3):
        return (p3[0] - p1[0]) * (p2[1] - p1[1]) - (p2[0] - p1[0]) * (p3[1] - p1[1])
    
    # 检查点是否在线段上
    def on_segment(p1, p2, p3):
        return (min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]) and
                min(p1[1], p2[1]) <= p3[1] <= max(p1[1], p2[1]))
    
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)
    
    # 如果两条线段相交
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # 特殊情况：共线
    if d1 == 0 and on_segment(p3, p4, p1):
        return True
    if d2 == 0 and on_segment(p3, p4, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, p3):
        return True
    if d4 == 0 and on_segment(p1, p2, p4):
        return True
    
    return False

def find_nearest_node(graph, node_positions, point):
    """
    找到图中距离给定点最近的节点
    
    参数:
        graph: NetworkX图对象
        node_positions: 节点位置字典
        point: 目标点 (x, z)
        
    返回:
        nearest_node: 最近节点的ID
        min_dist: 最小距离
    """
    min_dist = float('inf')
    nearest_node = None
    
    for node in graph.nodes():
        x, z = node_positions[node]
        dist = math.sqrt((x - point[0])**2 + (z - point[1])**2)
        
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node, min_dist

def get_path_coordinates(path, node_positions):
    """
    获取路径的坐标
    
    参数:
        path: 节点路径列表
        node_positions: 节点位置字典
        
    返回:
        coords: 路径坐标列表 [(x, z), ...]
    """
    return [node_positions[node] for node in path]

def visualize_graph(graph, node_positions, buildings=None, path=None, start=None, goal=None, ax=None):
    """
    可视化图形和路径
    
    参数:
        graph: NetworkX图对象
        node_positions: 节点位置字典
        buildings: 建筑物列表
        path: 路径节点列表
        start: 起点坐标 (x, z)
        goal: 终点坐标 (x, z)
        ax: matplotlib轴对象
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制建筑物
    if buildings:
        for building in buildings:
            bx, bz, width, height, length = building
            rect = Rectangle((bx - width/2, bz - length/2), width, length, 
                            facecolor='gray', alpha=0.5, edgecolor='black')
            ax.add_patch(rect)
    
    # 绘制图形
    nx.draw_networkx_nodes(graph, node_positions, node_size=10, node_color='blue', alpha=0.5, ax=ax)
    nx.draw_networkx_edges(graph, node_positions, width=0.5, alpha=0.3, ax=ax)
    
    # 绘制路径
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph, node_positions, edgelist=path_edges, 
                              width=2, edge_color='red', ax=ax)
    
    # 绘制起点和终点
    if start:
        ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    if goal:
        ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # 设置坐标轴
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')
    ax.legend()
    
    return ax

def convert_nx_graph_to_pyg_graph(G, node_positions=None, node_features=None):
    """
    将NetworkX图转换为PyTorch Geometric图
    
    参数:
        G: NetworkX图
        node_positions: 节点坐标字典（可选）
        node_features: 节点特征字典（可选）
        
    返回:
        data: PyTorch Geometric的Data对象
    """
    # 准备边索引
    edge_index = []
    edge_attr = []
    
    for u, v, data in G.edges(data=True):
        edge_index.append([u, v])
        edge_index.append([v, u])  # 为有向图添加两个方向
        
        weight = data.get('weight', 1.0)
        edge_attr.append([weight])
        edge_attr.append([weight])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # 准备节点特征
    if node_features is not None:
        x = []
        for node in G.nodes():
            x.append(node_features[node])
        x = torch.tensor(x, dtype=torch.float)
    elif node_positions is not None:
        x = []
        for node in G.nodes():
            pos = node_positions[node]
            x.append([pos[0], pos[1]])
        x = torch.tensor(x, dtype=torch.float)
    else:
        # 默认使用节点度作为特征
        x = []
        for node in G.nodes():
            x.append([G.degree(node)])
        x = torch.tensor(x, dtype=torch.float)
    
    # 创建PyG数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

if __name__ == "__main__":
    # 测试代码
    pass 