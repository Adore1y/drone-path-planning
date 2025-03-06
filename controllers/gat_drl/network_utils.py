#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网络工具模块
提供创建和操作导航图的函数
"""

import numpy as np
import networkx as nx
import math

def build_navigation_graph(buildings, boundaries):
    """
    构建导航图
    
    Args:
        buildings: 建筑物列表，每个元素为(x, z, width, height, length)
        boundaries: 场景边界 [min_x, min_z, max_x, max_z]
    
    Returns:
        graph: NetworkX图对象
    """
    # 创建空图
    graph = nx.Graph()
    
    # 解析边界
    min_x, min_z, max_x, max_z = boundaries
    
    # 定义图的分辨率
    resolution = 5.0  # 节点之间的距离
    
    # 根据边界创建节点网格
    for x in np.arange(min_x, max_x, resolution):
        for z in np.arange(min_z, max_z, resolution):
            # 检查点是否在任何建筑物内
            if not is_inside_building((x, z), buildings):
                # 创建节点
                node_id = f"n_{x:.1f}_{z:.1f}"
                graph.add_node(node_id, position=(x, 10.0, z))  # y固定为10.0（无人机飞行高度）
    
    # 连接相邻节点
    for node1 in graph.nodes():
        pos1 = graph.nodes[node1]['position']
        for node2 in graph.nodes():
            if node1 != node2:
                pos2 = graph.nodes[node2]['position']
                
                # 计算距离
                dx = pos1[0] - pos2[0]
                dz = pos1[2] - pos2[2]
                dist = math.sqrt(dx*dx + dz*dz)
                
                # 如果节点接近，并且连线不穿过建筑物，则添加边
                if dist < resolution * 1.5:
                    if not line_intersects_building(pos1, pos2, buildings):
                        graph.add_edge(node1, node2, weight=dist)
    
    print(f"创建导航图: 节点数={len(graph.nodes)}, 边数={len(graph.edges)}")
    return graph

def is_inside_building(point, buildings):
    """
    检查点是否在任何建筑物内
    
    Args:
        point: 点坐标 (x, z)
        buildings: 建筑物列表，每个元素为(x, z, width, height, length)
    
    Returns:
        bool: 如果点在任何建筑物内则为True
    """
    for bx, bz, width, height, length in buildings:
        # 检查点是否在建筑物的2D投影内
        if (bx - width/2 <= point[0] <= bx + width/2) and \
           (bz - length/2 <= point[1] <= bz + length/2):
            return True
    return False

def line_intersects_building(p1, p2, buildings):
    """
    检查线段是否穿过任何建筑物
    
    Args:
        p1: 线段起点 (x, y, z)
        p2: 线段终点 (x, y, z)
        buildings: 建筑物列表，每个元素为(x, z, width, height, length)
    
    Returns:
        bool: 如果线段穿过任何建筑物则为True
    """
    # 将3D点转换为2D
    p1_2d = (p1[0], p1[2])
    p2_2d = (p2[0], p2[2])
    
    for bx, bz, width, height, length in buildings:
        # 计算建筑物的边界框
        min_x = bx - width/2
        max_x = bx + width/2
        min_z = bz - length/2
        max_z = bz + length/2
        
        # 检查线段是否与矩形相交
        if line_intersects_rectangle(p1_2d, p2_2d, (min_x, min_z), (max_x, max_z)):
            return True
    
    return False

def line_intersects_rectangle(p1, p2, rect_min, rect_max):
    """
    检查线段是否与矩形相交
    
    Args:
        p1: 线段起点 (x, z)
        p2: 线段终点 (x, z)
        rect_min: 矩形左下角 (min_x, min_z)
        rect_max: 矩形右上角 (max_x, max_z)
    
    Returns:
        bool: 如果线段与矩形相交则为True
    """
    # 快速排除: 检查线段的边界框是否与矩形的边界框重叠
    if max(p1[0], p2[0]) < rect_min[0] or min(p1[0], p2[0]) > rect_max[0] or \
       max(p1[1], p2[1]) < rect_min[1] or min(p1[1], p2[1]) > rect_max[1]:
        return False
    
    # 检查线段是否完全在矩形内
    if (rect_min[0] <= p1[0] <= rect_max[0] and rect_min[1] <= p1[1] <= rect_max[1]) or \
       (rect_min[0] <= p2[0] <= rect_max[0] and rect_min[1] <= p2[1] <= rect_max[1]):
        return True
    
    # 检查线段是否与矩形的任何边相交
    # 矩形的四条边
    edges = [
        ((rect_min[0], rect_min[1]), (rect_max[0], rect_min[1])),  # 底边
        ((rect_max[0], rect_min[1]), (rect_max[0], rect_max[1])),  # 右边
        ((rect_max[0], rect_max[1]), (rect_min[0], rect_max[1])),  # 顶边
        ((rect_min[0], rect_max[1]), (rect_min[0], rect_min[1]))   # 左边
    ]
    
    for edge in edges:
        if line_intersects_line(p1, p2, edge[0], edge[1]):
            return True
    
    return False

def line_intersects_line(p1, p2, p3, p4):
    """
    检查两条线段是否相交
    
    Args:
        p1, p2: 第一条线段的端点
        p3, p4: 第二条线段的端点
    
    Returns:
        bool: 如果两条线段相交则为True
    """
    # 方向交叉乘积
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    # 检查(p1,p2)和(p3,p4)是否跨越对方
    d1 = cross_product(p3, p4, p1)
    d2 = cross_product(p3, p4, p2)
    d3 = cross_product(p1, p2, p3)
    d4 = cross_product(p1, p2, p4)
    
    # 如果(p1,p2)和(p3,p4)跨越对方，则相交
    return (d1 * d2 <= 0) and (d3 * d4 <= 0)

def find_nearest_node(graph, position):
    """
    找到最近的图节点
    
    Args:
        graph: NetworkX图对象
        position: 3D位置 [x, y, z]
    
    Returns:
        node_id: 最近节点的ID
    """
    min_dist = float('inf')
    nearest_node = None
    
    # 只关注x和z坐标
    x, _, z = position
    
    for node in graph.nodes():
        node_pos = graph.nodes[node]['position']
        node_x, _, node_z = node_pos
        
        # 计算2D平面距离
        dist = math.sqrt((x - node_x)**2 + (z - node_z)**2)
        
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node 