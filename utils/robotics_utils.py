#!/usr/bin/env python3
"""
机器人学相关的实用工具函数
"""
import numpy as np
import math

def euler_to_quaternion(roll, pitch, yaw):
    """欧拉角转四元数"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return [w, x, y, z]

def quaternion_to_euler(q):
    """四元数转欧拉角"""
    # 提取四元数分量
    w, x, y, z = q
    
    # 计算欧拉角
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    
    return [roll, pitch, yaw]

def rotation_matrix(roll, pitch, yaw):
    """计算旋转矩阵"""
    # 计算三角函数值
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    
    # 构建旋转矩阵
    R = np.array([
        [cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
        [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
        [-sp, sr*cp, cr*cp]
    ])
    
    return R

def calculate_distance(point1, point2):
    """计算两点之间的欧几里得距离"""
    return np.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

def calculate_energy(velocity, time_step):
    """估算能量消耗"""
    # 简化模型：能量与速度平方成正比
    return sum([v**2 for v in velocity]) * time_step

def check_collision(position, obstacles, safety_margin=1.0):
    """检查给定位置是否与障碍物碰撞"""
    for obstacle in obstacles:
        obs_pos = obstacle['position']
        obs_size = obstacle['size']
        
        # 检查碰撞
        if (abs(position[0] - obs_pos[0]) < obs_size[0]/2 + safety_margin and
            abs(position[1] - obs_pos[1]) < obs_size[1]/2 + safety_margin and
            abs(position[2] - obs_pos[2]) < obs_size[2]/2 + safety_margin):
            return True
    
    return False
