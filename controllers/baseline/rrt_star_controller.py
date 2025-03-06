#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于RRT*算法的无人机路径规划控制器
"""

from controller import Robot
import sys
import os
import numpy as np
import math
import time

# 添加父目录到Python路径，以便导入其他模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# 导入RRT*算法和评估工具
from utils.path_planning_utils import RRTStar
from utils.evaluation_utils import calculate_energy_consumption

class RRTStarController:
    """基于RRT*算法的无人机控制器"""
    
    def __init__(self, robot):
        """
        初始化控制器
        
        参数:
            robot: Webots Robot对象
        """
        self.robot = robot
        
        # 获取基本时间步长
        self.time_step = int(self.robot.getBasicTimeStep())
        
        # 初始化传感器
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.time_step)
        
        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        
        # 运动参数
        self.max_velocity = 5.0  # 最大速度 (m/s)
        self.max_angular_velocity = 1.0  # 最大角速度 (rad/s)
        self.max_ascent_rate = 2.0  # 最大上升速率 (m/s)
        self.max_descent_rate = 1.5  # 最大下降速率 (m/s)
        
        # 当前状态
        self.position = [0, 0, 0]  # [x, y, z]
        self.orientation = [0, 0, 0]  # [roll, pitch, yaw]
        self.velocity = [0, 0, 0]  # [vx, vy, vz]
        
        # 目标状态
        self.target_position = None
        self.target_velocity = [0, 0, 0]
        
        # 路径规划相关
        self.path = []  # 规划的路径点
        self.current_waypoint_index = 0
        self.buildings = []
        self.boundaries = [-50, -50, 50, 50]  # 默认边界
        
        # RRT*参数
        self.expand_dis = 5.0  # 扩展距离
        self.goal_sample_rate = 10  # 目标采样率
        self.max_iter = 1000  # 最大迭代次数
        
        # 高度控制
        self.safe_height = 5.0  # 安全飞行高度
        self.waypoint_reach_threshold = 1.0  # 到达航点的距离阈值
        
        # 性能评估
        self.start_time = None
        self.planning_time = 0
        self.path_length = 0
        self.energy_consumption = 0
        self.success = False
        
    def load_environment(self, buildings, boundaries):
        """
        加载环境信息
        
        参数:
            buildings: 建筑物列表，每个建筑物为 [x, y, width, height, length]
            boundaries: 环境边界 [min_x, min_y, max_x, max_y]
        """
        self.buildings = buildings
        self.boundaries = boundaries
        
        print(f"Environment loaded with {len(buildings)} buildings")
        
    def read_sensors(self):
        """读取传感器数据并更新当前状态"""
        # 读取GPS
        if self.gps.getSamplingPeriod() > 0:
            gps_values = self.gps.getValues()
            self.position = gps_values
        
        # 读取IMU
        if self.imu.getSamplingPeriod() > 0:
            imu_values = self.imu.getRollPitchYaw()
            self.orientation = imu_values
            
    def set_target(self, target_position):
        """
        设置目标位置并规划路径
        
        参数:
            target_position: 目标位置 [x, y, z]
            
        返回:
            success: 是否成功规划路径
        """
        self.target_position = target_position
        
        # 初始化RRT*算法
        start_pos = [self.position[0], self.position[1], self.position[2]]
        goal_pos = [target_position[0], target_position[1], target_position[2]]
        
        # 设置采样空间
        min_x, min_z, max_x, max_z = self.boundaries
        rand_area = [min_x, max_x, min_z, max_z]
        
        print(f"Planning path from {start_pos} to {goal_pos}")
        print(f"Random area: {rand_area}")
        
        # 运行RRT*算法
        start_time = time.time()
        rrt_star = RRTStar(
            start=start_pos,
            goal=goal_pos,
            obstacle_list=self.buildings,
            rand_area=rand_area,
            expand_dis=self.expand_dis,
            goal_sample_rate=self.goal_sample_rate,
            max_iter=self.max_iter
        )
        
        path, path_length = rrt_star.planning()
        self.planning_time = time.time() - start_time
        
        if path is None:
            print("Error: No path found")
            return False
            
        # 调整路径高度为安全高度
        self.path = []
        
        # 从起点到第一个路径点的过渡
        self.path.append([start_pos[0], start_pos[1], start_pos[2]])  # 起始位置
        
        # 如果起始高度不是安全高度，先上升到安全高度
        if abs(start_pos[1] - self.safe_height) > 0.5:
            self.path.append([start_pos[0], self.safe_height, start_pos[2]])
        
        # 添加路径点，设置为安全高度
        for i in range(1, len(path)-1):
            x, y, z = path[i]
            self.path.append([x, self.safe_height, z])
        
        # 从最后一个路径点到目标点的过渡
        if len(path) > 1:
            last_x, _, last_z = path[-2]
            # 先到达目标点上方
            self.path.append([goal_pos[0], self.safe_height, goal_pos[2]])
        
        # 最后下降到目标高度
        self.path.append([goal_pos[0], goal_pos[1], goal_pos[2]])
        
        # 重置路径跟踪
        self.current_waypoint_index = 0
        self.success = False
        self.start_time = time.time()
        
        # 计算路径长度和能量消耗
        self.path_length = 0
        for i in range(1, len(self.path)):
            x1, y1, z1 = self.path[i-1]
            x2, y2, z2 = self.path[i]
            segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            self.path_length += segment_length
        
        self.energy_consumption = calculate_energy_consumption(self.path)
        
        print(f"Path planned with {len(self.path)} waypoints, length: {self.path_length:.2f} m")
        return True
        
    def set_velocity(self, velocity):
        """
        设置速度
        
        参数:
            velocity: 速度向量 [vx, vy, vz]
        """
        # 限制水平速度
        vx, vy, vz = velocity
        horizontal_speed = math.sqrt(vx**2 + vy**2)
        
        if horizontal_speed > self.max_velocity:
            scale = self.max_velocity / horizontal_speed
            vx *= scale
            vy *= scale
        
        # 限制垂直速度
        if vz > 0:  # 上升
            vz = min(vz, self.max_ascent_rate)
        else:  # 下降
            vz = max(vz, -self.max_descent_rate)
        
        self.target_velocity = [vx, vy, vz]
    
    def move_to_waypoint(self):
        """
        移动到当前航点
        
        返回:
            done: 是否到达目标
        """
        if not self.path or self.current_waypoint_index >= len(self.path):
            self.set_velocity([0, 0, 0])
            return True
        
        # 获取当前航点
        waypoint = self.path[self.current_waypoint_index]
        
        # 计算到航点的向量
        dx = waypoint[0] - self.position[0]
        dy = waypoint[1] - self.position[1]
        dz = waypoint[2] - self.position[2]
        
        # 计算距离
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # 如果足够接近航点，前进到下一个航点
        if distance < self.waypoint_reach_threshold:
            self.current_waypoint_index += 1
            
            # 如果已经到达最后一个航点，即目标点
            if self.current_waypoint_index >= len(self.path):
                self.set_velocity([0, 0, 0])
                self.success = True
                return True
            
            # 获取新的航点
            waypoint = self.path[self.current_waypoint_index]
            
            # 重新计算到航点的向量
            dx = waypoint[0] - self.position[0]
            dy = waypoint[1] - self.position[1]
            dz = waypoint[2] - self.position[2]
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # 计算期望速度（按比例靠近航点）
        speed = min(self.max_velocity, 0.5 * distance)
        
        # 计算单位方向向量
        if distance > 0:
            vx = dx / distance * speed
            vy = dy / distance * speed
            vz = dz / distance * speed
        else:
            vx, vy, vz = 0, 0, 0
        
        # 设置速度
        self.set_velocity([vx, vy, vz])
        
        return False
    
    def run_step(self):
        """
        执行一个控制步骤
        
        返回:
            done: 是否到达目标
        """
        # 读取传感器
        self.read_sensors()
        
        # 移动到当前航点
        return self.move_to_waypoint()
    
    def get_performance_metrics(self):
        """
        获取性能指标
        
        返回:
            metrics: 性能指标字典
        """
        execution_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            "success_rate": 1.0 if self.success else 0.0,
            "path_length": self.path_length,
            "execution_time": execution_time,
            "planning_time": self.planning_time,
            "energy_consumption": self.energy_consumption,
            "avg_altitude": self.safe_height
        }

# 控制器入口函数
def main():
    # 创建并初始化控制器
    robot = Robot()
    controller = RRTStarController(robot)
    
    # 尝试加载环境
    try:
        # 动态确定当前世界文件
        world_name = robot.getCustomData() or "dense"
        
        if "dense" in world_name.lower():
            from worlds.dense_city_buildings import BUILDINGS, BOUNDARIES
            scenario = "dense"
        elif "sparse" in world_name.lower():
            from worlds.sparse_city_buildings import BUILDINGS, BOUNDARIES
            scenario = "sparse"
        else:
            from worlds.mixed_city_buildings import BUILDINGS, BOUNDARIES
            scenario = "mixed"
        
        print(f"Loaded {scenario} city environment")
        controller.load_environment(BUILDINGS, BOUNDARIES)
        
    except ImportError:
        print("Warning: Could not load buildings, using empty environment")
        controller.load_environment([], [-50, -50, 50, 50])
    
    # 获取目标点（假设在supervisor中设置）
    target_position = [30, 5, 30]  # 默认目标
    
    # 等待初始化传感器
    for i in range(10):
        robot.step(controller.time_step)
    
    # 设置目标点并规划路径
    controller.read_sensors()
    print(f"UAV starting position: {controller.position}")
    success = controller.set_target(target_position)
    
    if not success:
        print("Failed to plan path, stopping")
        return
    
    # 主循环
    while robot.step(controller.time_step) != -1:
        done = controller.run_step()
        
        if done:
            # 获取性能指标
            metrics = controller.get_performance_metrics()
            print("Mission completed!")
            print(f"Success: {controller.success}")
            print(f"Path length: {metrics['path_length']:.2f} m")
            print(f"Execution time: {metrics['execution_time']:.2f} s")
            print(f"Planning time: {metrics['planning_time']:.2f} s")
            print(f"Energy consumption: {metrics['energy_consumption']:.2f}")
            
            # 结束模拟
            break

if __name__ == "__main__":
    main() 