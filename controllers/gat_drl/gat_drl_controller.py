#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAT-DRL控制器 - Webots版本
用于在Webots中使用训练好的GAT-DRL模型进行路径规划
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import time
import random
from stable_baselines3 import PPO
from datetime import datetime
from pathlib import Path

# 确保控制器路径正确
try:
    from controller import Robot, Motor, GPS, Gyro, Camera, Display, Supervisor
except ImportError:
    # 如果找不到Webots控制器模块，就添加它到路径中
    webots_home = os.environ.get('WEBOTS_HOME', '/Applications/Webots.app')
    controller_path = os.path.join(webots_home, 'lib', 'controller', 'python')
    sys.path.append(controller_path)
    from controller import Robot, Motor, GPS, Gyro, Camera, Display, Supervisor

# 添加自定义库路径
controllers_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(controllers_path)
sys.path.append(project_root)
from controllers.gat_drl.gat_environment_encoder import GATEnvironmentEncoder
import controllers.gat_drl.network_utils as network_utils
import networkx as nx

class GATDRLController:
    """GAT-DRL控制器类，用于Webots中的无人机控制"""
    
    def __init__(self, robot):
        """初始化GAT-DRL控制器
        
        Args:
            robot: Webots机器人实例
        """
        # 保存机器人实例
        self.robot = robot
        
        # 获取控制周期
        self.timestep = int(robot.getBasicTimeStep())
        
        # 初始化设备
        self._init_devices()
        
        # 初始化路径规划相关变量
        self.target_position = None
        self.current_path = []
        self.current_waypoint_index = 0
        self.tolerance = 1.0  # 容差范围，单位是米
        self.graph = None
        self.buildings = None
        self.boundaries = None
        
        # 性能指标
        self.start_time = None
        self.path_length = 0.0
        self.energy_consumption = 0.0
        self.last_position = None
        self.start_position = None
        self.goal_position = None
        self.success = False
        
        # 载入GAT-DRL模型
        self.model = self._load_model()
        
        # 获取场景信息
        scenario = os.environ.get('WEBOTS_SCENARIO', 'mixed')
        trial_id = os.environ.get('WEBOTS_TRIAL_ID', '1')
        
        self.scenario = scenario
        self.trial_id = trial_id
        
        # 日志
        print(f"初始化GAT-DRL控制器 - 场景: {scenario}, 测试ID: {trial_id}")
    
    def _init_devices(self):
        """初始化无人机设备"""
        # 电机初始化
        self.front_left_motor = self.robot.getDevice('front left propeller')
        self.front_right_motor = self.robot.getDevice('front right propeller')
        self.rear_left_motor = self.robot.getDevice('rear left propeller')
        self.rear_right_motor = self.robot.getDevice('rear right propeller')
        
        # 设置电机模式
        self.front_left_motor.setPosition(float('inf'))
        self.front_right_motor.setPosition(float('inf'))
        self.rear_left_motor.setPosition(float('inf'))
        self.rear_right_motor.setPosition(float('inf'))
        
        # 设置电机速度
        self.front_left_motor.setVelocity(0.0)
        self.front_right_motor.setVelocity(0.0)
        self.rear_left_motor.setVelocity(0.0)
        self.rear_right_motor.setVelocity(0.0)
        
        # GPS初始化
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        
        # 陀螺仪初始化
        self.gyro = self.robot.getDevice('gyro')
        self.gyro.enable(self.timestep)
        
        # 相机初始化
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        
        # 显示器初始化（用于调试）
        try:
            self.display = self.robot.getDevice('display')
            self.has_display = True
        except:
            self.has_display = False
    
    def _load_model(self):
        """加载预训练的GAT-DRL模型"""
        try:
            model_path = os.path.join(project_root, 'models', 'gat_drl_model.zip')
            model = PPO.load(model_path)
            print(f"成功加载模型: {model_path}")
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None
    
    def load_environment(self, city_data_path=None):
        """加载环境数据（建筑物和边界）
        
        Args:
            city_data_path: 城市数据JSON文件路径
        """
        # 如果未指定路径，则使用当前场景的默认路径
        if city_data_path is None:
            city_data_path = os.path.join(project_root, 'worlds', f'{self.scenario}_city_data.json')
        
        # 检查文件是否存在
        if not os.path.exists(city_data_path):
            print(f"警告: 找不到城市数据文件 {city_data_path}")
            return False
        
        # 载入城市数据
        try:
            with open(city_data_path, 'r') as f:
                data = json.load(f)
                
            self.buildings = data['buildings']
            self.boundaries = data['boundaries']
            
            # 创建导航图
            min_x, min_z, max_x, max_z = self.boundaries
            self.graph = network_utils.build_navigation_graph(self.buildings, self.boundaries)
            
            print(f"成功加载环境数据 - 建筑物数量: {len(self.buildings)}, 边界: {self.boundaries}")
            return True
        except Exception as e:
            print(f"加载环境数据失败: {e}")
            return False
    
    def set_target(self, target_position):
        """设置目标位置
        
        Args:
            target_position: 目标位置 [x, y, z]
        """
        self.target_position = target_position
        self.goal_position = target_position
        print(f"设置目标位置: {target_position}")
    
    def get_current_position(self):
        """获取当前位置"""
        if self.gps:
            return self.gps.getValues()
        return [0, 0, 0]
    
    def generate_path(self):
        """使用GAT-DRL生成路径"""
        if not self.model or not self.graph or not self.target_position:
            print("错误: 无法生成路径，模型、环境图或目标未设置")
            return False
        
        # 获取当前位置
        current_position = self.get_current_position()
        if not self.start_position:
            self.start_position = current_position
            self.last_position = current_position
        
        # 找到最近的图节点
        start_node = network_utils.find_nearest_node(self.graph, current_position)
        goal_node = network_utils.find_nearest_node(self.graph, self.target_position)
        
        # 使用GAT-DRL模型生成路径
        try:
            # 创建环境编码器
            encoder = GATEnvironmentEncoder()
            
            # 编码环境
            obs = encoder.encode_environment(
                graph=self.graph,
                start_node=start_node,
                goal_node=goal_node,
                obs_size=(30, 30, 1)  # 假设观察空间大小是30x30x1
            )
            
            # 使用模型预测
            action, _states = self.model.predict(obs, deterministic=True)
            
            # 根据动作生成路径
            path = self._action_to_path(start_node, goal_node, action)
            
            # 如果路径生成成功，则更新当前路径
            if path:
                self.current_path = path
                self.current_waypoint_index = 0
                print(f"成功生成路径，共{len(path)}个路径点")
                return True
            else:
                print("路径生成失败")
                return False
        except Exception as e:
            print(f"生成路径时发生错误: {e}")
            return False
    
    def _action_to_path(self, start_node, goal_node, action):
        """将模型输出的动作转换为路径
        
        Args:
            start_node: 起始节点
            goal_node: 目标节点
            action: 模型预测的动作
        
        Returns:
            路径列表
        """
        # 简单处理：使用A*生成基本路径，然后根据action进行调整
        path = list(nx.astar_path(self.graph, start_node, goal_node))
        
        # 转换为坐标列表
        path_coords = []
        for node in path:
            x, y, z = self.graph.nodes[node]['position']
            path_coords.append([x, y, z])
        
        return path_coords
    
    def update_path_metrics(self):
        """更新路径指标（长度、能耗等）"""
        current_position = self.get_current_position()
        
        # 如果这是第一次调用，初始化上一个位置
        if self.last_position is None:
            self.last_position = current_position
            return
        
        # 计算路径增量
        dx = current_position[0] - self.last_position[0]
        dy = current_position[1] - self.last_position[1]
        dz = current_position[2] - self.last_position[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # 更新路径长度
        self.path_length += dist
        
        # 更新能耗（简单模型：每单位距离消耗固定能量）
        power = 120.0  # 无人机平均功率（瓦特）
        duration = self.timestep / 1000.0  # 控制周期（秒）
        energy = power * duration  # 能耗（焦耳）
        self.energy_consumption += energy
        
        # 更新上一个位置
        self.last_position = current_position
    
    def fly_to_waypoint(self):
        """飞向当前路径点"""
        if not self.current_path or self.current_waypoint_index >= len(self.current_path):
            # 达到路径终点
            self.hover()
            return True
        
        # 获取当前路径点
        waypoint = self.current_path[self.current_waypoint_index]
        
        # 获取当前位置
        current_position = self.get_current_position()
        
        # 计算方向向量
        dx = waypoint[0] - current_position[0]
        dy = waypoint[1] - current_position[1]
        dz = waypoint[2] - current_position[2]
        
        # 计算距离
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # 如果接近当前路径点，则移动到下一个路径点
        if distance < self.tolerance:
            self.current_waypoint_index += 1
            return self.current_waypoint_index >= len(self.current_path)
        
        # 计算方向单位向量
        distance = max(distance, 0.1)  # 避免除以零
        dx /= distance
        dy /= distance
        dz /= distance
        
        # 设置飞行参数
        base_speed = 5.0  # 基础速度
        
        # 根据距离调整速度（远距离快，近距离慢）
        speed = min(base_speed, distance)
        
        # 设置电机速度
        vertical_input = dy * speed + 68.5  # 悬停所需基本速度 + 垂直方向分量
        yaw_input = 0  # 简化模型，不处理偏航
        pitch_input = -dz * speed  # 前后移动
        roll_input = dx * speed  # 左右移动
        
        # 控制电机
        self.front_left_motor.setVelocity(vertical_input + yaw_input + pitch_input - roll_input)
        self.front_right_motor.setVelocity(vertical_input - yaw_input + pitch_input + roll_input)
        self.rear_left_motor.setVelocity(vertical_input - yaw_input - pitch_input - roll_input)
        self.rear_right_motor.setVelocity(vertical_input + yaw_input - pitch_input + roll_input)
        
        return False
    
    def hover(self):
        """悬停"""
        # 设置电机速度
        self.front_left_motor.setVelocity(68.5)
        self.front_right_motor.setVelocity(68.5)
        self.rear_left_motor.setVelocity(68.5)
        self.rear_right_motor.setVelocity(68.5)
    
    def land(self):
        """降落"""
        # 设置电机速度
        self.front_left_motor.setVelocity(0.0)
        self.front_right_motor.setVelocity(0.0)
        self.rear_left_motor.setVelocity(0.0)
        self.rear_right_motor.setVelocity(0.0)
    
    def is_at_target(self):
        """检查是否达到目标位置"""
        if not self.target_position:
            return False
        
        # 获取当前位置
        current_position = self.get_current_position()
        
        # 计算与目标的距离
        dx = current_position[0] - self.target_position[0]
        dy = current_position[1] - self.target_position[1]
        dz = current_position[2] - self.target_position[2]
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # 如果距离小于容差，则认为已达到目标
        return distance < self.tolerance
    
    def run(self):
        """主控制循环"""
        # 初始化状态
        state = "INIT"
        self.hover()
        
        # 记录开始时间
        self.start_time = time.time()
        
        # 主循环
        while self.robot.step(self.timestep) != -1:
            # 更新路径指标
            self.update_path_metrics()
            
            # 状态机
            if state == "INIT":
                # 等待GPS稳定
                if self.gps.getValues()[1] > 0.5:  # 检查高度
                    print("GPS已就绪，加载环境...")
                    if self.load_environment():
                        state = "READY"
                    else:
                        state = "ERROR"
            
            elif state == "READY":
                # 等待目标位置设置
                if self.target_position:
                    print("开始生成路径...")
                    if self.generate_path():
                        state = "FLYING"
                    else:
                        state = "ERROR"
            
            elif state == "FLYING":
                # 飞向目标
                target_reached = self.fly_to_waypoint()
                
                # 检查是否到达目标
                if target_reached or self.is_at_target():
                    print("到达目标！")
                    self.success = True
                    state = "LANDING"
            
            elif state == "LANDING":
                # 降落
                self.hover()
                time.sleep(1)  # 悬停一秒
                self.land()
                
                # 保存路径和指标
                self.save_metrics()
                self.save_path()
                
                # 结束仿真
                print("任务完成")
                break
            
            elif state == "ERROR":
                print("发生错误，停止仿真")
                break
            
            # 检查超时（3分钟）
            if time.time() - self.start_time > 180:
                print("任务超时")
                self.save_metrics()
                break
    
    def save_path(self):
        """保存飞行路径到CSV文件"""
        if not self.current_path:
            return
        
        # 创建结果目录
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_file = os.path.join(results_dir, f'{self.scenario}_GAT-DRL_path_{self.trial_id}.csv')
        
        # 将路径转换为DataFrame
        path_df = pd.DataFrame(self.current_path, columns=['x', 'y', 'z'])
        
        # 保存为CSV
        path_df.to_csv(path_file, index=False)
        print(f"保存路径数据: {path_file}")
    
    def save_metrics(self):
        """保存性能指标到CSV文件"""
        # 创建结果目录
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 计算耗时
        computation_time = time.time() - self.start_time
        
        # 准备数据
        metrics_data = {
            'algorithm': 'GAT-DRL',
            'scenario': self.scenario,
            'trial': self.trial_id,
            'path_length': self.path_length,
            'energy_consumption': self.energy_consumption,
            'computation_time': computation_time,
            'success': self.success
        }
        
        # 生成文件名
        metrics_file = os.path.join(results_dir, 'webots_simulation_results.csv')
        
        # 检查文件是否存在
        if os.path.exists(metrics_file):
            # 如果存在，追加数据
            metrics_df = pd.read_csv(metrics_file)
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_data])], ignore_index=True)
        else:
            # 如果不存在，创建新文件
            metrics_df = pd.DataFrame([metrics_data])
        
        # 保存为CSV
        metrics_df.to_csv(metrics_file, index=False)
        print(f"保存性能指标: {metrics_file}")
        
        # 同时保存单个测试的详细指标
        trial_metrics_file = os.path.join(results_dir, f'{self.scenario}_GAT-DRL_metrics_{self.trial_id}.csv')
        pd.DataFrame([metrics_data]).to_csv(trial_metrics_file, index=False)

def set_random_target(robot, boundaries):
    """设置随机目标位置
    
    Args:
        robot: 机器人控制器实例
        boundaries: 场景边界 [min_x, min_z, max_x, max_z]
    """
    # 获取边界
    min_x, min_z, max_x, max_z = boundaries
    
    # 设置随机目标位置（在边界内）
    margin = 10.0  # 边缘余量
    x = random.uniform(min_x + margin, max_x - margin)
    y = 1.0  # 固定高度
    z = random.uniform(min_z + margin, max_z - margin)
    
    # 设置目标
    robot.set_target([x, y, z])

def main():
    """主函数"""
    # 创建机器人实例
    robot = Robot()
    
    # 创建控制器
    controller = GATDRLController(robot)
    
    # 等待一些时间让GPS稳定
    for i in range(10):
        robot.step(int(robot.getBasicTimeStep()))
    
    # 加载环境
    controller.load_environment()
    
    # 设置随机目标（或从环境变量中获取）
    set_random_target(controller, controller.boundaries)
    
    # 运行控制器
    controller.run()

if __name__ == "__main__":
    main() 