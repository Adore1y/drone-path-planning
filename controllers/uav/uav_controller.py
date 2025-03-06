#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基本无人机控制器
提供基础运动控制功能，可作为各种算法的接口
"""

from controller import Robot
import sys
import os
import numpy as np
import math

# 添加父目录到Python路径，以便导入其他模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

class UAVController:
    """无人机控制器基类"""
    
    def __init__(self):
        """初始化控制器"""
        # 创建Robot实例
        self.robot = Robot()
        
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
        
        # 模拟高斯传感器噪声
        self.position_noise_std = 0.1  # 位置噪声标准差 (m)
        self.orientation_noise_std = 0.01  # 方向噪声标准差 (rad)
        
    def read_sensors(self):
        """读取传感器数据并更新当前状态"""
        # 读取GPS
        if self.gps.getSamplingPeriod() > 0:
            gps_values = self.gps.getValues()
            
            # 添加噪声
            noise = np.random.normal(0, self.position_noise_std, 3)
            self.position = [
                gps_values[0] + noise[0],
                gps_values[1] + noise[1],
                gps_values[2] + noise[2]
            ]
        
        # 读取IMU
        if self.imu.getSamplingPeriod() > 0:
            imu_values = self.imu.getRollPitchYaw()
            
            # 添加噪声
            noise = np.random.normal(0, self.orientation_noise_std, 3)
            self.orientation = [
                imu_values[0] + noise[0],
                imu_values[1] + noise[1],
                imu_values[2] + noise[2]
            ]
    
    def set_target_position(self, target_position):
        """
        设置目标位置
        
        参数:
            target_position: 目标位置 [x, y, z]
        """
        self.target_position = target_position
    
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
    
    def move_to_target(self):
        """移动到目标位置"""
        if self.target_position is None:
            return
        
        # 计算到目标的向量
        dx = self.target_position[0] - self.position[0]
        dy = self.target_position[1] - self.position[1]
        dz = self.target_position[2] - self.position[2]
        
        # 计算距离
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # 如果足够接近目标，停止
        if distance < 0.1:
            self.set_velocity([0, 0, 0])
            return True
        
        # 计算期望速度（按比例靠近目标）
        speed = min(self.max_velocity, 0.5 * distance)
        
        # 水平方向单位向量
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        if horizontal_dist > 0:
            vx = dx / horizontal_dist * speed
            vy = dy / horizontal_dist * speed
        else:
            vx = 0
            vy = 0
        
        # 垂直方向速度
        vz_magnitude = min(abs(dz), self.max_ascent_rate if dz > 0 else self.max_descent_rate)
        vz = vz_magnitude * (1 if dz > 0 else -1)
        
        # 设置速度
        self.set_velocity([vx, vy, vz])
        
        return False
    
    def apply_velocity(self):
        """应用目标速度"""
        # 将速度应用到机器人
        # 在Webots中，我们通过直接修改位置来模拟速度
        
        vx, vy, vz = self.target_velocity
        
        # 计算新位置（基于时间步长）
        dt = self.time_step / 1000.0  # 转换为秒
        
        new_x = self.position[0] + vx * dt
        new_y = self.position[1] + vy * dt
        new_z = self.position[2] + vz * dt
        
        # 使用Supervisor API更新位置
        # 注意：这里我们使用的是简化模型，实际的四旋翼动力学会更复杂
        
        # 在实际的Webots环境中，无法直接设置位置
        # 这里只是模拟，实际控制需要通过马达或推进器
        
        # 更新当前速度
        self.velocity = self.target_velocity.copy()
        
        return [new_x, new_y, new_z]
    
    def run_step(self):
        """
        执行一个控制步骤
        
        返回:
            done: 是否到达目标
        """
        # 读取传感器
        self.read_sensors()
        
        # 移动到目标
        if self.target_position:
            reached = self.move_to_target()
            if reached:
                return True
        
        # 应用速度
        new_position = self.apply_velocity()
        
        # 这里真正的控制将在Webots物理引擎中进行
        # 我们在这里只是计算期望的运动
        
        return False
            
    def run(self):
        """控制器主循环"""
        # 主循环
        while self.robot.step(self.time_step) != -1:
            done = self.run_step()
            
            # 如果到达目标，可以在这里做一些处理
            if done and self.target_position:
                print(f"Reached target position: {self.target_position}")
                # 清除目标，等待新指令
                self.target_position = None

# 简单测试
def main():
    controller = UAVController()
    
    # 设置目标位置
    controller.set_target_position([10, 5, 10])
    
    controller.run()

if __name__ == "__main__":
    main() 