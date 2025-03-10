#!/usr/bin/env python3
"""
无人机能量消耗模型
用于估算无人机在不同操作状态下的能量消耗
"""

import numpy as np
import math

class DroneEnergyModel:
    """
    无人机能量消耗模型类
    基于无人机物理特性计算能量消耗

    参考文献:
    1. Di Franco, C., & Buttazzo, G. (2015). Energy-aware UAV path planning for surveillance missions
    2. Zeng, Y., & Zhang, R. (2017). Energy-efficient UAV communication with trajectory optimization
    """
    
    def __init__(self, config=None):
        # 无人机物理参数（默认值基于通用四旋翼）
        self.mass = 1.0  # kg
        self.gravity = 9.81  # m/s^2
        self.rotor_count = 4  # 四旋翼
        self.rotor_radius = 0.127  # m
        self.air_density = 1.225  # kg/m^3
        self.drag_coefficient = 0.3  # 无单位
        self.power_consumption_hover = 100.0  # W (悬停功率)
        self.frontal_area = 0.01  # m^2
        self.power_efficiency = 0.7  # 电机效率
        self.battery_capacity = 5200.0  # mAh
        self.battery_voltage = 11.1  # V (3S LiPo)
        
        # 额外参数计算
        self.max_thrust = self.mass * self.gravity * 2.0  # 最大推力为体重的两倍
        
        # 加载自定义配置
        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def calculate_hover_power(self):
        """计算悬停功率 (W)"""
        # 基于气动学理论的悬停功率
        # P = (T^1.5) / sqrt(2 * rho * A)
        # 其中 T 是推力, rho 是空气密度, A 是旋翼盘面积
        
        thrust_per_rotor = (self.mass * self.gravity) / self.rotor_count
        disc_area = math.pi * (self.rotor_radius ** 2)
        
        hover_power_per_rotor = (thrust_per_rotor ** 1.5) / math.sqrt(2 * self.air_density * disc_area)
        total_hover_power = hover_power_per_rotor * self.rotor_count / self.power_efficiency
        
        return total_hover_power
    
    def calculate_power_consumption(self, velocity, acceleration, angular_velocity, thrust_ratio):
        """
        计算给定状态下的功率消耗 (W)
        
        参数:
            velocity (np.array): 三轴线速度 [vx, vy, vz] (m/s)
            acceleration (np.array): 三轴加速度 [ax, ay, az] (m/s^2)
            angular_velocity (np.array): 三轴角速度 [wx, wy, wz] (rad/s)
            thrust_ratio (float): 推力/最大推力比例 [0.0 - 1.0]
        
        返回:
            float: 功率消耗 (W)
        """
        # 1. 基础悬停功率
        base_power = self.calculate_hover_power()
        
        # 2. 计算平移功率 (功率随速度的三次方增加)
        velocity_magnitude = np.linalg.norm(velocity)
        drag_force = 0.5 * self.air_density * (velocity_magnitude ** 2) * self.frontal_area * self.drag_coefficient
        translation_power = drag_force * velocity_magnitude / self.power_efficiency
        
        # 3. 计算加速功率
        accel_magnitude = np.linalg.norm(acceleration)
        acceleration_power = self.mass * accel_magnitude * velocity_magnitude / self.power_efficiency
        
        # 4. 计算旋转功率 (角速度引起的额外功率消耗)
        angular_magnitude = np.linalg.norm(angular_velocity)
        rotational_power = 0.1 * base_power * (angular_magnitude / math.pi)  # 简化模型
        
        # 5. 计算推力相关功率 (非线性关系)
        # 推力功率与推力的1.5次方成正比
        thrust_power_factor = (thrust_ratio ** 1.5)
        thrust_power = base_power * thrust_power_factor
        
        # 总功率为各组分之和
        total_power = thrust_power + translation_power + acceleration_power + rotational_power
        
        return total_power
    
    def calculate_energy_consumption(self, state, action, dt=0.1):
        """
        计算在给定状态和动作下的能量消耗 (Joules)
        
        参数:
            state (dict): 无人机状态 (包含位置,速度,角度等)
            action (np.array): 控制输入 [roll, pitch, yaw_rate, thrust]
            dt (float): 时间步长 (秒)
        
        返回:
            float: 能量消耗 (J)
        """
        # 从状态中提取信息
        velocity = np.array(state.get('linear_velocity', [0.0, 0.0, 0.0]))
        angular_velocity = np.array(state.get('angular_velocity', [0.0, 0.0, 0.0]))
        
        # 计算加速度 (简化模型，根据roll/pitch估算)
        roll = action[0]
        pitch = action[1]
        thrust = action[3]
        
        # 估算加速度
        # 假设roll/pitch角度导致重力分量产生水平加速度
        g = self.gravity
        ax = g * math.sin(pitch)
        ay = g * math.sin(roll)
        az = thrust * self.max_thrust / self.mass - g
        acceleration = np.array([ax, ay, az])
        
        # 计算功率
        power = self.calculate_power_consumption(
            velocity=velocity,
            acceleration=acceleration,
            angular_velocity=angular_velocity,
            thrust_ratio=thrust
        )
        
        # 能量 = 功率 × 时间
        energy = power * dt
        
        return energy
    
    def calculate_flight_time(self, average_power):
        """
        基于平均功率计算预计飞行时间（分钟）
        
        参数:
            average_power: 平均功率消耗 (W)
        
        返回:
            预计飞行时间 (min)
        """
        # 电池能量 (Wh) = 容量 (mAh) * 电压 (V) / 1000
        battery_energy_wh = self.battery_capacity * self.battery_voltage / 1000.0
        
        # 电池能量 (J) = 能量 (Wh) * 3600
        battery_energy_j = battery_energy_wh * 3600.0
        
        # 飞行时间 (s) = 能量 (J) / 功率 (W)
        flight_time_s = battery_energy_j / average_power
        
        # 转换为分钟
        flight_time_min = flight_time_s / 60.0
        
        return flight_time_min
    
    def efficiency_score(self, total_energy, distance_traveled):
        """
        计算能源效率评分 (距离/能量)
        
        参数:
            total_energy: 总能量消耗 (J)
            distance_traveled: 飞行距离 (m)
        
        返回:
            效率分数 (m/kJ)
        """
        if total_energy <= 0:
            return 0
        
        # 转换为 m/kJ
        score = distance_traveled / (total_energy / 1000.0)
        return score

# 测试代码
if __name__ == "__main__":
    # 创建能量模型
    energy_model = DroneEnergyModel()
    
    # 测试悬停功率
    hover_power = energy_model.calculate_hover_power()
    print(f"Hover power: {hover_power:.2f} W")
    
    # 测试飞行状态下的功率
    state = {
        'linear_velocity': [5.0, 0.0, 0.0],
        'angular_velocity': [0.0, 0.0, 0.2]
    }
    action = [0.1, 0.2, 0.0, 0.6]  # roll, pitch, yaw_rate, thrust
    
    energy = energy_model.calculate_energy_consumption(state, action)
    print(f"Energy consumption for 0.1s: {energy:.2f} J")
    
    # 估算飞行时间
    flight_time = energy_model.calculate_flight_time(hover_power)
    print(f"Estimated flight time at hover: {flight_time:.2f} minutes") 