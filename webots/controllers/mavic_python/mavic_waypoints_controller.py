#!/usr/bin/env python3

"""
Mavic Pro 无人机路径点导航控制器
实现基于预设路径点的飞行路径规划和简单避障
"""

from controller import Robot, Motor, Gyro, GPS, Camera, Compass, InertialUnit, DistanceSensor
import os
import csv
import time
import math
import numpy as np

# 状态定义
STATE_TAKEOFF = "TAKEOFF"
STATE_NAVIGATE = "NAVIGATE"
STATE_HOVER = "HOVER"
STATE_LAND = "LAND"
STATE_EMERGENCY = "EMERGENCY"
STATE_FINISHED = "FINISHED"

# PID控制器类
class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.output_limits = output_limits
        
    def compute(self, error, dt):
        # 比例项
        p_term = self.kp * error
        
        # 积分项 - 有积分饱和限制
        self.integral += error * dt
        if self.output_limits:
            self.integral = max(min(self.integral, self.output_limits[1] / self.ki), 
                              self.output_limits[0] / self.ki)
        i_term = self.ki * self.integral
        
        # 微分项
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        # 计算输出
        output = p_term + i_term + d_term
        
        # 应用输出限制
        if self.output_limits:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])
            
        return output
    
    def reset(self):
        self.prev_error = 0
        self.integral = 0

# 创建数据记录目录
def setup_data_recording():
    data_dir = "flight_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    data_file = os.path.join(data_dir, f"waypoints_flight_{timestamp}.csv")
    
    with open(data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "时间(秒)", "状态", 
            "X位置(米)", "Y位置(米)", "Z位置(米)", 
            "目标X(米)", "目标Y(米)", "目标Z(米)",
            "距离目标(米)", "当前航点",
            "横滚角(弧度)", "俯仰角(弧度)", "偏航角(弧度)",
            "线速度X(米/秒)", "线速度Y(米/秒)", "线速度Z(米/秒)",
            "前左电机速度", "前右电机速度", "后左电机速度", "后右电机速度",
            "避障状态"
        ])
    
    return data_file

# 从世界文件中提取路径点
def extract_waypoints_from_world(world_file_path):
    waypoints = []
    in_waypoints_section = False
    
    try:
        with open(world_file_path, 'r') as f:
            for line in f:
                if '# BEGIN_WAYPOINTS' in line:
                    in_waypoints_section = True
                    continue
                if '# END_WAYPOINTS' in line:
                    in_waypoints_section = False
                    continue
                if in_waypoints_section and line.startswith('#'):
                    # 提取坐标
                    parts = line[1:].strip().split()
                    if len(parts) == 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            waypoints.append((x, y, z))
                        except ValueError:
                            pass
    except Exception as e:
        print(f"读取路径点时出错: {e}")
        # 如果无法读取，使用默认路径点
        waypoints = [
            (0, 0, 1.5),   # 起点
            (5, 0, 2.0),   # 点1
            (5, 5, 2.5),   # 点2
            (0, 5, 2.0),   # 点3
            (0, 0, 1.5)    # 回到起点
        ]
    
    return waypoints

# 计算两点之间的欧几里得距离
def distance_3d(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)

# 计算平面角度（偏航角）
def calculate_yaw(current_x, current_y, target_x, target_y):
    angle = math.atan2(target_y - current_y, target_x - current_x)
    return angle

# 主控制器类
class MavicController:
    def __init__(self, world_file_path):
        # 初始化机器人实例
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0  # 时间步长（秒）
        
        # 初始化设备
        self.setup_devices()
        
        # PID控制器
        self.pid_x = PIDController(1.0, 0.1, 0.5, (-1, 1))
        self.pid_y = PIDController(1.0, 0.1, 0.5, (-1, 1))
        self.pid_z = PIDController(2.0, 0.1, 0.5, (-2, 2))
        self.pid_yaw = PIDController(2.0, 0.0, 0.5, (-1, 1))
        
        # 状态变量
        self.state = STATE_TAKEOFF
        self.target_altitude = 1.5
        self.waypoints = extract_waypoints_from_world(world_file_path)
        self.current_waypoint_index = 0
        self.waypoint_reached_distance = 0.5  # 米
        self.hovering_time = 0
        self.hovering_duration = 2.0  # 每个路径点悬停2秒
        
        # 位置和速度
        self.position = [0, 0, 0]
        self.prev_position = [0, 0, 0]
        self.velocity = [0, 0, 0]
        
        # 避障状态
        self.obstacle_avoidance_active = False
        self.avoidance_vector = [0, 0, 0]
        
        # 数据记录
        self.data_file = setup_data_recording()
        
        print("Mavic路径点导航控制器已初始化")
        print(f"共{len(self.waypoints)}个路径点")
        for i, (x, y, z) in enumerate(self.waypoints):
            print(f"路径点 {i}: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    def setup_devices(self):
        # 初始化电机
        self.front_left_motor = self.robot.getDevice("front left propeller motor")
        self.front_right_motor = self.robot.getDevice("front right propeller motor")
        self.rear_left_motor = self.robot.getDevice("rear left propeller motor")
        self.rear_right_motor = self.robot.getDevice("rear right propeller motor")
        
        # 设置电机为速度控制模式
        self.front_left_motor.setPosition(float('inf'))
        self.front_right_motor.setPosition(float('inf'))
        self.rear_left_motor.setPosition(float('inf'))
        self.rear_right_motor.setPosition(float('inf'))
        
        # 初始速度为0
        self.front_left_motor.setVelocity(0.0)
        self.front_right_motor.setVelocity(0.0)
        self.rear_left_motor.setVelocity(0.0)
        self.rear_right_motor.setVelocity(0.0)
        
        # 启用传感器
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)
        
        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(self.timestep)
        
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)
        
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.timestep)
        
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        
        # 启用距离传感器（如果存在）
        try:
            self.front_ds = self.robot.getDevice("front distance sensor")
            self.front_ds.enable(self.timestep)
            
            self.left_ds = self.robot.getDevice("left distance sensor")
            self.left_ds.enable(self.timestep)
            
            self.right_ds = self.robot.getDevice("right distance sensor")
            self.right_ds.enable(self.timestep)
            
            self.down_ds = self.robot.getDevice("down distance sensor")
            self.down_ds.enable(self.timestep)
            
            self.has_distance_sensors = True
        except:
            print("无距离传感器，将使用简化的避障功能")
            self.has_distance_sensors = False
    
    def get_sensor_data(self):
        # 获取位置
        self.prev_position = self.position.copy()
        self.position = self.gps.getValues()
        
        # 计算速度
        self.velocity = [
            (self.position[0] - self.prev_position[0]) / self.dt,
            (self.position[1] - self.prev_position[1]) / self.dt,
            (self.position[2] - self.prev_position[2]) / self.dt
        ]
        
        # 获取姿态
        self.roll = self.imu.getRollPitchYaw()[0]
        self.pitch = self.imu.getRollPitchYaw()[1]
        self.yaw = self.imu.getRollPitchYaw()[2]
        
        # 获取陀螺仪数据
        self.angular_velocity = self.gyro.getValues()
        
        # 获取当前目标路径点
        if self.current_waypoint_index < len(self.waypoints):
            self.target_position = self.waypoints[self.current_waypoint_index]
        else:
            self.target_position = self.waypoints[0]  # 循环回起点
        
        # 计算到目标的距离
        self.distance_to_target = distance_3d(self.position, self.target_position)
        
        # 检查障碍物
        self.check_obstacles()
    
    def check_obstacles(self):
        # 简单的模拟避障 - 在真实场景中应使用距离传感器
        # 这里我们模拟一个简化的避障行为
        
        # 重置避障状态
        self.obstacle_avoidance_active = False
        self.avoidance_vector = [0, 0, 0]
        
        if self.has_distance_sensors:
            # 使用距离传感器进行避障
            front_dist = self.front_ds.getValue()
            left_dist = self.left_ds.getValue()
            right_dist = self.right_ds.getValue()
            down_dist = self.down_ds.getValue()
            
            obstacle_threshold = 2.0  # 检测障碍物的阈值距离
            
            if front_dist < obstacle_threshold:
                self.obstacle_avoidance_active = True
                self.avoidance_vector[0] = -1.0  # 向后避障
            
            if left_dist < obstacle_threshold:
                self.obstacle_avoidance_active = True
                self.avoidance_vector[1] = 0.5  # 向右避障
            
            if right_dist < obstacle_threshold:
                self.obstacle_avoidance_active = True
                self.avoidance_vector[1] = -0.5  # 向左避障
            
            if down_dist < 1.0:
                self.obstacle_avoidance_active = True
                self.avoidance_vector[2] = 0.5  # 向上避障
    
    def compute_control(self):
        # 根据不同状态计算控制输出
        if self.state == STATE_TAKEOFF:
            # 起飞阶段 - 简单的垂直上升到目标高度
            error_z = self.target_altitude - self.position[2]
            
            if abs(error_z) < 0.1:
                print("起飞完成，开始路径点导航")
                self.state = STATE_NAVIGATE
                # 重置PID控制器
                self.pid_x.reset()
                self.pid_y.reset()
                self.pid_z.reset()
                self.pid_yaw.reset()
                return 0, 0, error_z * 2.0, 0
            
            return 0, 0, error_z * 2.0, 0
            
        elif self.state == STATE_NAVIGATE:
            # 检查是否达到当前路径点
            if self.distance_to_target < self.waypoint_reached_distance:
                print(f"到达路径点 {self.current_waypoint_index}")
                self.state = STATE_HOVER
                self.hovering_time = 0
                return 0, 0, 0, 0
            
            # 计算到目标的方向
            target_x, target_y, target_z = self.target_position
            
            # 计算目标偏航角 - 朝向目标点
            target_yaw = calculate_yaw(self.position[0], self.position[1], target_x, target_y)
            error_yaw = target_yaw - self.yaw
            # 确保角度在 -pi 到 pi 之间
            while error_yaw > math.pi:
                error_yaw -= 2 * math.pi
            while error_yaw < -math.pi:
                error_yaw += 2 * math.pi
                
            # 计算位置误差
            error_x = target_x - self.position[0]
            error_y = target_y - self.position[1]
            error_z = target_z - self.position[2]
            
            # 应用避障向量
            if self.obstacle_avoidance_active:
                error_x += self.avoidance_vector[0] * 2.0
                error_y += self.avoidance_vector[1] * 2.0
                error_z += self.avoidance_vector[2] * 2.0
            
            # 使用PID控制器计算控制输出
            control_x = self.pid_x.compute(error_x, self.dt)
            control_y = self.pid_y.compute(error_y, self.dt)
            control_z = self.pid_z.compute(error_z, self.dt)
            control_yaw = self.pid_yaw.compute(error_yaw, self.dt)
            
            return control_x, control_y, control_z, control_yaw
            
        elif self.state == STATE_HOVER:
            # 悬停一段时间
            self.hovering_time += self.dt
            
            if self.hovering_time >= self.hovering_duration:
                # 悬停结束，前往下一个路径点或结束
                self.current_waypoint_index += 1
                
                if self.current_waypoint_index >= len(self.waypoints):
                    print("所有路径点已访问，准备降落")
                    self.state = STATE_LAND
                else:
                    print(f"前往路径点 {self.current_waypoint_index}")
                    self.state = STATE_NAVIGATE
            
            # 保持当前高度
            error_z = self.target_position[2] - self.position[2]
            return 0, 0, error_z * 1.5, 0
            
        elif self.state == STATE_LAND:
            # 降落过程 - 缓慢下降
            if self.position[2] < 0.3:
                print("已着陆，任务完成")
                self.state = STATE_FINISHED
                return 0, 0, 0, 0
            
            # 设置降落速度
            return 0, 0, -0.5, 0
            
        elif self.state == STATE_EMERGENCY:
            # 紧急模式 - 立即降落
            return 0, 0, -1.0, 0
            
        elif self.state == STATE_FINISHED:
            # 任务完成，关闭电机
            return 0, 0, 0, 0
        
        # 默认返回值
        return 0, 0, 0, 0
    
    def apply_motor_controls(self, x_control, y_control, z_control, yaw_control):
        # 将控制输入转换为电机输出
        # 这是一个简化的四旋翼控制模型
        
        if self.state == STATE_FINISHED:
            # 如果已完成，关闭所有电机
            self.front_left_motor.setVelocity(0)
            self.front_right_motor.setVelocity(0)
            self.rear_left_motor.setVelocity(0)
            self.rear_right_motor.setVelocity(0)
            return
        
        # 基础油门 - 提供基本升力
        base_throttle = 68.5
        
        # 根据控制输入调整每个电机的速度
        front_left = base_throttle + z_control - y_control - x_control - yaw_control
        front_right = base_throttle + z_control - y_control + x_control + yaw_control
        rear_left = base_throttle + z_control + y_control - x_control + yaw_control
        rear_right = base_throttle + z_control + y_control + x_control - yaw_control
        
        # 限制速度范围
        max_speed = 150.0
        front_left = max(0, min(front_left, max_speed))
        front_right = max(0, min(front_right, max_speed))
        rear_left = max(0, min(rear_left, max_speed))
        rear_right = max(0, min(rear_right, max_speed))
        
        # 设置电机速度
        self.front_left_motor.setVelocity(front_left)
        self.front_right_motor.setVelocity(front_right)
        self.rear_left_motor.setVelocity(rear_left)
        self.rear_right_motor.setVelocity(rear_right)
        
        # 保存电机速度供数据记录使用
        self.motor_speeds = [front_left, front_right, rear_left, rear_right]
    
    def record_data(self):
        # 记录飞行数据
        with open(self.data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.robot.getTime(), self.state,
                self.position[0], self.position[1], self.position[2],
                self.target_position[0], self.target_position[1], self.target_position[2],
                self.distance_to_target, self.current_waypoint_index,
                self.roll, self.pitch, self.yaw,
                self.velocity[0], self.velocity[1], self.velocity[2],
                self.motor_speeds[0], self.motor_speeds[1], self.motor_speeds[2], self.motor_speeds[3],
                "避障中" if self.obstacle_avoidance_active else "正常"
            ])
    
    def run(self):
        # 主控制循环
        while self.robot.step(self.timestep) != -1:
            # 获取传感器数据
            self.get_sensor_data()
            
            # 计算控制输出
            x_control, y_control, z_control, yaw_control = self.compute_control()
            
            # 应用到电机
            self.apply_motor_controls(x_control, y_control, z_control, yaw_control)
            
            # 记录数据
            self.record_data()
            
            # 打印状态信息
            if int(self.robot.getTime() * 10) % 10 == 0:  # 每1秒打印一次
                print(f"时间: {self.robot.getTime():.1f}秒, 状态: {self.state}")
                print(f"位置: ({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})")
                print(f"目标: ({self.target_position[0]:.2f}, {self.target_position[1]:.2f}, {self.target_position[2]:.2f})")
                print(f"距离目标: {self.distance_to_target:.2f}米, 路径点: {self.current_waypoint_index}/{len(self.waypoints)}")
                if self.obstacle_avoidance_active:
                    print("警告: 避障中")
                print("-" * 30)
            
            # 检查是否完成
            if self.state == STATE_FINISHED:
                print("任务完成，数据已保存")
                break


# 主程序
if __name__ == "__main__":
    # 世界文件路径
    world_file_path = "../../worlds/mixed_scenario.wbt"
    
    # 创建并运行控制器
    controller = MavicController(world_file_path)
    controller.run() 