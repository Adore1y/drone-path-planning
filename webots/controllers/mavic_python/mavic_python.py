#!/usr/bin/env python3

"""
Mavic Pro 无人机控制器示例
这个控制器演示了如何控制 Mavic 无人机进行基本飞行
"""

from controller import Robot, Motor, Gyro, GPS, Camera, Compass, InertialUnit
import os
import csv
import time
import math

# 创建机器人实例
robot = Robot()

# 获取仿真的时间步长
timestep = int(robot.getBasicTimeStep())

# 初始化设备
front_left_motor = robot.getDevice("front left propeller motor")
front_right_motor = robot.getDevice("front right propeller motor")
rear_left_motor = robot.getDevice("rear left propeller motor")
rear_right_motor = robot.getDevice("rear right propeller motor")

# 设置电机为速度控制模式
front_left_motor.setPosition(float('inf'))
front_right_motor.setPosition(float('inf'))
rear_left_motor.setPosition(float('inf'))
rear_right_motor.setPosition(float('inf'))

# 设置电机初始速度为0
front_left_motor.setVelocity(0.0)
front_right_motor.setVelocity(0.0)
rear_left_motor.setVelocity(0.0)
rear_right_motor.setVelocity(0.0)

# 获取并启用传感器
gps = robot.getDevice("gps")
gps.enable(timestep)

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

compass = robot.getDevice("compass")
compass.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

camera = robot.getDevice("camera")
camera.enable(timestep)

# 创建数据记录目录和文件
data_dir = "flight_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

timestamp = time.strftime("%Y%m%d_%H%M%S")
data_file = os.path.join(data_dir, f"flight_data_{timestamp}.csv")

# 打开CSV文件并写入表头
with open(data_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "时间(秒)", "状态", 
        "X位置(米)", "Y位置(米)", "Z位置(米)", 
        "横滚角(弧度)", "俯仰角(弧度)", "偏航角(弧度)",
        "线速度X(米/秒)", "线速度Y(米/秒)", "线速度Z(米/秒)",
        "角速度X(弧度/秒)", "角速度Y(弧度/秒)", "角速度Z(弧度/秒)",
        "前左电机速度", "前右电机速度", "后左电机速度", "后右电机速度"
    ])

# 计算速度所需的变量
prev_position = [0, 0, 0]
prev_time = 0
velocity = [0, 0, 0]

# 无人机状态
state = "TAKEOFF"  # 初始状态为起飞
target_altitude = 1.0  # 目标高度（米）
hover_time = 0  # 悬停计时器

# 主控制循环
while robot.step(timestep) != -1:
    # 获取仿真时间
    current_time = robot.getTime()
    
    # 获取传感器数据
    gps_values = gps.getValues()
    altitude = gps_values[2]
    
    roll = imu.getRollPitchYaw()[0]
    pitch = imu.getRollPitchYaw()[1]
    yaw = imu.getRollPitchYaw()[2]
    
    gyro_values = gyro.getValues()
    
    # 计算速度（简单的一阶差分）
    if prev_time > 0:
        dt = current_time - prev_time
        if dt > 0:
            velocity[0] = (gps_values[0] - prev_position[0]) / dt
            velocity[1] = (gps_values[1] - prev_position[1]) / dt
            velocity[2] = (gps_values[2] - prev_position[2]) / dt
    
    prev_position = gps_values.copy()
    prev_time = current_time
    
    # 简单的状态机控制
    motor_speed_fl = 0
    motor_speed_fr = 0
    motor_speed_rl = 0
    motor_speed_rr = 0
    
    if state == "TAKEOFF":
        # 起飞阶段 - 垂直上升到目标高度
        if altitude < target_altitude:
            # 设置所有螺旋桨以相同速度旋转以垂直上升
            motor_speed = 10.0
            motor_speed_fl = motor_speed
            motor_speed_fr = motor_speed
            motor_speed_rl = motor_speed
            motor_speed_rr = motor_speed
            
            front_left_motor.setVelocity(motor_speed_fl)
            front_right_motor.setVelocity(motor_speed_fr)
            rear_left_motor.setVelocity(motor_speed_rl)
            rear_right_motor.setVelocity(motor_speed_rr)
        else:
            # 达到目标高度，切换到悬停状态
            state = "HOVER"
            print(f"已达到目标高度: {altitude:.2f}米，开始悬停")
    
    elif state == "HOVER":
        # 悬停阶段 - 保持当前位置和高度
        hover_time += timestep / 1000.0  # 转换为秒
        
        # 简单的悬停控制（实际上需要更复杂的PID控制）
        motor_speed = 7.5  # 悬停所需的大致速度
        motor_speed_fl = motor_speed
        motor_speed_fr = motor_speed
        motor_speed_rl = motor_speed
        motor_speed_rr = motor_speed
        
        front_left_motor.setVelocity(motor_speed_fl)
        front_right_motor.setVelocity(motor_speed_fr)
        rear_left_motor.setVelocity(motor_speed_rl)
        rear_right_motor.setVelocity(motor_speed_rr)
        
        # 悬停10秒后切换到前进状态
        if hover_time > 10.0:
            state = "FORWARD"
            print("悬停完成，开始前进")
    
    elif state == "FORWARD":
        # 前进阶段 - 向前飞行
        # 通过调整前后螺旋桨的速度差来产生前进的俯仰
        motor_speed_fl = 6.5   # 前方螺旋桨速度稍低
        motor_speed_fr = 6.5
        motor_speed_rl = 8.5    # 后方螺旋桨速度稍高
        motor_speed_rr = 8.5
        
        front_left_motor.setVelocity(motor_speed_fl)
        front_right_motor.setVelocity(motor_speed_fr)
        rear_left_motor.setVelocity(motor_speed_rl)
        rear_right_motor.setVelocity(motor_speed_rr)
    
    # 记录数据到CSV文件
    with open(data_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            current_time, state,
            gps_values[0], gps_values[1], gps_values[2],
            roll, pitch, yaw,
            velocity[0], velocity[1], velocity[2],
            gyro_values[0], gyro_values[1], gyro_values[2],
            motor_speed_fl, motor_speed_fr, motor_speed_rl, motor_speed_rr
        ])
    
    # 打印当前状态信息 (每秒打印一次)
    if current_time % 1 < 0.1:
        print(f"时间: {current_time:.2f}秒, 状态: {state}, 高度: {altitude:.2f}米")
        print(f"位置: X={gps_values[0]:.2f}, Y={gps_values[1]:.2f}, Z={gps_values[2]:.2f}")
        print(f"姿态(rad): 横滚={roll:.2f}, 俯仰={pitch:.2f}, 偏航={yaw:.2f}")
        print(f"速度(m/s): X={velocity[0]:.2f}, Y={velocity[1]:.2f}, Z={velocity[2]:.2f}")
        print("-" * 40)

# 仿真结束时的处理
print(f"仿真结束，数据已保存到 {data_file}") 