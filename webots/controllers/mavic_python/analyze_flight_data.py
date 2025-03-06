#!/usr/bin/env python3

"""
无人机飞行数据分析与可视化脚本
该脚本用于分析由mavic_python.py控制器生成的飞行数据
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 查找最新的数据文件
data_dir = "flight_data"
list_of_files = glob.glob(os.path.join(data_dir, "*.csv"))
if not list_of_files:
    print(f"错误: 在'{data_dir}'目录中没有找到CSV数据文件!")
    exit(1)

latest_file = max(list_of_files, key=os.path.getctime)
print(f"分析最新的数据文件: {latest_file}")

# 读取CSV数据
try:
    df = pd.read_csv(latest_file)
    print(f"成功加载数据, 共{len(df)}行记录")
except Exception as e:
    print(f"读取数据文件时出错: {e}")
    exit(1)

# 创建输出目录
output_dir = os.path.join(data_dir, "分析结果")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 基本数据统计
print("\n基本数据统计:")
numeric_columns = df.select_dtypes(include=[np.number]).columns
stats = df[numeric_columns].describe()
print(stats)

# 保存统计结果
stats.to_csv(os.path.join(output_dir, "统计结果.csv"))
print(f"统计结果已保存到 {os.path.join(output_dir, '统计结果.csv')}")

# 1. 绘制高度随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(df['时间(秒)'], df['Z位置(米)'])
plt.title('无人机高度随时间的变化')
plt.xlabel('时间 (秒)')
plt.ylabel('高度 (米)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, '高度变化.png'))

# 2. 绘制三轴位置随时间变化
plt.figure(figsize=(12, 8))
plt.plot(df['时间(秒)'], df['X位置(米)'], 'r-', label='X位置')
plt.plot(df['时间(秒)'], df['Y位置(米)'], 'g-', label='Y位置')
plt.plot(df['时间(秒)'], df['Z位置(米)'], 'b-', label='Z位置')
plt.title('无人机位置随时间的变化')
plt.xlabel('时间 (秒)')
plt.ylabel('位置 (米)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, '位置变化.png'))

# 3. 绘制三轴速度随时间变化
plt.figure(figsize=(12, 8))
plt.plot(df['时间(秒)'], df['线速度X(米/秒)'], 'r-', label='X速度')
plt.plot(df['时间(秒)'], df['线速度Y(米/秒)'], 'g-', label='Y速度')
plt.plot(df['时间(秒)'], df['线速度Z(米/秒)'], 'b-', label='Z速度')
plt.title('无人机速度随时间的变化')
plt.xlabel('时间 (秒)')
plt.ylabel('速度 (米/秒)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, '速度变化.png'))

# 4. 绘制姿态角随时间变化
plt.figure(figsize=(12, 8))
plt.plot(df['时间(秒)'], df['横滚角(弧度)'], 'r-', label='横滚角')
plt.plot(df['时间(秒)'], df['俯仰角(弧度)'], 'g-', label='俯仰角')
plt.plot(df['时间(秒)'], df['偏航角(弧度)'], 'b-', label='偏航角')
plt.title('无人机姿态角随时间的变化')
plt.xlabel('时间 (秒)')
plt.ylabel('姿态角 (弧度)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, '姿态角变化.png'))

# 5. 绘制电机速度随时间变化
plt.figure(figsize=(12, 8))
plt.plot(df['时间(秒)'], df['前左电机速度'], 'r-', label='前左电机')
plt.plot(df['时间(秒)'], df['前右电机速度'], 'g-', label='前右电机')
plt.plot(df['时间(秒)'], df['后左电机速度'], 'b-', label='后左电机')
plt.plot(df['时间(秒)'], df['后右电机速度'], 'y-', label='后右电机')
plt.title('无人机电机速度随时间的变化')
plt.xlabel('时间 (秒)')
plt.ylabel('电机速度')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, '电机速度变化.png'))

# 6. 绘制3D飞行轨迹
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['X位置(米)'], df['Y位置(米)'], df['Z位置(米)'], 'b-')
ax.set_title('无人机3D飞行轨迹')
ax.set_xlabel('X (米)')
ax.set_ylabel('Y (米)')
ax.set_zlabel('Z (米)')
ax.grid(True)
plt.savefig(os.path.join(output_dir, '3D飞行轨迹.png'))

# 7. 按照飞行状态统计数据
state_stats = df.groupby('状态').agg({
    'Z位置(米)': ['mean', 'min', 'max'],
    '线速度X(米/秒)': ['mean', 'min', 'max'],
    '线速度Y(米/秒)': ['mean', 'min', 'max'],
    '线速度Z(米/秒)': ['mean', 'min', 'max']
})
print("\n按飞行状态统计:")
print(state_stats)
state_stats.to_csv(os.path.join(output_dir, '按状态统计.csv'))

# 8. 计算各状态的持续时间
state_duration = df.groupby('状态').agg({
    '时间(秒)': ['count', 'min', 'max']
})
state_duration['时间(秒)', '持续时间'] = state_duration['时间(秒)', 'max'] - state_duration['时间(秒)', 'min']
print("\n各状态持续时间:")
print(state_duration)
state_duration.to_csv(os.path.join(output_dir, '状态持续时间.csv'))

# 9. 计算总飞行距离
x_diff = np.diff(df['X位置(米)'])
y_diff = np.diff(df['Y位置(米)'])
z_diff = np.diff(df['Z位置(米)'])
distances = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
total_distance = np.sum(distances)
print(f"\n总飞行距离: {total_distance:.2f} 米")

# 10. 计算平均速度
total_time = df['时间(秒)'].iloc[-1] - df['时间(秒)'].iloc[0]
if total_time > 0:
    avg_speed = total_distance / total_time
    print(f"平均速度: {avg_speed:.2f} 米/秒")

# 保存所有图表
plt.close('all')
print(f"\n所有分析图表已保存到目录: {output_dir}")

print("\n数据分析完成!")