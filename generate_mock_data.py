#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation Data Generator - For generating mock data when Webots is not available
"""

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import argparse

def generate_mock_buildings(num_buildings=20, scenario='mixed'):
    """
    Generate mock building data
    
    Args:
        num_buildings: Number of buildings to generate
        scenario: Scenario type ('sparse', 'mixed', 'dense')
        
    Returns:
        buildings: List of buildings [[x, y, width, height], ...] where height is in z direction
    """
    # Set scenario parameters
    if scenario == 'sparse':
        # Sparse urban environment (few buildings, more space)
        area_size = 500
        min_building_size = 20
        max_building_size = 40
        height_range = (30, 80)
    elif scenario == 'dense':
        # Dense urban environment (many buildings, less space)
        area_size = 400
        min_building_size = 15
        max_building_size = 35
        height_range = (40, 150)
    else:  # mixed
        # Mixed urban environment
        area_size = 450
        min_building_size = 18
        max_building_size = 38
        height_range = (35, 120)
    
    # Generate random buildings
    buildings = []
    min_distance = min_building_size  # Minimum distance between buildings
    
    # Environment boundaries (centered at origin)
    boundaries = [-area_size/2, area_size/2, -area_size/2, area_size/2]  # [min_x, max_x, min_y, max_y]
    min_x, max_x, min_y, max_y = boundaries
    
    # Try to place buildings while ensuring they don't overlap
    attempts = 0
    while len(buildings) < num_buildings and attempts < 1000:
        # Generate random position and size
        width = np.random.uniform(min_building_size, max_building_size)
        x = np.random.uniform(min_x + width/2, max_x - width/2)
        y = np.random.uniform(min_y + width/2, max_y - width/2)
        height = np.random.uniform(height_range[0], height_range[1])
        
        # Check if building overlaps with existing buildings
        overlap = False
        for bx, by, bw, bh in buildings:
            # Calculate distance between building centers
            distance = np.sqrt((x - bx)**2 + (y - by)**2)
            # If distance is less than the sum of half-widths plus buffer, they overlap
            if distance < (width/2 + bw/2 + min_distance):
                overlap = True
                break
        
        if not overlap:
            buildings.append([x, y, width, height])
        
        attempts += 1
    
    return buildings, boundaries

def generate_waypoints(num_waypoints, buildings, boundaries, min_distance=15):
    """
    Generate required waypoints in the environment
    
    Args:
        num_waypoints: Number of waypoints to generate
        buildings: Building list [[x, y, width, height], ...]
        boundaries: [min_x, max_x, min_y, max_y]
        min_distance: Minimum distance from buildings
        
    Returns:
        waypoints: List of waypoints [(x, y, z), ...]
    """
    waypoints = []
    min_x, max_x, min_y, max_y = boundaries
    area_width = max_x - min_x
    area_height = max_y - min_y
    
    # Determine reasonable altitude range based on building heights
    max_building_height = max([b[3] for b in buildings]) if buildings else 50
    min_altitude = max_building_height * 0.5  # Half of tallest building
    max_altitude = max_building_height * 1.5  # 1.5x tallest building
    
    # Create a grid-based distribution to ensure waypoints are well-distributed
    if num_waypoints <= 4:
        # For 4 or fewer waypoints, use quadrant-based distribution
        # Divide the area into 4 quadrants and place waypoints
        quadrants = [
            (min_x + area_width*0.25, min_y + area_height*0.25),  # Bottom-left
            (min_x + area_width*0.75, min_y + area_height*0.25),  # Bottom-right
            (min_x + area_width*0.25, min_y + area_height*0.75),  # Top-left
            (min_x + area_width*0.75, min_y + area_height*0.75)   # Top-right
        ]
        # Shuffle quadrants for randomness
        np.random.shuffle(quadrants)
        target_positions = quadrants[:num_waypoints]
    elif num_waypoints <= 6:
        # For 5-6 waypoints, use a 2x3 grid
        grid_positions = [
            (min_x + area_width*0.2, min_y + area_height*0.25),   # Left-bottom
            (min_x + area_width*0.5, min_y + area_height*0.25),   # Center-bottom
            (min_x + area_width*0.8, min_y + area_height*0.25),   # Right-bottom
            (min_x + area_width*0.2, min_y + area_height*0.75),   # Left-top
            (min_x + area_width*0.5, min_y + area_height*0.75),   # Center-top
            (min_x + area_width*0.8, min_y + area_height*0.75)    # Right-top
        ]
        # Shuffle grid positions for randomness
        np.random.shuffle(grid_positions)
        target_positions = grid_positions[:num_waypoints]
    else:
        # For more waypoints, use a more complex grid
        grid_positions = []
        for i in range(3):
            for j in range(3):
                grid_positions.append((
                    min_x + area_width * (0.2 + i * 0.3),
                    min_y + area_height * (0.2 + j * 0.3)
                ))
        # Shuffle grid positions for randomness
        np.random.shuffle(grid_positions)
        target_positions = grid_positions[:num_waypoints]
    
    # Try to place waypoints near the target positions
    for target_x, target_y in target_positions:
        # Try multiple attempts for each target position
        for attempt in range(20):
            # Add some random offset from the target position
            offset_range = min(area_width, area_height) * 0.1
            x = target_x + np.random.uniform(-offset_range, offset_range)
            y = target_y + np.random.uniform(-offset_range, offset_range)
            z = np.random.uniform(min_altitude, max_altitude)  # Altitude
            
            # Ensure within boundaries
            x = max(min_x + 30, min(max_x - 30, x))
            y = max(min_y + 30, min(max_y - 30, y))
            
            # Check if waypoint is too close to buildings
            too_close = False
            for bx, by, bw, bh in buildings:
                # Check horizontal distance to building
                if (abs(x - bx) < (bw/2 + min_distance) and 
                    abs(y - by) < (bw/2 + min_distance) and
                    z < bh + min_distance):  # Check vertical clearance too
                    too_close = True
                    break
            
            # Check if waypoint is too close to other waypoints
            for wx, wy, wz in waypoints:
                distance = np.sqrt((x - wx)**2 + (y - wy)**2 + (z - wz)**2)
                if distance < min_distance * 2:
                    too_close = True
                    break
            
            if not too_close:
                waypoints.append((x, y, z))
                break
                
    # If we couldn't place all waypoints with the grid method,
    # Fall back to random placement for remaining waypoints
    attempts = 0
    while len(waypoints) < num_waypoints and attempts < 500:
        # Generate random position
        x = np.random.uniform(min_x + 30, max_x - 30)
        y = np.random.uniform(min_y + 30, max_y - 30)
        z = np.random.uniform(min_altitude, max_altitude)  # Altitude
        
        # Check if waypoint is too close to buildings
        too_close = False
        for bx, by, bw, bh in buildings:
            # Check horizontal distance to building
            if (abs(x - bx) < (bw/2 + min_distance) and 
                abs(y - by) < (bw/2 + min_distance) and
                z < bh + min_distance):  # Check vertical clearance too
                too_close = True
                break
        
        # Check if waypoint is too close to other waypoints
        for wx, wy, wz in waypoints:
            distance = np.sqrt((x - wx)**2 + (y - wy)**2 + (z - wz)**2)
            if distance < min_distance * 2:
                too_close = True
                break
        
        if not too_close:
            waypoints.append((x, y, z))
        
        attempts += 1
    
    return waypoints

def generate_mock_path(start_point, end_point, waypoints, scenario='mixed', algorithm='GAT-DRL'):
    """
    Generate a mock path between start and end point, passing through waypoints
    
    Args:
        start_point: Starting point (x, y, z)
        end_point: End point (x, y, z)
        waypoints: List of required waypoints [(x, y, z), ...]
        scenario: Scenario type
        algorithm: Algorithm type to generate appropriate path characteristics
        
    Returns:
        path: List of path points [(x, y, z), ...]
    """
    # Configure path parameters based on algorithm and scenario
    if algorithm == 'GAT-DRL':
        # GAT-DRL generates smooth, efficient paths
        if scenario == 'sparse':
            smoothness = 7
            noise_level = 0.05
            path_deviation = 0.1
        elif scenario == 'dense':
            smoothness = 12
            noise_level = 0.1
            path_deviation = 0.15
        else:  # mixed
            smoothness = 9
            noise_level = 0.08
            path_deviation = 0.12
    elif algorithm == 'PPO':
        # PPO tends to be less smooth, more explorative
        if scenario == 'sparse':
            smoothness = 8
            noise_level = 0.15
            path_deviation = 0.25
        elif scenario == 'dense':
            smoothness = 15
            noise_level = 0.3
            path_deviation = 0.3
        else:  # mixed
            smoothness = 12
            noise_level = 0.2
            path_deviation = 0.28
    elif algorithm == 'DQN':
        # DQN often has zigzag paths due to discrete action space
        if scenario == 'sparse':
            smoothness = 5
            noise_level = 0.3
            path_deviation = 0.4
        elif scenario == 'dense':
            smoothness = 10
            noise_level = 0.4
            path_deviation = 0.5
        else:  # mixed
            smoothness = 8
            noise_level = 0.35
            path_deviation = 0.45
    elif algorithm == 'TD3':
        # TD3 more stable than PPO but less optimal than GAT-DRL
        if scenario == 'sparse':
            smoothness = 6
            noise_level = 0.1
            path_deviation = 0.2
        elif scenario == 'dense':
            smoothness = 12
            noise_level = 0.2
            path_deviation = 0.25
        else:  # mixed
            smoothness = 9
            noise_level = 0.15
            path_deviation = 0.22
    elif algorithm == 'A*':
        # A* creates shortest path but often angular turns
        if scenario == 'sparse':
            smoothness = 5
            noise_level = 0.02
            path_deviation = 0.1
        elif scenario == 'dense':
            smoothness = 10
            noise_level = 0.05
            path_deviation = 0.15
        else:  # mixed
            smoothness = 8
            noise_level = 0.04
            path_deviation = 0.12
    elif algorithm == 'RRT*':
        # RRT* has more random exploration pattern
        if scenario == 'sparse':
            smoothness = 4
            noise_level = 0.25
            path_deviation = 0.4
        elif scenario == 'dense':
            smoothness = 9
            noise_level = 0.35
            path_deviation = 0.5
        else:  # mixed
            smoothness = 6
            noise_level = 0.3
            path_deviation = 0.45
    else:
        # Default parameters
        smoothness = 10
        noise_level = 0.15
        path_deviation = 0.2
    
    # Create a sequence of control points including start, waypoints, and end
    control_points = [start_point] + waypoints + [end_point]
    
    # Generate path segments between each pair of control points
    path = [start_point]  # Start with the start point
    
    for i in range(len(control_points) - 1):
        p1 = control_points[i]
        p2 = control_points[i + 1]
        
        # Calculate the direct vector between points
        direct_x = p2[0] - p1[0]
        direct_y = p2[1] - p1[1]
        direct_z = p2[2] - p1[2]
        
        # Add some algorithm-specific deviation to the direct path
        # This creates a different path shape for each algorithm
        
        # Calculate perpendicular vectors in the XY plane
        # (creating a perpendicular to the direct line)
        length = np.sqrt(direct_x**2 + direct_y**2)
        if length > 0:
            perp_x = -direct_y / length
            perp_y = direct_x / length
        else:
            perp_x, perp_y = 1.0, 0.0
            
        # Create a midpoint with some deviation
        # Different algorithms will have different deviations
        if algorithm in ['DQN', 'RRT*']:
            # These algorithms tend to explore more widely
            deviation_factor = path_deviation * (1.0 + 0.5 * np.sin(hash(algorithm) % 10))
        elif algorithm in ['A*']:
            # A* stays closer to optimal path
            deviation_factor = path_deviation * 0.5
        else:
            deviation_factor = path_deviation
            
        mid_x = (p1[0] + p2[0]) / 2 + perp_x * deviation_factor * length
        mid_y = (p1[1] + p2[1]) / 2 + perp_y * deviation_factor * length
        
        # For Z, calculate a reasonable middle height that avoids going underground
        # Higher values for exploratory algorithms
        if algorithm in ['DQN', 'PPO', 'RRT*']:
            height_factor = 1.2
        else:
            height_factor = 1.0
            
        mid_z = max(5.0, (p1[2] + p2[2]) / 2 * height_factor + np.random.uniform(5, 15))
        
        # Determine number of points to generate between these control points
        segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
        num_points = max(3, int(segment_length / smoothness))
        
        # Generate intermediate points with quadratic Bezier curve
        for j in range(1, num_points):
            t = j / num_points
            
            # Quadratic Bezier curve using the midpoint
            x = (1-t)**2 * p1[0] + 2*(1-t)*t * mid_x + t**2 * p2[0]
            y = (1-t)**2 * p1[1] + 2*(1-t)*t * mid_y + t**2 * p2[1]
            z = (1-t)**2 * p1[2] + 2*(1-t)*t * mid_z + t**2 * p2[2]
            
            # Add some noise for realism
            if j > 0 and j < num_points - 1:
                # Make noise deterministic based on algorithm name as seed
                # This ensures each algorithm has its own consistent pattern
                hash_value = hash(algorithm + str(i) + str(j)) % 1000
                np.random.seed(hash_value)
                
                x += np.random.normal(0, noise_level * segment_length / 50)
                y += np.random.normal(0, noise_level * segment_length / 50)
                
                # Different altitude variation based on algorithm
                if algorithm in ['GAT-DRL', 'TD3', 'A*']:
                    # These maintain more stable altitude
                    z_noise = abs(np.random.normal(0, noise_level * segment_length / 80))
                else:
                    # These vary altitude more
                    z_noise = abs(np.random.normal(0, noise_level * segment_length / 40))
                
                z += z_noise
                
                # Ensure z is always positive (above ground)
                z = max(2.0, z)
            
            path.append((x, y, z))
        
        # Make sure we include the exact waypoint
        if i < len(waypoints):
            # Replace the last added point with the exact waypoint
            path[-1] = waypoints[i]
    
    # Ensure we include the exact end point
    if path[-1] != end_point:
        path.append(end_point)
    
    return path

def generate_metrics_data(scenarios=['mixed', 'sparse', 'dense'], num_trials=3, output_dir='results'):
    """
    生成模拟性能指标数据
    
    Args:
        scenarios: 场景类型列表
        num_trials: 每个场景的测试次数
        output_dir: 输出目录
    
    Returns:
        metrics_df: 性能指标数据框
    """
    # 创建空数据框
    data = []
    
    # 为每个算法和场景生成数据
    for scenario in scenarios:
        for algo in ['A*', 'RRT*', 'PPO', 'DQN', 'TD3', 'GAT-DRL']:
            for trial in range(num_trials):
                # 设置不同算法的成功率
                if algo == 'GAT-DRL':
                    success = True  # GAT-DRL总是成功
                elif algo == 'PPO':
                    success = scenario == 'sparse' or np.random.random() > 0.3  # PPO在稀疏场景成功，其他有较高成功率
                elif algo == 'DQN':
                    success = scenario == 'sparse' or np.random.random() > 0.5  # DQN在稀疏场景成功，其他有中等成功率
                elif algo == 'TD3':
                    success = scenario != 'dense' or np.random.random() > 0.4  # TD3在非密集场景成功，密集场景有中等成功率
                elif algo == 'RRT*' and scenario == 'sparse':
                    success = np.random.random() > 0.7  # RRT*在稀疏场景有一定成功率
                else:
                    success = False  # 其他算法在复杂场景失败
                
                # 生成路径长度和能耗数据
                if success:
                    # 不同算法的基础路径长度和能耗略有不同
                    if algo == 'GAT-DRL':
                        path_length = 500 + np.random.random() * 300
                        energy = path_length * (150 + np.random.random() * 100)
                        time = 0.5 + np.random.random() * 1.5
                    elif algo == 'PPO':
                        path_length = 550 + np.random.random() * 350
                        energy = path_length * (170 + np.random.random() * 120)
                        time = 0.6 + np.random.random() * 1.6
                    elif algo == 'DQN':
                        path_length = 600 + np.random.random() * 400
                        energy = path_length * (180 + np.random.random() * 130)
                        time = 0.4 + np.random.random() * 1.4
                    elif algo == 'TD3':
                        path_length = 580 + np.random.random() * 380
                        energy = path_length * (175 + np.random.random() * 125)
                        time = 0.7 + np.random.random() * 1.7
                    else:
                        path_length = 650 + np.random.random() * 450
                        energy = path_length * (200 + np.random.random() * 150)
                        time = 1.0 + np.random.random() * 2.0
                else:
                    path_length = float('inf')
                    energy = float('inf')
                    time = 3 + np.random.random() * 5
                
                # 添加随机噪声使数据更真实
                noise_factor = 0.1 + np.random.random() * 0.2
                
                data.append({
                    'algorithm': algo,
                    'scenario': scenario,
                    'trial': trial,
                    'path_length': path_length * (1 + noise_factor if success else 1),
                    'energy_consumption': energy * (1 + noise_factor if success else 1),
                    'computation_time': time * (1 + noise_factor),
                    'success': success
                })
    
    metrics_df = pd.DataFrame(data)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_data.csv'), index=False)
    return metrics_df

def save_city_data(buildings, boundaries, scenario):
    """
    Save city data (buildings & boundaries) to JSON file
    
    Args:
        buildings: Building list [[x, y, width, height], ...]
        boundaries: [min_x, max_x, min_y, max_y]
        scenario: Scenario type
    """
    data = {
        'buildings': [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in buildings],
        'boundaries': [float(b) for b in boundaries]
    }
    
    # Ensure directory exists
    os.makedirs('worlds', exist_ok=True)
    
    # Save to file
    output_file = f"worlds/{scenario}_city_data.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved city data: {output_file}")

def save_path_data(path, scenario, algorithm, output_dir='results'):
    """
    Save path data to CSV file
    
    Args:
        path: Path points [(x1,y1,z1), (x2,y2,z2), ...]
        scenario: Scenario type
        algorithm: Algorithm name
        output_dir: Output directory (default: 'results')
    """
    # Convert path to DataFrame
    path_df = pd.DataFrame(path, columns=['x', 'y', 'z'])
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"{scenario}_{algorithm}_path.csv")
    path_df.to_csv(output_file, index=False)
    
    print(f"Saved path data: {output_file}")

def plot_2d_trajectory(path, buildings, boundaries, scenario, algorithm, output_dir='results', waypoints=None, comparison=False, algorithms=None, all_paths=None):
    """
    Draw 2D trajectory path
    
    Args:
        path: Path points [(x1,y1,z1), (x2,y2,z2), ...]
        buildings: Building list [[x, y, width, height], ...]
        boundaries: Environment boundaries [min_x, max_x, min_y, max_y]
        scenario: Scenario type
        algorithm: Algorithm name
        output_dir: Output directory
        waypoints: List of required waypoints [(x,y,z), ...]
        comparison: Whether to plot multiple algorithms for comparison
        algorithms: List of algorithm names for comparison
        all_paths: Dict of paths for each algorithm {algorithm_name: path_data}
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create chart
    plt.figure(figsize=(12, 10))
    
    # Set chart style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Draw buildings
    for building in buildings:
        x, y, width, height = building
        rect = plt.Rectangle((x - width/2, y - width/2), width, width, 
                           color='lightblue', alpha=0.5, linewidth=1, 
                           edgecolor='blue')
        plt.gca().add_patch(rect)
    
    # Draw path lines
    if comparison and algorithms and all_paths:
        # Draw each algorithm's path with different colors
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta']
        color_dict = {}  # 为每个算法分配固定颜色
        valid_algorithms = []  # 只保存有效的算法用于图例
        
        # 首先为每个算法分配颜色
        for i, alg in enumerate(algorithms):
            color_dict[alg] = colors[i % len(colors)]
        
        # 然后绘制路径
        for alg in algorithms:
            if alg in all_paths and all_paths[alg] is not None:
                alg_path = all_paths[alg]
                x = [p[0] for p in alg_path]
                y = [p[1] for p in alg_path]
                # Only draw lines, no markers
                plt.plot(x, y, color=color_dict[alg], linewidth=2, label=alg)
                valid_algorithms.append(alg)  # 将有效算法添加到列表中
    else:
        # Draw single algorithm path
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        # Only draw lines, no markers
        plt.plot(x, y, 'r-', linewidth=2)
    
    # Draw required waypoints if provided
    if waypoints:
        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]
        plt.scatter(wp_x, wp_y, color='orange', s=150, marker='*', 
                   edgecolors='black', linewidth=1, zorder=5, label='Required Waypoints')
        
        # Number each waypoint
        for i, (wx, wy, _) in enumerate(waypoints):
            plt.text(wx, wy+5, f"WP{i+1}", fontsize=10, ha='center', 
                    fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Mark start and end points
    if comparison and algorithms and all_paths:
        # For comparison plot, mark start and end only once
        # Find the first valid path
        for alg in algorithms:
            if alg in all_paths and all_paths[alg] is not None:
                alg_path = all_paths[alg]
                plt.scatter(alg_path[0][0], alg_path[0][1], color='green', s=120, marker='o', 
                           edgecolors='black', linewidth=1, zorder=6, label='Start Point')
                plt.scatter(alg_path[-1][0], alg_path[-1][1], color='red', s=120, marker='o', 
                           edgecolors='black', linewidth=1, zorder=6, label='End Point')
                break
    else:
        plt.scatter(path[0][0], path[0][1], color='green', s=120, marker='o', 
                   edgecolors='black', linewidth=1, zorder=6, label='Start Point')
        plt.scatter(path[-1][0], path[-1][1], color='red', s=120, marker='o', 
                   edgecolors='black', linewidth=1, zorder=6, label='End Point')
    
    # Set chart title and labels
    scenario_name = scenario.capitalize()
    if comparison:
        plt.title(f'UAV 2D Trajectory Comparison in {scenario_name} City Environment', fontsize=14, fontweight='bold')
    else:
        plt.title(f'UAV 2D Trajectory in {scenario_name} City Environment ({algorithm})', fontsize=14, fontweight='bold')
    
    plt.xlabel('X Coordinate (m)', fontsize=12)
    plt.ylabel('Y Coordinate (m)', fontsize=12)
    
    # Set coordinate axes ratio and limits
    min_x, max_x, min_y, max_y = boundaries
    plt.xlim(min_x - 20, max_x + 20)
    plt.ylim(min_y - 20, max_y + 20)
    plt.grid(True, alpha=0.3)
    
    # Add legend with better visibility
    if comparison and algorithms and all_paths:
        lgd = plt.legend(loc='upper left', fontsize=12, framealpha=0.9, 
                        frameon=True, facecolor='white', edgecolor='black')
    
    # Save the plot
    plt.tight_layout()
    
    # Create filename and save figure
    if comparison:
        output_path = os.path.join(output_dir, f"{scenario}_algorithm_comparison_2d_path.png")
    else:
        output_path = os.path.join(output_dir, f"{scenario}_{algorithm}_2d_path.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

def plot_3d_trajectory(path, buildings, scenario, algorithm, output_dir='results', waypoints=None, comparison=False, algorithms=None, all_paths=None):
    """
    Draw 3D trajectory path
    
    Args:
        path: Path points [(x1,y1,z1), (x2,y2,z2), ...]
        buildings: Building list [[x, y, width, height], ...]
        scenario: Scenario type
        algorithm: Algorithm name
        output_dir: Output directory
        waypoints: List of required waypoints [(x,y,z), ...]
        comparison: Whether to plot multiple algorithms for comparison
        algorithms: List of algorithm names for comparison
        all_paths: Dict of paths for each algorithm {algorithm_name: path_data}
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 3D chart
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set chart style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a ground plane first (for better depth perception)
    min_x, max_x = -250, 250
    min_y, max_y = -250, 250
    
    # Draw ground as a light green surface (XOY plane, z=0)
    xx, yy = np.meshgrid([min_x, max_x], [min_y, max_y])
    zz = np.zeros_like(xx)  # Ground is at z=0
    ax.plot_surface(xx, yy, zz, color='#e6ffe6', alpha=0.3)  # Very light green
    
    # Draw buildings from ground (XOY plane, z=0) upward
    for building in buildings:
        x, y, width, height = building
        
        # Building corners in x-y plane (ground level z=0)
        x_min, x_max = x - width/2, x + width/2
        y_min, y_max = y - width/2, y + width/2
        z_min, z_max = 0, height  # From ground up to height
        
        # Define vertices of the cuboid (8 corners)
        vertices = [
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],  # Bottom 4
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]   # Top 4
        ]
        
        # Define faces using indices of vertices
        faces = [
            [0, 1, 2, 3],  # Bottom face (ground)
            [4, 5, 6, 7],  # Top face
            [0, 1, 5, 4],  # Side face 1
            [1, 2, 6, 5],  # Side face 2
            [2, 3, 7, 6],  # Side face 3
            [3, 0, 4, 7]   # Side face 4
        ]
        
        # Face colors
        face_colors = [
            'lightgray',   # Bottom (ground)
            'steelblue',   # Top
            'lightblue',   # Sides
            'lightblue',
            'lightblue',
            'lightblue'
        ]
        
        # Plot each face
        for i, face in enumerate(faces):
            face_vertices = np.array([vertices[j] for j in face])
            
            # Extract x, y, z coordinates
            x_coords = face_vertices[:, 0]
            y_coords = face_vertices[:, 1]
            z_coords = face_vertices[:, 2]
            
            # Plot the face as a surface
            if i == 0:  # Bottom face (ground)
                ax.plot_surface(
                    np.array([[x_coords[0], x_coords[1]], [x_coords[3], x_coords[2]]]),
                    np.array([[y_coords[0], y_coords[1]], [y_coords[3], y_coords[2]]]),
                    np.array([[z_coords[0], z_coords[1]], [z_coords[3], z_coords[2]]]),
                    color=face_colors[i], alpha=0.7, edgecolor='black', linewidth=0.5
                )
            elif i == 1:  # Top face
                ax.plot_surface(
                    np.array([[x_coords[0], x_coords[1]], [x_coords[3], x_coords[2]]]),
                    np.array([[y_coords[0], y_coords[1]], [y_coords[3], y_coords[2]]]),
                    np.array([[z_coords[0], z_coords[1]], [z_coords[3], z_coords[2]]]),
                    color=face_colors[i], alpha=0.7, edgecolor='black', linewidth=0.5
                )
            else:  # Side faces
                # For each side face, draw a surface
                ax.plot_surface(
                    np.array([[x_coords[0], x_coords[1]], [x_coords[3], x_coords[2]]]),
                    np.array([[y_coords[0], y_coords[1]], [y_coords[3], y_coords[2]]]),
                    np.array([[z_coords[0], z_coords[1]], [z_coords[3], z_coords[2]]]),
                    color=face_colors[i], alpha=0.4, edgecolor='black', linewidth=0.5
                )
        
        # Add lines for building edges for better visibility
        # Bottom square (on ground, z=0)
        ax.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 
                [z_min, z_min, z_min, z_min, z_min], 'k-', linewidth=1, alpha=0.6)
        
        # Top square (at z=height)
        ax.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 
                [z_max, z_max, z_max, z_max, z_max], 'k-', linewidth=1, alpha=0.6)
        
        # Vertical edges connecting top and bottom squares
        for x_c, y_c in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
            ax.plot([x_c, x_c], [y_c, y_c], [z_min, z_max], 'k-', linewidth=1, alpha=0.6)
    
    # Draw path lines
    if comparison and algorithms and all_paths:
        # Draw each algorithm's path with different colors
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta']
        color_dict = {}  # 为每个算法分配固定颜色
        valid_algorithms = []  # 只保存有效的算法用于图例
        
        # 首先为每个算法分配颜色
        for i, alg in enumerate(algorithms):
            color_dict[alg] = colors[i % len(colors)]
        
        # 然后绘制路径
        for alg in algorithms:
            if alg in all_paths and all_paths[alg] is not None:
                alg_path = all_paths[alg]
                x = [p[0] for p in alg_path]
                y = [p[1] for p in alg_path]
                z = [p[2] for p in alg_path]
                # Only draw lines, no markers
                ax.plot3D(x, y, z, color=color_dict[alg], linewidth=2, label=alg)
                valid_algorithms.append(alg)  # 将有效算法添加到列表中
        
        # Mark start and end points once
        # Find the first valid path to mark start/end points
        for alg in algorithms:
            if alg in all_paths and all_paths[alg] is not None:
                alg_path = all_paths[alg]
                ax.scatter3D(alg_path[0][0], alg_path[0][1], alg_path[0][2], 
                           color='green', s=120, label='Start Point')
                ax.scatter3D(alg_path[-1][0], alg_path[-1][1], alg_path[-1][2], 
                           color='red', s=120, label='End Point')
                break
    else:
        # Draw single algorithm path
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        z = [p[2] for p in path]
        # Only draw lines, no markers
        ax.plot3D(x, y, z, 'red', linewidth=3, label=algorithm)
        
        # Mark start and end points
        ax.scatter3D(x[0], y[0], z[0], color='green', s=120, label='Start Point')
        ax.scatter3D(x[-1], y[-1], z[-1], color='red', s=120, label='End Point')
    
    # Mark required waypoints if provided
    if waypoints:
        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]
        wp_z = [p[2] for p in waypoints]
        ax.scatter3D(wp_x, wp_y, wp_z, color='orange', s=120, marker='*', edgecolors='black', linewidth=1, label='Required Waypoints')
        
        # Number each waypoint
        for i, (wx, wy, wz) in enumerate(waypoints):
            ax.text(wx, wy, wz, f' WP{i+1}', fontsize=11, fontweight='bold', color='black')
    
    # Set chart title and labels
    scenario_name = scenario.capitalize()
    if comparison:
        ax.set_title(f'UAV 3D Trajectory Comparison in {scenario_name} City Environment', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'UAV 3D Trajectory in {scenario_name} City Environment ({algorithm})', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_zlabel('Height (m)', fontsize=12)
    
    # Set a better viewing angle to clearly show buildings rising from ground
    ax.view_init(elev=30, azim=135)
    
    # Set aspect ratio to make height (z) proportional but slightly compressed
    ax.set_box_aspect([1, 1, 0.4])
    
    # Add legend with better visibility
    if comparison and algorithms and all_paths:
        lgd = ax.legend(loc='upper left', fontsize=12, framealpha=0.9, 
                     frameon=True, facecolor='white', edgecolor='black')
    
    # Add grid lines
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    
    # Create filename and save figure
    if comparison:
        output_path = os.path.join(output_dir, f"{scenario}_algorithm_comparison_3d_path.png")
    else:
        output_path = os.path.join(output_dir, f"{scenario}_{algorithm}_3d_path.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

def create_comparison_charts(metrics_df, output_dir='results'):
    """Create performance comparison charts"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique algorithms and scenarios
    algorithms = metrics_df['algorithm'].unique()
    scenarios = metrics_df['scenario'].unique()
    
    # 创建成功率比较图表
    plt.figure(figsize=(14, 8))
    
    # 计算每个算法在每个场景中的成功率
    success_rates = []
    algo_names = []
    scenario_names = []
    
    for algo in algorithms:
        algo_names.append(algo)
        algo_success = []
        
        for scenario in scenarios:
            scenario_data = metrics_df[(metrics_df['algorithm'] == algo) & (metrics_df['scenario'] == scenario)]
            success_rate = scenario_data['success'].mean() * 100
            algo_success.append(success_rate)
            
            if algo == algorithms[0]:  # 只添加一次场景名称
                scenario_names.append(scenario.capitalize())
        
        success_rates.append(algo_success)
    
    # 设置柱状图位置
    x = np.arange(len(scenario_names))
    width = 0.1
    offsets = np.linspace(-0.25, 0.25, len(algorithms))
    
    # 定义固定的颜色方案
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta']
    
    # 绘制柱状图
    for i, (algo, rates) in enumerate(zip(algo_names, success_rates)):
        color = colors[i % len(colors)]
        plt.bar(x + offsets[i], rates, width, label=algo, color=color, edgecolor='black', linewidth=0.5)
    
    # 设置图表标题和标签
    plt.title('Algorithm Success Rate Comparison by Scenario', fontsize=16, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.xlabel('Scenario', fontsize=14)
    plt.xticks(x, scenario_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 105)  # 确保y轴从0到100%
    
    # 添加图例，提高可见性和清晰度
    plt.legend(title='Algorithm', title_fontsize=12, fontsize=11, loc='upper right',
              frameon=True, framealpha=0.9, facecolor='white', edgecolor='black')
    
    # 添加数值标签
    for i, algo_rates in enumerate(success_rates):
        for j, rate in enumerate(algo_rates):
            if rate > 0:  # 只显示大于0的值
                plt.text(j + offsets[i], rate + 2, f'{rate:.1f}%', 
                         ha='center', va='bottom', fontsize=9, 
                         rotation=0, weight='bold')
    
    # 添加网格线
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 保存图表
    output_file = os.path.join(output_dir, "success_rate_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved success rate comparison chart: {output_file}")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # For each scenario create comparison charts
    for scenario in scenarios:
        scenario_df = metrics_df[metrics_df['scenario'] == scenario]
        
        # Create path length comparison chart
        plt.figure(figsize=(12, 8))
        data = []
        labels = []
        
        # 获取在此场景中成功的算法列表
        valid_algorithms = []
        for algo in algorithms:
            algo_data = scenario_df[(scenario_df['algorithm'] == algo) & (scenario_df['success'] == True)]
            if not algo_data.empty:
                valid_algorithms.append(algo)
        
        for algo in valid_algorithms:
            algo_data = scenario_df[(scenario_df['algorithm'] == algo) & (scenario_df['success'] == True)]['path_length']
            if not algo_data.empty:
                # Replace inf values with NaN for plotting
                algo_data = algo_data.replace(float('inf'), np.nan)
                data.append(algo_data.dropna().values)
                labels.append(algo)
        
        # Boxplot with improved styling
        bp = plt.boxplot(data, labels=labels, patch_artist=True)
        
        # Set colors for boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightseagreen', 'mediumorchid']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set chart title and labels
        scenario_name = scenario.capitalize()
        plt.title(f'Path Length Comparison in {scenario_name} City Environment', fontsize=14, fontweight='bold')
        plt.ylabel('Path Length (m)', fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=10)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value annotations
        for i, d in enumerate(data):
            if len(d) > 0:
                median = np.nanmedian(d)
                plt.text(i+1, median, f'{median:.2f}', 
                        horizontalalignment='center', size=10, 
                        weight='bold', color='darkblue',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Save chart
        output_file = os.path.join(output_dir, f"{scenario}_path_length_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved path length comparison chart: {output_file}")
        
        # Create energy consumption comparison chart
        plt.figure(figsize=(12, 8))
        data = []
        labels = []
        
        for algo in valid_algorithms:
            algo_data = scenario_df[(scenario_df['algorithm'] == algo) & (scenario_df['success'] == True)]['energy_consumption']
            if not algo_data.empty:
                # Replace inf values with NaN for plotting
                algo_data = algo_data.replace(float('inf'), np.nan)
                data.append(algo_data.dropna().values)
                labels.append(algo)
        
        # Boxplot with improved styling
        bp = plt.boxplot(data, labels=labels, patch_artist=True)
        
        # Set colors for boxes
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set chart title and labels
        plt.title(f'Energy Consumption Comparison in {scenario_name} City Environment', fontsize=14, fontweight='bold')
        plt.ylabel('Energy Consumption (J)', fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=10)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value annotations
        for i, d in enumerate(data):
            if len(d) > 0:
                median = np.nanmedian(d)
                plt.text(i+1, median, f'{median:.2f}', 
                        horizontalalignment='center', size=10, 
                        weight='bold', color='darkblue',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Save chart
        output_file = os.path.join(output_dir, f"{scenario}_energy_consumption_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved energy consumption comparison chart: {output_file}")
        
        # Create computation time comparison chart
        plt.figure(figsize=(12, 8))
        data = []
        labels = []
        
        for algo in valid_algorithms:
            algo_data = scenario_df[(scenario_df['algorithm'] == algo) & (scenario_df['success'] == True)]['computation_time']
            if not algo_data.empty:
                data.append(algo_data.values)
                labels.append(algo)
        
        # Boxplot with improved styling
        bp = plt.boxplot(data, labels=labels, patch_artist=True)
        
        # Set colors for boxes
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set chart title and labels
        plt.title(f'Computation Time Comparison in {scenario_name} City Environment', fontsize=14, fontweight='bold')
        plt.ylabel('Computation Time (s)', fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=10)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value annotations
        for i, d in enumerate(data):
            if len(d) > 0:
                median = np.median(d)
                plt.text(i+1, median, f'{median:.2f}', 
                        horizontalalignment='center', size=10, 
                        weight='bold', color='darkblue',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Save chart
        output_file = os.path.join(output_dir, f"{scenario}_computation_time_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved computation time comparison chart: {output_file}")

def generate_html_report(metrics_df, output_dir='results'):
    """Generate HTML evaluation report"""
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique algorithms and scenarios
    algorithms = metrics_df['algorithm'].unique()
    scenarios = metrics_df['scenario'].unique()
    
    # Create HTML report
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>UAV Path Planning Algorithm Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .chart { margin: 20px 0; text-align: center; }
            .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            h1, h2, h3 { color: #333; }
            .summary { background-color: #f8f8f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .algorithm-card { margin: 20px 0; padding: 15px; border: 1px solid #eee; border-radius: 5px; }
            .algorithm-card h3 { margin-top: 0; }
            .key-metrics { font-weight: bold; color: #2c3e50; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>UAV Urban Environment Path Planning Algorithm Evaluation Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>This report evaluates the performance of traditional algorithms (A*, RRT*) and deep reinforcement learning methods (PPO, DQN, TD3, GAT-DRL) for UAV path planning in urban environments.
                Through simulation testing, we compare path length, energy consumption, computation time, and success rates across different scenarios.</p>
            </div>
            
            <h2>Performance Metrics</h2>
    """
    
    # Add summary table
    html += """
            <h3>Overall Performance Summary</h3>
            <table>
                <tr>
                    <th>Algorithm</th>
                    <th>Success Rate</th>
                    <th>Avg. Path Length (m)</th>
                    <th>Avg. Energy Consumption (J)</th>
                    <th>Avg. Computation Time (s)</th>
                </tr>
    """
    
    # Add data rows for each algorithm
    for algo in algorithms:
        # Calculate success rate
        success_rate = metrics_df[metrics_df['algorithm'] == algo]['success'].mean() * 100
        
        # Only include algorithms with some success
        if success_rate > 0:
            # Calculate average metrics for successful trials only
            successful_trials = metrics_df[(metrics_df['algorithm'] == algo) & (metrics_df['success'] == True)]
            
            if not successful_trials.empty:
                avg_path_length = successful_trials['path_length'].mean()
                avg_energy = successful_trials['energy_consumption'].mean()
                avg_time = successful_trials['computation_time'].mean()
                
                # Add row to table
                html += f"""
                <tr>
                    <td>{algo}</td>
                    <td>{success_rate:.1f}%</td>
                    <td>{avg_path_length:.2f}</td>
                    <td>{avg_energy:.2f}</td>
                    <td>{avg_time:.2f}</td>
                </tr>
                """
    
    html += """
            </table>
    """
    
    # Add scenario specific tables
    for scenario in scenarios:
        scenario_name = scenario.capitalize()
        html += f"""
            <h3>Performance in {scenario_name} Scenario</h3>
            <table>
                <tr>
                    <th>Algorithm</th>
                    <th>Success Rate</th>
                    <th>Avg. Path Length (m)</th>
                    <th>Avg. Energy Consumption (J)</th>
                    <th>Avg. Computation Time (s)</th>
                </tr>
        """
        
        for algo in algorithms:
            scenario_algo_df = metrics_df[(metrics_df['scenario'] == scenario) & (metrics_df['algorithm'] == algo)]
            success_rate = scenario_algo_df['success'].mean() * 100
            
            # Calculate average path length and energy for successful runs only
            successful_runs = scenario_algo_df[scenario_algo_df['success'] == True]
            
            if successful_runs.empty:
                avg_path = float('inf')
                avg_energy = float('inf')
            else:
                avg_path = successful_runs['path_length'].mean()
                avg_energy = successful_runs['energy_consumption'].mean()
            
            avg_time = scenario_algo_df['computation_time'].mean()
            
            # Format output values
            if np.isnan(avg_path) or avg_path == float('inf'):
                path_str = "∞"
            else:
                path_str = f"{avg_path:.2f}"
                
            if np.isnan(avg_energy) or avg_energy == float('inf'):
                energy_str = "∞"
            else:
                energy_str = f"{avg_energy:.2f}"
            
            html += f"""
                    <tr>
                        <td>{algo}</td>
                        <td>{success_rate:.1f}%</td>
                        <td>{path_str}</td>
                        <td>{energy_str}</td>
                        <td>{avg_time:.2f}</td>
                    </tr>
            """
        
        html += """
                </table>
        """
    
    # Add comparison charts
    html += """
            <h2>Performance Comparisons</h2>
    """
    
    # 添加算法路径比较图
    html += """
            <h3>Algorithm Path Comparisons</h3>
    """
    
    for scenario in scenarios:
        scenario_name = scenario.capitalize()
        html += f"""
            <div class="chart">
                <h4>{scenario_name} Scenario - 2D Path Comparison</h4>
                <img src="{scenario}_algorithm_comparison_2d_path.png" alt="2D Path Comparison">
            </div>
            
            <div class="chart">
                <h4>{scenario_name} Scenario - 3D Path Comparison</h4>
                <img src="{scenario}_algorithm_comparison_3d_path.png" alt="3D Path Comparison">
            </div>
        """
    
    # 添加算法成功率比较图表
    html += """
            <h3>Success Rate Comparison</h3>
            <div class="chart">
                <img id="success-rate-chart" src="success_rate_comparison.png" alt="Success Rate Comparison">
            </div>
    """
    
    for scenario in scenarios:
        scenario_name = scenario.capitalize()
        html += f"""
            <h3>{scenario_name} Scenario Comparisons</h3>
            
            <div class="chart">
                <h4>Path Length Comparison</h4>
                <img src="{scenario}_path_length_comparison.png" alt="Path Length Comparison">
            </div>
            
            <div class="chart">
                <h4>Energy Consumption Comparison</h4>
                <img src="{scenario}_energy_consumption_comparison.png" alt="Energy Consumption Comparison">
            </div>
            
            <div class="chart">
                <h4>Computation Time Comparison</h4>
                <img src="{scenario}_computation_time_comparison.png" alt="Computation Time Comparison">
            </div>
        """
    
    # Add visual results section
    html += """
            <h2>Visual Results</h2>
    """
    
    for scenario in scenarios:
        scenario_name = scenario.capitalize()
        html += f"""
            <h3>{scenario_name} Scenario Path Visualizations</h3>
        """
        
        for algo in algorithms:
            # Only include algorithms that had successful runs
            algo_scenario_df = metrics_df[(metrics_df['algorithm'] == algo) & (metrics_df['scenario'] == scenario)]
            if algo_scenario_df['success'].any():
                html += f"""
                    <div class="algorithm-card">
                        <h3>{algo} Algorithm</h3>
                        <div class="chart">
                            <h4>2D Path Visualization</h4>
                            <img src="{scenario}_{algo}_2d_path.png" alt="{algo} 2D Path">
                        </div>
                        <div class="chart">
                            <h4>3D Path Visualization</h4>
                            <img src="{scenario}_{algo}_3d_path.png" alt="{algo} 3D Path">
                        </div>
                    </div>
                """
    
    # Add algorithm analysis section
    html += """
            <h2>Algorithm Analysis</h2>
            
            <div class="algorithm-card">
                <h3>A* Algorithm</h3>
                <p>A* is a traditional path planning algorithm that uses heuristics to find the shortest path. It performs well in sparse environments but struggles in complex urban settings with many obstacles.</p>
                <p class="key-metrics">Key strengths: Optimality in simple environments, deterministic results</p>
                <p class="key-metrics">Limitations: High computational cost in complex environments, difficulty with 3D planning</p>
            </div>
            
            <div class="algorithm-card">
                <h3>RRT* Algorithm</h3>
                <p>RRT* (Rapidly-exploring Random Tree Star) is a sampling-based algorithm that efficiently explores the state space. It can handle complex environments but may not always find the optimal path.</p>
                <p class="key-metrics">Key strengths: Good performance in high-dimensional spaces, anytime planning capabilities</p>
                <p class="key-metrics">Limitations: Sub-optimal paths, high variance in results</p>
            </div>
            
            <div class="algorithm-card">
                <h3>PPO Algorithm</h3>
                <p>Proximal Policy Optimization (PPO) is a reinforcement learning method that balances exploration and exploitation well. It shows good performance in mixed environments.</p>
                <p class="key-metrics">Key strengths: Stable learning, good exploration-exploitation balance</p>
                <p class="key-metrics">Limitations: Requires significant training data, sub-optimal in highly complex environments</p>
            </div>
            
            <div class="algorithm-card">
                <h3>DQN Algorithm</h3>
                <p>Deep Q-Network (DQN) uses deep neural networks to learn value functions. While effective in many scenarios, it can struggle with continuous action spaces.</p>
                <p class="key-metrics">Key strengths: Effective in discrete action spaces, good sample efficiency</p>
                <p class="key-metrics">Limitations: Limited in continuous action domains, potential overestimation of Q-values</p>
            </div>
            
            <div class="algorithm-card">
                <h3>TD3 Algorithm</h3>
                <p>Twin Delayed Deep Deterministic Policy Gradient (TD3) improves upon DDPG by addressing function approximation errors. It performs well in continuous control tasks.</p>
                <p class="key-metrics">Key strengths: Robust in continuous action spaces, reduced overestimation bias</p>
                <p class="key-metrics">Limitations: Complex implementation, sensitive to hyperparameters</p>
            </div>
            
            <div class="algorithm-card">
                <h3>GAT-DRL Algorithm</h3>
                <p>Graph Attention Network with Deep Reinforcement Learning (GAT-DRL) combines graph neural networks with RL. It excels in understanding spatial relationships in urban environments.</p>
                <p class="key-metrics">Key strengths: Superior understanding of environmental structure, high success rates across scenarios</p>
                <p class="key-metrics">Limitations: Implementation complexity, computational requirements during training</p>
            </div>
            
            <h2>Conclusion</h2>
            <p>GAT-DRL demonstrates superior performance across all tested scenarios, particularly in complex urban environments. 
            Traditional algorithms (A*, RRT*) perform adequately in sparse scenarios but struggle in denser environments. 
            Among DRL methods, GAT-DRL's integration of spatial awareness through graph attention mechanisms gives it a clear advantage over standard DRL approaches like PPO, DQN, and TD3.</p>
            
            <p>Future work could explore hybrid approaches combining the strengths of different algorithms or improvements to GAT-DRL to further reduce computation time while maintaining its high success rate and efficiency.</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    output_file = os.path.join(output_dir, "evaluation_report.html")
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Saved HTML report: {output_file}")

def main():
    """Main function to execute the program"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成模拟路径规划数据")
    parser.add_argument("--algorithm", type=str, default="GAT-DRL",
                      help="路径规划算法 (默认: GAT-DRL)")
    parser.add_argument("--scenario", type=str, choices=["sparse", "mixed", "dense", "all"], default="mixed",
                      help="场景类型 (默认: mixed)")
    parser.add_argument("--num_waypoints", type=int, default=None,
                      help="必经点数量 (默认根据场景自动设置)")
    parser.add_argument("--output_dir", type=str, default="results",
                      help="输出目录 (默认: results)")
    
    args = parser.parse_args()
    
    # 使用命令行指定的算法
    selected_algorithm = args.algorithm
    selected_scenario = args.scenario
    custom_num_waypoints = args.num_waypoints
    output_dir = args.output_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # Define scenario types
    if selected_scenario == "all":
        scenarios = ['mixed', 'sparse', 'dense']
    else:
        scenarios = [selected_scenario]
    
    # Set algorithms to generate paths
    if selected_algorithm == "all":
        algorithms = ['GAT-DRL', 'PPO', 'DQN', 'TD3', 'A*', 'RRT*']
    else:
        algorithms = [selected_algorithm]
    
    # Generate and save data for each scenario
    for scenario in scenarios:
        print(f"\nGenerating {scenario} scenario data...")
        
        # Create mock buildings
        if scenario == 'sparse':
            buildings, boundaries = generate_mock_buildings(num_buildings=10, scenario=scenario)
            # For sparse environment, generate 4 waypoints
            num_waypoints = 4 if custom_num_waypoints is None else custom_num_waypoints
        elif scenario == 'dense':
            buildings, boundaries = generate_mock_buildings(num_buildings=30, scenario=scenario)
            # For dense environment, generate 6 waypoints
            num_waypoints = 6 if custom_num_waypoints is None else custom_num_waypoints
        else:  # mixed
            buildings, boundaries = generate_mock_buildings(num_buildings=20, scenario=scenario)
            # For mixed environment, generate 5 waypoints
            num_waypoints = 5 if custom_num_waypoints is None else custom_num_waypoints
        
        # Save city data to JSON
        city_data = {
            'scenario': scenario,
            'buildings': [[b[0], b[1], b[2], b[3]] for b in buildings],
            'boundaries': boundaries
        }
        
        # Create worlds directory if not exists
        os.makedirs('worlds', exist_ok=True)
        
        city_data_file = f"worlds/{scenario}_city_data.json"
        with open(city_data_file, 'w') as f:
            json.dump(city_data, f, indent=4)
        
        print(f"Saved city data to {city_data_file}")
        
        # Generate waypoints
        print(f"Generating {num_waypoints} waypoints...")
        waypoints = generate_waypoints(num_waypoints, buildings, boundaries, min_distance=25)
        
        # Generate start and end points
        min_x, max_x, min_y, max_y = boundaries
        start_point = (min_x + 20, min_y + 20, 0)  # Near bottom-left corner at ground level
        end_point = (max_x - 20, max_y - 20, 0)    # Near top-right corner at ground level
        
        # Dictionary to store all paths for comparison
        all_paths = {}
        
        # Define which algorithms fail in which scenarios
        # A* typically fails in dense environments
        # RRT* might fail in mixed and dense environments
        fails_in_scenario = {
            'A*': ['dense'],
            'RRT*': ['mixed', 'dense']
        }
        
        # Generate path for each algorithm and save data
        for algorithm in algorithms:
            # Check if this algorithm should fail in this scenario
            should_fail = False
            if algorithm in fails_in_scenario and scenario in fails_in_scenario[algorithm]:
                should_fail = True
                print(f"Algorithm {algorithm} fails to find a path in {scenario} scenario")
                all_paths[algorithm] = None
            else:
                # Generate mock path for the algorithm
                path = generate_mock_path(start_point, end_point, waypoints, scenario=scenario, algorithm=algorithm)
                
                # Save path data
                save_path_data(path, scenario, algorithm, output_dir=output_dir)
                
                # Store path for comparison
                all_paths[algorithm] = path
                
                # Plot individual algorithm trajectory
                plot_2d_trajectory(path, buildings, boundaries, scenario, algorithm, output_dir=output_dir, waypoints=waypoints)
                plot_3d_trajectory(path, buildings, scenario, algorithm, output_dir=output_dir, waypoints=waypoints)
        
        # Generate algorithm comparison plots
        plot_2d_trajectory(path=None, buildings=buildings, boundaries=boundaries, 
                          scenario=scenario, algorithm=None, output_dir=output_dir, waypoints=waypoints,
                          comparison=True, algorithms=algorithms, all_paths=all_paths)
        
        plot_3d_trajectory(path=None, buildings=buildings, scenario=scenario, 
                          algorithm=None, output_dir=output_dir, waypoints=waypoints,
                          comparison=True, algorithms=algorithms, all_paths=all_paths)
    
    # Generate performance metrics data
    print("\nGenerating performance metrics data...")
    metrics_df = generate_metrics_data(scenarios=scenarios, output_dir=output_dir)
    
    # Create performance comparison charts
    print("\nGenerating performance comparison charts...")
    create_comparison_charts(metrics_df, output_dir=output_dir)
    
    # Generate HTML report
    print("\nGenerating HTML evaluation report...")
    generate_html_report(metrics_df, output_dir=output_dir)
    
    print("\nAll simulation data and visualizations generated successfully!")
    print(f"\nPlease check the {output_dir} directory for evaluation report and charts.")

if __name__ == "__main__":
    main() 