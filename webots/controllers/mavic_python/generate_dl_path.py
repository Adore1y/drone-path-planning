#!/usr/bin/env python3

"""
Deep Learning Path Generator
For generating simulated deep learning navigation paths to compare with conventional waypoint navigation
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import argparse
import random
from scipy.interpolate import interp1d, splprep, splev
from sklearn.cluster import KMeans

# Command line arguments
parser = argparse.ArgumentParser(description='Generate simulated deep learning path data')
parser.add_argument('--data_dir', type=str, default='flight_data', help='Data directory')
parser.add_argument('--world_file', type=str, default='../../worlds/mixed_scenario.wbt', help='World file path')
parser.add_argument('--model_type', type=str, default='drl', choices=['drl', 'cnn', 'lstm', 'hybrid'], help='Simulated deep learning model type')
parser.add_argument('--smoothness', type=float, default=0.7, help='Path smoothness (0-1)')
parser.add_argument('--efficiency', type=float, default=0.8, help='Path efficiency (0-1)')
parser.add_argument('--obstacle_avoidance', type=float, default=0.85, help='Obstacle avoidance efficiency (0-1)')
args = parser.parse_args()

def ensure_dir(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_waypoints_from_world(world_file_path):
    """Extract waypoints from world file"""
    try:
        waypoints = []
        extracting = False
        
        with open(world_file_path, 'r') as file:
            for line in file:
                if '# BEGIN_WAYPOINTS' in line:
                    extracting = True
                    continue
                elif '# END_WAYPOINTS' in line:
                    extracting = False
                    continue
                
                if extracting and line.strip() and not line.strip().startswith('#'):
                    coords = line.strip().split()
                    if len(coords) >= 3:
                        try:
                            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                            waypoints.append([x, y, z])
                        except ValueError:
                            pass
        
        if not waypoints:
            # Default waypoints
            print("No waypoints found, using default waypoints")
            waypoints = [
                [0, 0, 1],
                [2, 2, 1.5],
                [4, 0, 2],
                [6, -2, 1.5],
                [8, 0, 1]
            ]
        
        return np.array(waypoints)
    
    except Exception as e:
        print(f"Error extracting waypoints: {e}")
        # Default waypoints
        return np.array([
            [0, 0, 1],
            [2, 2, 1.5],
            [4, 0, 2],
            [6, -2, 1.5],
            [8, 0, 1]
        ])

def extract_obstacles_from_world(world_file_path):
    """Extract obstacle positions from world file"""
    try:
        obstacles = []
        
        with open(world_file_path, 'r') as file:
            content = file.read()
            
            # Find all obstacle definitions
            box_matches = re.finditer(r'Box\s*\{[^}]*?translation\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)[^}]*?size\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)', content, re.DOTALL)
            
            for match in box_matches:
                x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                width, height, depth = float(match.group(4)), float(match.group(5)), float(match.group(6))
                
                # Store obstacle position and size
                obstacles.append({
                    'position': [x, y, z],
                    'size': [width, height, depth]
                })
        
        return obstacles
    
    except Exception as e:
        print(f"Error extracting obstacles: {e}")
        return []

def generate_waypoint_path(waypoints, smoothness=0.7, num_points=1000):
    """Generate smooth path between waypoints"""
    if len(waypoints) < 2:
        return waypoints
    
    # Apply spline interpolation to trajectory
    if len(waypoints) > 3:
        tck, u = splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=smoothness, k=min(3, len(waypoints)-1))
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new, z_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new, z_new])
    else:
        # If too few waypoints, use linear interpolation
        t = np.linspace(0, 1, len(waypoints))
        t_new = np.linspace(0, 1, num_points)
        
        x_interp = interp1d(t, waypoints[:, 0])
        y_interp = interp1d(t, waypoints[:, 1])
        z_interp = interp1d(t, waypoints[:, 2])
        
        x_new = x_interp(t_new)
        y_new = y_interp(t_new)
        z_new = z_interp(t_new)
        
        return np.column_stack([x_new, y_new, z_new])

def is_point_in_obstacle(point, obstacle, margin=0.5):
    """Check if point is inside obstacle (including safety margin)"""
    pos = obstacle['position']
    size = obstacle['size']
    
    # Calculate obstacle boundaries (considering margin)
    min_x = pos[0] - size[0]/2 - margin
    max_x = pos[0] + size[0]/2 + margin
    min_y = pos[1] - size[1]/2 - margin
    max_y = pos[1] + size[1]/2 + margin
    min_z = pos[2] - size[2]/2 - margin
    max_z = pos[2] + size[2]/2 + margin
    
    # Check if point is inside boundaries
    return (min_x <= point[0] <= max_x and
            min_y <= point[1] <= max_y and
            min_z <= point[2] <= max_z)

def apply_obstacle_avoidance(path, obstacles, avoidance_efficiency=0.85):
    """Apply obstacle avoidance to modify path"""
    if not obstacles:
        return path
    
    avoided_path = path.copy()
    
    for i in range(len(avoided_path)):
        # Check if current point is inside any obstacle
        for obstacle in obstacles:
            if is_point_in_obstacle(avoided_path[i], obstacle):
                # Find avoidance direction (based on efficiency parameter, may not be optimal)
                pos = obstacle['position']
                size = obstacle['size']
                
                # Calculate vector from obstacle center to current point
                direction = avoided_path[i] - np.array(pos)
                
                # Normalize vector
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    direction = np.array([0, 0, 1])  # Default move upward
                
                # Adjust avoidance distance based on efficiency
                avoid_distance = max(size) * (1.0 + (1.0 - avoidance_efficiency) * 0.5)
                
                # Apply avoidance, move point
                avoided_path[i] = np.array(pos) + direction * avoid_distance
                
                # For DRL model, add some randomness to simulate exploration behavior
                if args.model_type == 'drl':
                    random_factor = (1.0 - avoidance_efficiency) * 0.3
                    avoided_path[i] += np.random.normal(0, random_factor, 3)
    
    # Apply different smoothing strategies based on model type
    if args.model_type == 'lstm':
        # LSTM models may produce smoother trajectories
        tck, u = splprep([avoided_path[:, 0], avoided_path[:, 1], avoided_path[:, 2]], s=args.smoothness*5, k=3)
        u_new = np.linspace(0, 1, len(avoided_path))
        x_new, y_new, z_new = splev(u_new, tck)
        avoided_path = np.column_stack([x_new, y_new, z_new])
    elif args.model_type == 'cnn':
        # CNN models may produce piecewise linear trajectories
        # Use K-means clustering to generate segments
        n_segments = max(3, int(len(avoided_path) / 100))
        kmeans = KMeans(n_clusters=n_segments).fit(avoided_path)
        centroids = kmeans.cluster_centers_
        
        # Sort by x-coordinate
        centroids = centroids[centroids[:, 0].argsort()]
        
        # Generate piecewise linear trajectory
        segmented_path = generate_waypoint_path(centroids, smoothness=0.3, num_points=len(avoided_path))
        
        # Partially adopt segmented path
        mask = np.random.random(len(avoided_path)) < 0.7
        avoided_path[mask] = segmented_path[mask]
    
    return avoided_path

def apply_efficiency_optimization(path, waypoints, efficiency=0.8):
    """Apply efficiency optimization to make path more direct (based on efficiency parameter)"""
    if len(waypoints) < 2:
        return path
    
    # Generate ideal path that directly connects all waypoints
    ideal_path = generate_waypoint_path(waypoints, smoothness=0.1, num_points=len(path))
    
    # Interpolate between actual path and ideal path based on efficiency parameter
    optimized_path = path.copy()
    for i in range(len(path)):
        optimized_path[i] = path[i] * (1 - efficiency) + ideal_path[i] * efficiency
    
    return optimized_path

def add_model_specific_features(path, model_type):
    """Add features specific to the deep learning model type"""
    modified_path = path.copy()
    
    if model_type == 'drl':
        # DRL models may have some exploratory oscillations
        noise = np.random.normal(0, 0.05, path.shape)
        modified_path += noise
    elif model_type == 'cnn':
        # CNN models may make more direct decisions in certain areas
        for i in range(1, len(modified_path)-1):
            if np.random.random() < 0.1:  # 10% chance
                modified_path[i] = (modified_path[i-1] + modified_path[i+1]) / 2
    elif model_type == 'lstm':
        # LSTM models may produce smoother trajectories
        window_size = 5
        for i in range(window_size, len(modified_path)-window_size):
            if np.random.random() < 0.3:  # 30% chance
                window = modified_path[i-window_size:i+window_size+1]
                modified_path[i] = np.mean(window, axis=0)
    elif model_type == 'hybrid':
        # Hybrid models may combine multiple characteristics
        # Apply DRL exploration
        noise = np.random.normal(0, 0.03, path.shape)
        modified_path += noise
        
        # Apply LSTM smoothing
        window_size = 3
        for i in range(window_size, len(modified_path)-window_size):
            if np.random.random() < 0.2:
                window = modified_path[i-window_size:i+window_size+1]
                modified_path[i] = np.mean(window, axis=0)
    
    return modified_path

def generate_dl_path():
    """Generate simulated deep learning path data"""
    # Create output directory
    dl_data_dir = os.path.join(args.data_dir, 'dl_simulated_paths')
    ensure_dir(dl_data_dir)
    
    # Extract waypoints and obstacles
    waypoints = extract_waypoints_from_world(args.world_file)
    obstacles = extract_obstacles_from_world(args.world_file)
    
    print(f"Extracted {len(waypoints)} waypoints and {len(obstacles)} obstacles")
    
    # Generate base trajectory
    base_path = generate_waypoint_path(waypoints, smoothness=args.smoothness)
    
    # Apply obstacle avoidance
    path_with_avoidance = apply_obstacle_avoidance(base_path, obstacles, args.obstacle_avoidance)
    
    # Apply efficiency optimization
    optimized_path = apply_efficiency_optimization(path_with_avoidance, waypoints, args.efficiency)
    
    # Add model-specific features
    final_path = add_model_specific_features(optimized_path, args.model_type)
    
    # Generate timestamps and velocity data
    # Assume drone moves at average 1m/s
    distances = np.zeros(len(final_path))
    for i in range(1, len(final_path)):
        distances[i] = np.linalg.norm(final_path[i] - final_path[i-1]) + distances[i-1]
    
    total_distance = distances[-1]
    timestamps = distances / total_distance * (total_distance)  # Total seconds equals total distance
    
    # Calculate velocities at each point (based on position changes)
    velocities = np.zeros((len(final_path), 3))
    for i in range(1, len(final_path)):
        dt = timestamps[i] - timestamps[i-1]
        if dt > 0:
            velocities[i] = (final_path[i] - final_path[i-1]) / dt
    
    # Calculate speed magnitude at each point
    speed = np.linalg.norm(velocities, axis=1)
    
    # Create DataFrame
    dl_data = pd.DataFrame({
        'time': timestamps,
        'x': final_path[:, 0],
        'y': final_path[:, 1],
        'z': final_path[:, 2],
        'velocity_x': velocities[:, 0],
        'velocity_y': velocities[:, 1],
        'velocity_z': velocities[:, 2],
        'speed': speed,
        'roll': np.zeros(len(final_path)),  # Simplified, assume drone stays level
        'pitch': np.zeros(len(final_path)),
        'yaw': np.zeros(len(final_path)),
        'state': ['NAVIGATE'] * len(final_path),
        'target_waypoint': [-1] * len(final_path),  # DL model may not use explicit waypoints
        'obstacle_avoiding': [0] * len(final_path)  # Initialize to 0
    })
    
    # Mark obstacle avoidance states
    for i in range(len(final_path)):
        for obstacle in obstacles:
            # Check if point is near obstacle
            pos = obstacle['position']
            size = obstacle['size']
            max_size = max(size)
            
            distance = np.linalg.norm(final_path[i] - np.array(pos))
            if distance < max_size * 2:  # If point is within extended obstacle range
                dl_data.loc[i, 'obstacle_avoiding'] = 1
                break
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(dl_data_dir, f"dl_{args.model_type}_path_{timestamp}.csv")
    dl_data.to_csv(output_file, index=False)
    print(f"Deep learning path data generated: {output_file}")
    
    # Visualize path
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='blue', marker='o', s=100, label='Waypoints')
    
    # Plot deep learning path
    ax.plot(final_path[:, 0], final_path[:, 1], final_path[:, 2], c='red', linewidth=2, label=f'{args.model_type.upper()} Path')
    
    # Plot obstacles
    for obstacle in obstacles:
        pos = obstacle['position']
        ax.scatter(pos[0], pos[1], pos[2], c='black', marker='s', s=50)
    
    # Mark start and end points
    ax.scatter(final_path[0, 0], final_path[0, 1], final_path[0, 2], c='green', marker='^', s=200, label='Start')
    ax.scatter(final_path[-1, 0], final_path[-1, 1], final_path[-1, 2], c='purple', marker='v', s=200, label='End')
    
    # Add labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Simulated {args.model_type.upper()} Deep Learning Path\nEfficiency: {args.efficiency}, Smoothness: {args.smoothness}, Avoidance: {args.obstacle_avoidance}')
    ax.legend()
    
    # Save figure
    fig_file = os.path.join(dl_data_dir, f"dl_{args.model_type}_path_{timestamp}.png")
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"Path visualization saved: {fig_file}")
    
    plt.close()
    
    return output_file

if __name__ == "__main__":
    generate_dl_path()
    print(f"Generation complete! Model type: {args.model_type}")
    print("Use analyze_waypoints.py script to compare this path with actual flight paths") 