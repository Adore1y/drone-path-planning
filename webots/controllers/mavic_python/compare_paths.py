#!/usr/bin/env python3

"""
Path Comparison Script
For comparing different path planning algorithms
Can compare actual flight paths with various algorithm paths
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import re
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev

# Command line argument parsing
parser = argparse.ArgumentParser(description='Compare multiple flight paths')
parser.add_argument('--files', nargs='+', help='List of data files to compare')
parser.add_argument('--output_dir', type=str, default='flight_data/comparison_results', help='Output directory for results')
parser.add_argument('--world_file', type=str, default='../../worlds/mixed_scenario.wbt', help='Path to world file')
parser.add_argument('--labels', nargs='+', help='Labels for each path')
parser.add_argument('--colors', nargs='+', help='Colors for each path')
parser.add_argument('--latest', action='store_true', help='Use the latest data files')
parser.add_argument('--all_algorithms', action='store_true', help='Include all available algorithm paths')
parser.add_argument('--algorithms', nargs='+', choices=['dqn', 'ppo', 'td3', 'gat-drl', 'a-star', 'rrt', 'potential-field'], 
                    help='Specific algorithms to include in comparison')
parser.add_argument('--top_view', action='store_true', help='Generate top view')
parser.add_argument('--metrics_only', action='store_true', help='Calculate metrics only, no images')
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

def find_latest_files():
    """Find the latest actual flight data and algorithm paths"""
    flight_files = []
    
    # Find latest actual flight data
    flight_data_pattern = 'flight_data/mavic_flight_data_*.csv'
    flight_data_files = glob.glob(flight_data_pattern)
    if flight_data_files:
        latest_flight_file = max(flight_data_files, key=os.path.getmtime)
        flight_files.append(latest_flight_file)
    
    # If needed, find latest algorithm paths
    if args.all_algorithms:
        # Find all algorithm types
        algorithm_types = ['dqn', 'ppo', 'td3', 'gat-drl', 'a-star', 'rrt', 'potential-field']
        for alg in algorithm_types:
            alg_pattern = f'flight_data/algorithm_paths/{alg}_path_*.csv'
            alg_files = glob.glob(alg_pattern)
            if alg_files:
                latest_alg_file = max(alg_files, key=os.path.getmtime)
                flight_files.append(latest_alg_file)
    elif args.algorithms:
        # Find specific algorithm types
        for alg in args.algorithms:
            alg_pattern = f'flight_data/algorithm_paths/{alg}_path_*.csv'
            alg_files = glob.glob(alg_pattern)
            if alg_files:
                latest_alg_file = max(alg_files, key=os.path.getmtime)
                flight_files.append(latest_alg_file)
    
    return flight_files

def load_flight_data(file_path):
    """Load flight data"""
    try:
        data = pd.read_csv(file_path)
        # Ensure key columns exist
        required_columns = ['x', 'y', 'z']
        for col in required_columns:
            if col not in data.columns:
                print(f"Error: Column '{col}' missing in file {file_path}")
                return None
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def generate_default_colors(num_paths):
    """Generate default color list"""
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 
                      'brown', 'pink', 'gray', 'olive', 'navy', 'teal', 'coral']
    return default_colors[:num_paths] if num_paths <= len(default_colors) else \
           [plt.cm.tab20(i) for i in range(num_paths)]

def generate_default_labels(file_paths):
    """Generate default labels based on filenames"""
    labels = []
    for file_path in file_paths:
        if 'mavic_flight_data_' in file_path:
            labels.append('Actual Flight')
        elif 'dqn_path' in file_path:
            labels.append('DQN')
        elif 'ppo_path' in file_path:
            labels.append('PPO')
        elif 'td3_path' in file_path:
            labels.append('TD3')
        elif 'gat-drl_path' in file_path:
            labels.append('GAT-DRL')
        elif 'a-star_path' in file_path:
            labels.append('A*')
        elif 'rrt_path' in file_path:
            labels.append('RRT')
        elif 'potential-field_path' in file_path:
            labels.append('Potential Field')
        else:
            # Extract label from filename
            basename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(basename)[0]
            labels.append(name_without_ext)
    
    return labels

def calculate_path_metrics(data, waypoints):
    """Calculate path metrics"""
    metrics = {}
    
    # Extract coordinates
    coords = data[['x', 'y', 'z']].values
    
    # 1. Total path length
    path_length = 0
    for i in range(1, len(coords)):
        path_length += np.linalg.norm(coords[i] - coords[i-1])
    metrics['path_length'] = path_length
    
    # 2. Average speed
    if 'speed' in data.columns:
        metrics['avg_speed'] = data['speed'].mean()
        metrics['max_speed'] = data['speed'].max()
    
    # 3. Path smoothness (based on variance of direction changes)
    smoothness = 0
    if len(coords) > 2:
        # Calculate direction changes between consecutive points
        directions = []
        for i in range(1, len(coords)-1):
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                v1_normalized = v1 / np.linalg.norm(v1)
                v2_normalized = v2 / np.linalg.norm(v2)
                dot_product = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
                angle = np.arccos(dot_product)
                directions.append(angle)
        
        if directions:
            # Lower standard deviation of direction changes means smoother path
            smoothness = 1.0 / (1.0 + np.std(directions))
    
    metrics['smoothness'] = smoothness
    
    # 4. Average deviation from ideal path
    # For each point, find the closest segment and calculate distance
    if len(waypoints) >= 2:
        deviations = []
        
        # For each point in the path
        for point in coords:
            min_dist = float('inf')
            
            # For each segment formed by waypoint pairs
            for i in range(len(waypoints) - 1):
                p1 = waypoints[i]
                p2 = waypoints[i+1]
                
                # Calculate distance to segment
                line_vec = p2 - p1
                point_vec = point - p1
                line_len = np.linalg.norm(line_vec)
                
                if line_len > 0:
                    line_unitvec = line_vec / line_len
                    point_vec_scaled = point_vec / line_len
                    
                    # Calculate projection length
                    t = max(0, min(1, np.dot(point_vec_scaled, line_unitvec)))
                    
                    # Calculate projection point
                    projection = p1 + t * line_vec
                    
                    # Calculate distance
                    dist = np.linalg.norm(point - projection)
                    min_dist = min(min_dist, dist)
            
            deviations.append(min_dist)
        
        metrics['avg_deviation'] = np.mean(deviations)
        metrics['max_deviation'] = np.max(deviations)
    
    # 5. Obstacle avoidance count
    if 'obstacle_avoiding' in data.columns:
        # Count changes from 0 to 1
        avoiding_changes = data['obstacle_avoiding'].diff() > 0
        metrics['obstacle_avoidance_count'] = avoiding_changes.sum()
    
    # 6. Path efficiency (ratio of direct distance to actual path length)
    if len(coords) > 1:
        start_point = coords[0]
        end_point = coords[-1]
        direct_distance = np.linalg.norm(end_point - start_point)
        if path_length > 0:
            metrics['path_efficiency'] = direct_distance / path_length
    
    return metrics

def plot_3d_paths(file_paths, data_list, labels, colors, waypoints, obstacles, output_dir):
    """Plot 3D comparison of multiple paths"""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each path
    for i, (file_path, data, label, color) in enumerate(zip(file_paths, data_list, labels, colors)):
        x = data['x'].values
        y = data['y'].values
        z = data['z'].values
        
        # Plot path
        ax.plot(x, y, z, color=color, linewidth=2, label=label)
        
        # Mark start and end points
        if i == 0:  # Only for the first path to avoid cluttering
            ax.scatter(x[0], y[0], z[0], c='green', marker='^', s=100, label='Start')
            ax.scatter(x[-1], y[-1], z[-1], c='red', marker='v', s=100, label='End')
    
    # Plot waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='blue', marker='o', s=80, label='Waypoints')
    
    # Number the waypoints
    for i, (wp_x, wp_y, wp_z) in enumerate(waypoints):
        ax.text(wp_x, wp_y, wp_z, f'{i+1}', fontsize=10, color='darkblue')
    
    # Plot obstacles
    for obstacle in obstacles:
        pos = obstacle['position']
        size = obstacle['size']
        
        # Plot obstacle as a point
        ax.scatter(pos[0], pos[1], pos[2], c='black', marker='s', s=30)
        
        # Draw wireframe box for obstacle
        x, y, z = pos
        dx, dy, dz = size[0]/2, size[1]/2, size[2]/2
        
        # Define the vertices of the cube
        vertices = [
            [x-dx, y-dy, z-dz], [x+dx, y-dy, z-dz], [x+dx, y+dy, z-dz], [x-dx, y+dy, z-dz],
            [x-dx, y-dy, z+dz], [x+dx, y-dy, z+dz], [x+dx, y+dy, z+dz], [x-dx, y+dy, z+dz]
        ]
        
        # Define the edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
        ]
        
        # Plot the edges
        for edge in edges:
            ax.plot3D(
                [vertices[edge[0]][0], vertices[edge[1]][0]],
                [vertices[edge[0]][1], vertices[edge[1]][1]],
                [vertices[edge[0]][2], vertices[edge[1]][2]],
                'k-', alpha=0.3
            )
    
    # Configure axes
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Add title and legend
    plt.title('Path Comparison')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        ensure_dir(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"path_comparison_3d_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"3D path comparison saved to: {filepath}")
    
    return fig

def plot_top_view(file_paths, data_list, labels, colors, waypoints, obstacles, output_dir):
    """Plot top-down view comparison of multiple paths"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot each path
    for file_path, data, label, color in zip(file_paths, data_list, labels, colors):
        x = data['x'].values
        y = data['y'].values
        
        # Plot path
        ax.plot(x, y, color=color, linewidth=2, label=label)
    
    # Plot waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], c='blue', marker='o', s=80, label='Waypoints')
    
    # Number the waypoints
    for i, (wp_x, wp_y, _) in enumerate(waypoints):
        ax.text(wp_x, wp_y, f'{i+1}', fontsize=10, color='darkblue')
    
    # Plot obstacles
    for obstacle in obstacles:
        pos = obstacle['position']
        size = obstacle['size']
        
        # Draw rectangle for obstacle footprint
        rect = plt.Rectangle((pos[0]-size[0]/2, pos[1]-size[1]/2), size[0], size[1], 
                           fill=True, color='black', alpha=0.3)
        ax.add_patch(rect)
    
    # Configure axes
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Add title and legend
    plt.title('Top View Path Comparison')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        ensure_dir(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"path_comparison_top_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Top view comparison saved to: {filepath}")
    
    return fig

def generate_metrics_comparison(data_list, labels, metrics_list, output_dir):
    """Generate metrics comparison table and charts"""
    # Create metrics dataframe
    metrics_df = pd.DataFrame(metrics_list, index=labels)
    
    # Save metrics to CSV
    output_file = os.path.join(output_dir, f"path_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    metrics_df.to_csv(output_file)
    print(f"Path metrics saved to: {output_file}")
    
    if args.metrics_only:
        return
    
    # Plot metrics comparison bar charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Path length comparison
    ax = axes[0, 0]
    metrics_df['path_length'].plot(kind='bar', ax=ax, color=[plt.cm.tab10(i) for i in range(len(labels))])
    ax.set_title('Path Length Comparison', fontsize=12)
    ax.set_ylabel('Length (m)', fontsize=10)
    ax.grid(axis='y')
    
    # Average deviation comparison
    ax = axes[0, 1]
    if 'avg_deviation' in metrics_df.columns:
        metrics_df['avg_deviation'].plot(kind='bar', ax=ax, color=[plt.cm.tab10(i) for i in range(len(labels))])
        ax.set_title('Average Deviation Comparison', fontsize=12)
        ax.set_ylabel('Deviation (m)', fontsize=10)
        ax.grid(axis='y')
    
    # Smoothness comparison
    ax = axes[1, 0]
    metrics_df['smoothness'].plot(kind='bar', ax=ax, color=[plt.cm.tab10(i) for i in range(len(labels))])
    ax.set_title('Path Smoothness Comparison', fontsize=12)
    ax.set_ylabel('Smoothness Index', fontsize=10)
    ax.grid(axis='y')
    
    # Path efficiency comparison
    ax = axes[1, 1]
    if 'path_efficiency' in metrics_df.columns:
        metrics_df['path_efficiency'].plot(kind='bar', ax=ax, color=[plt.cm.tab10(i) for i in range(len(labels))])
        ax.set_title('Path Efficiency Comparison', fontsize=12)
        ax.set_ylabel('Efficiency (0-1)', fontsize=10)
        ax.grid(axis='y')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(output_dir, f"metrics_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to: {output_file}")
    
    plt.close(fig)

def main():
    # Ensure output directory exists
    output_dir = args.output_dir
    ensure_dir(output_dir)
    
    # Determine which files to use
    file_paths = []
    if args.files:
        file_paths = args.files
    elif args.latest or args.all_algorithms or args.algorithms:
        file_paths = find_latest_files()
    else:
        print("No files specified. Use --files, --latest, --all_algorithms, or --algorithms option.")
        return
    
    if not file_paths:
        print("No matching files found.")
        return
    
    print(f"Comparing {len(file_paths)} files:")
    for file_path in file_paths:
        print(f"  - {file_path}")
    
    # Load data from each file
    data_list = []
    valid_file_paths = []
    for file_path in file_paths:
        data = load_flight_data(file_path)
        if data is not None:
            data_list.append(data)
            valid_file_paths.append(file_path)
    
    if not data_list:
        print("No valid data files found.")
        return
    
    # Extract waypoints and obstacles
    waypoints = extract_waypoints_from_world(args.world_file)
    print(f"Extracted {len(waypoints)} waypoints from world file")
    
    obstacles = extract_obstacles_from_world(args.world_file)
    print(f"Extracted {len(obstacles)} obstacles from world file")
    
    # Generate labels and colors if not provided
    labels = args.labels if args.labels else generate_default_labels(valid_file_paths)
    colors = args.colors if args.colors else generate_default_colors(len(valid_file_paths))
    
    # Calculate metrics for each path
    metrics_list = []
    for data in data_list:
        metrics = calculate_path_metrics(data, waypoints)
        metrics_list.append(metrics)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_list, index=labels)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(output_dir, f"path_metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_file)
    print(f"Path metrics saved to: {metrics_file}")
    
    # Generate plots
    if not args.metrics_only:
        fig_3d = plot_3d_paths(valid_file_paths, data_list, labels, colors, waypoints, obstacles, output_dir)
        
        if args.top_view:
            fig_top = plot_top_view(valid_file_paths, data_list, labels, colors, waypoints, obstacles, output_dir)
    
    # Generate metrics comparison
    fig_metrics = generate_metrics_comparison(data_list, labels, metrics_list, output_dir)
    
    print("Path comparison complete!")

if __name__ == "__main__":
    main() 