#!/usr/bin/env python3

"""
Path Comparison Script
For comparing different path planning algorithms
Can compare actual flight paths with simulated deep learning paths
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
parser.add_argument('--dl_latest', action='store_true', help='Include the latest deep learning simulated data')
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
    """Find the latest actual flight data and deep learning simulated data"""
    flight_files = []
    
    # Find latest actual flight data
    flight_data_pattern = 'flight_data/flight_data_*.csv'
    flight_data_files = glob.glob(flight_data_pattern)
    if flight_data_files:
        latest_flight_file = max(flight_data_files, key=os.path.getmtime)
        flight_files.append(latest_flight_file)
    
    # If needed, find latest deep learning simulated data
    if args.dl_latest:
        dl_data_pattern = 'flight_data/dl_simulated_paths/dl_*_path_*.csv'
        dl_data_files = glob.glob(dl_data_pattern)
        if dl_data_files:
            latest_dl_file = max(dl_data_files, key=os.path.getmtime)
            flight_files.append(latest_dl_file)
    
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
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    return default_colors[:num_paths] if num_paths <= len(default_colors) else \
           [plt.cm.tab10(i) for i in range(num_paths)]

def generate_default_labels(file_paths):
    """Generate default labels based on filenames"""
    labels = []
    for file_path in file_paths:
        if 'flight_data_' in file_path:
            labels.append('Actual Flight')
        elif 'dl_drl_path' in file_path:
            labels.append('DRL Model')
        elif 'dl_cnn_path' in file_path:
            labels.append('CNN Model')
        elif 'dl_lstm_path' in file_path:
            labels.append('LSTM Model')
        elif 'dl_hybrid_path' in file_path:
            labels.append('Hybrid Model')
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
    """Plot 3D path diagram"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each path
    for i, (data, label, color) in enumerate(zip(data_list, labels, colors)):
        coords = data[['x', 'y', 'z']].values
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=2, label=label)
        
        # Mark start and end points
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, marker='^', s=100)
        ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color=color, marker='v', s=100)
    
    # Plot waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='blue', marker='o', s=80, label='Waypoints')
    
    # Plot ideal path
    if len(waypoints) >= 2:
        # Use spline interpolation for smooth ideal path
        tck, u = splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0, k=min(3, len(waypoints)-1))
        u_new = np.linspace(0, 1, 100)
        ideal_x, ideal_y, ideal_z = splev(u_new, tck)
        ax.plot(ideal_x, ideal_y, ideal_z, 'k--', linewidth=1.5, label='Ideal Path')
    
    # Plot obstacles
    for obstacle in obstacles:
        pos = obstacle['position']
        ax.scatter(pos[0], pos[1], pos[2], c='black', marker='s', s=50)
    
    # Set title and labels
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f'Path Comparison ({timestamp})', fontsize=14)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.legend(fontsize=10)
    
    # Save the figure
    output_file = os.path.join(output_dir, f"path_comparison_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"3D path comparison saved to: {output_file}")
    
    plt.close(fig)

def plot_top_view(file_paths, data_list, labels, colors, waypoints, obstacles, output_dir):
    """Plot top view (X-Y plane)"""
    if not args.top_view:
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each path
    for data, label, color in zip(data_list, labels, colors):
        coords = data[['x', 'y']].values
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2, label=label)
        
        # Mark start and end points
        ax.scatter(coords[0, 0], coords[0, 1], color=color, marker='^', s=100)
        ax.scatter(coords[-1, 0], coords[-1, 1], color=color, marker='v', s=100)
    
    # Plot waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], c='blue', marker='o', s=80, label='Waypoints')
    
    # Plot ideal path
    if len(waypoints) >= 2:
        # Use spline interpolation for smooth ideal path
        tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=0, k=min(3, len(waypoints)-1))
        u_new = np.linspace(0, 1, 100)
        ideal_x, ideal_y = splev(u_new, tck)
        ax.plot(ideal_x, ideal_y, 'k--', linewidth=1.5, label='Ideal Path')
    
    # Plot obstacles (top view)
    for obstacle in obstacles:
        pos = obstacle['position']
        size = obstacle['size']
        left = pos[0] - size[0]/2
        bottom = pos[1] - size[1]/2
        rect = plt.Rectangle((left, bottom), size[0], size[1], color='gray', alpha=0.5)
        ax.add_patch(rect)
    
    # Set title and labels
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f'Path Comparison - Top View ({timestamp})', fontsize=14)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=10)
    
    # Save the figure
    output_file = os.path.join(output_dir, f"path_comparison_top_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Top view comparison saved to: {output_file}")
    
    plt.close(fig)

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
    ensure_dir(args.output_dir)
    
    # Determine files to compare
    file_paths = []
    if args.files:
        file_paths = args.files
    elif args.latest:
        file_paths = find_latest_files()
    
    if not file_paths:
        print("Error: No comparison files specified. Use --files or --latest argument")
        return
    
    print(f"Comparing {len(file_paths)} files:")
    for path in file_paths:
        print(f"  - {path}")
    
    # Load data
    data_list = []
    valid_file_paths = []
    for file_path in file_paths:
        data = load_flight_data(file_path)
        if data is not None:
            data_list.append(data)
            valid_file_paths.append(file_path)
    
    if not data_list:
        print("Error: Could not load any data files")
        return
    
    # Prepare labels and colors
    labels = args.labels if args.labels else generate_default_labels(valid_file_paths)
    colors = args.colors if args.colors else generate_default_colors(len(data_list))
    
    # Ensure label and color counts match
    if len(labels) < len(data_list):
        labels.extend([f'Path {i+1}' for i in range(len(labels), len(data_list))])
    if len(colors) < len(data_list):
        colors.extend(generate_default_colors(len(data_list) - len(colors)))
    
    # Extract waypoints and obstacles
    waypoints = extract_waypoints_from_world(args.world_file)
    obstacles = extract_obstacles_from_world(args.world_file)
    
    print(f"Extracted {len(waypoints)} waypoints and {len(obstacles)} obstacles")
    
    # Calculate metrics for each path
    metrics_list = []
    for data in data_list:
        metrics = calculate_path_metrics(data, waypoints)
        metrics_list.append(metrics)
    
    # Plot 3D path comparison
    if not args.metrics_only:
        plot_3d_paths(valid_file_paths, data_list, labels, colors, waypoints, obstacles, args.output_dir)
        
        # Plot top view
        if args.top_view:
            plot_top_view(valid_file_paths, data_list, labels, colors, waypoints, obstacles, args.output_dir)
    
    # Generate metrics comparison
    generate_metrics_comparison(data_list, labels, metrics_list, args.output_dir)
    
    print("Path comparison complete!")

if __name__ == "__main__":
    main() 