#!/usr/bin/env python3

"""
Flight Data Analysis Tool for Waypoint Navigation
Analyzes flight data from waypoint navigation controller and generates visualizations
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import argparse
from datetime import datetime

# Command line arguments
parser = argparse.ArgumentParser(description='Analyze waypoint navigation flight data')
parser.add_argument('--data_dir', type=str, default='flight_data', help='Data directory')
parser.add_argument('--world_file', type=str, default='../../worlds/mixed_scenario.wbt', help='World file path')
parser.add_argument('--latest', action='store_true', help='Use latest flight data file')
parser.add_argument('--file', type=str, help='Specific flight data file to analyze')
parser.add_argument('--output_dir', type=str, default='analysis_results', help='Output directory for analysis results')
parser.add_argument('--save_only', action='store_true', help='Save figures without displaying')
args = parser.parse_args()

def ensure_dir(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_latest_data(data_dir, pattern="mavic_flight_data_*.csv"):
    """Find latest data file in directory"""
    try:
        files = glob.glob(os.path.join(data_dir, pattern))
        if not files:
            print(f"No matching files found in {data_dir}")
            return None
        
        # Sort by modification time (most recent first)
        latest_file = max(files, key=os.path.getmtime)
        print(f"Found latest file: {latest_file}")
        return latest_file
    
    except Exception as e:
        print(f"Error finding latest data: {e}")
        return None

def load_flight_data(file_path):
    """Load flight data from CSV file"""
    try:
        print(f"Loading flight data from: {file_path}")
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. {len(data)} data points.")
        return data
    
    except Exception as e:
        print(f"Error loading flight data: {e}")
        return None

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

def plot_3d_path(flight_data, waypoints, obstacles=None, output_dir=None, filename_prefix='flight_path_3d'):
    """Plot 3D flight path and waypoints"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract flight data
    x = flight_data['x'].values
    y = flight_data['y'].values
    z = flight_data['z'].values
    
    # Color the path by state
    states = flight_data['state'].values
    state_colors = {
        'TAKEOFF': 'orange',
        'NAVIGATE': 'blue',
        'HOVER': 'green',
        'LAND': 'purple',
        'EMERGENCY': 'red',
        'FINISHED': 'black'
    }
    
    # Plot path segments by state
    prev_state = states[0]
    segment_start = 0
    
    for i in range(1, len(states)):
        if states[i] != prev_state or i == len(states) - 1:
            color = state_colors.get(prev_state, 'gray')
            ax.plot(x[segment_start:i], y[segment_start:i], z[segment_start:i], 
                   color=color, linewidth=2, label=f'State: {prev_state}' if prev_state not in [s for s in flight_data['state'].values[:segment_start]] else None)
            segment_start = i
            prev_state = states[i]
    
    # Plot waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='red', marker='o', s=100, label='Waypoints')
    
    # Number the waypoints
    for i, (wp_x, wp_y, wp_z) in enumerate(waypoints):
        ax.text(wp_x, wp_y, wp_z, f'{i+1}', fontsize=12, color='darkred')
    
    # Plot the ideal path through waypoints
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'r--', linewidth=1, label='Ideal Path')
    
    # Plot obstacles if available
    if obstacles:
        for i, obstacle in enumerate(obstacles):
            pos = obstacle['position']
            ax.scatter(pos[0], pos[1], pos[2], c='black', marker='s', s=50, label='Obstacle' if i == 0 else None)
    
    # Mark takeoff and landing points
    ax.scatter(x[0], y[0], z[0], c='green', marker='^', s=200, label='Takeoff')
    ax.scatter(x[-1], y[-1], z[-1], c='red', marker='v', s=200, label='Landing')
    
    # Configure axes
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add title and legend
    plt.title(f'Drone Flight Path and Waypoints\nTotal waypoints: {len(waypoints)}\nTimestamp: {timestamp}')
    
    # Handle legend - combine same states
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    # Save figure if output directory is specified
    if output_dir:
        ensure_dir(output_dir)
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp_file}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"3D path plot saved: {filepath}")
    
    return fig

def plot_top_view(flight_data, waypoints, obstacles=None, output_dir=None, filename_prefix='flight_path_top'):
    """Plot top-down view of flight path"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract flight data
    x = flight_data['x'].values
    y = flight_data['y'].values
    z = flight_data['z'].values
    
    # Color the path by z-height
    points = ax.scatter(x, y, c=z, cmap='viridis', s=15, label='Flight Path')
    cbar = plt.colorbar(points)
    cbar.set_label('Height (m)')
    
    # Plot waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], c='red', marker='o', s=100, label='Waypoints')
    
    # Number the waypoints
    for i, (wp_x, wp_y, wp_z) in enumerate(waypoints):
        ax.text(wp_x, wp_y, f'{i+1}', fontsize=12, color='darkred')
    
    # Plot the ideal path through waypoints
    ax.plot(waypoints[:, 0], waypoints[:, 1], 'r--', linewidth=1, label='Ideal Path')
    
    # Plot obstacles if available
    if obstacles:
        for i, obstacle in enumerate(obstacles):
            pos = obstacle['position']
            size = obstacle['size']
            # Draw rectangle for obstacle footprint
            rect = plt.Rectangle((pos[0]-size[0]/2, pos[1]-size[1]/2), size[0], size[1], 
                               fill=True, color='black', alpha=0.5, label='Obstacle' if i == 0 else None)
            ax.add_patch(rect)
    
    # Mark takeoff and landing points
    ax.scatter(x[0], y[0], c='green', marker='^', s=200, label='Takeoff')
    ax.scatter(x[-1], y[-1], c='red', marker='v', s=200, label='Landing')
    
    # Configure axes
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Add title and legend
    plt.title('Top View of Drone Flight Path')
    
    # Handle legend - combine same labels
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    # Save figure if output directory is specified
    if output_dir:
        ensure_dir(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Top view plot saved: {filepath}")
    
    return fig

def plot_altitude_profile(flight_data, waypoints, output_dir=None, filename_prefix='altitude_profile'):
    """Plot altitude profile over time"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract flight data
    time = flight_data['time'].values
    z = flight_data['z'].values
    states = flight_data['state'].values
    
    # Plot altitude vs time
    ax.plot(time, z, 'b-', linewidth=2, label='Drone Altitude')
    
    # Highlight different flight states
    state_colors = {
        'TAKEOFF': 'orange',
        'NAVIGATE': 'blue',
        'HOVER': 'green',
        'LAND': 'purple',
        'EMERGENCY': 'red',
        'FINISHED': 'black'
    }
    
    # Create background color regions for different states
    prev_state = states[0]
    segment_start = 0
    
    for i in range(1, len(states)):
        if states[i] != prev_state or i == len(states) - 1:
            end_idx = i if states[i] != prev_state else i + 1
            color = state_colors.get(prev_state, 'gray')
            ax.axvspan(time[segment_start], time[end_idx-1], alpha=0.2, color=color, 
                      label=f'State: {prev_state}' if prev_state not in [s for s in flight_data['state'].values[:segment_start]] else None)
            segment_start = i
            prev_state = states[i]
    
    # Plot waypoint altitudes
    target_waypoint = flight_data['target_waypoint'].values
    for i, wp_z in enumerate(waypoints[:, 2]):
        # Find where this waypoint was targeted
        wp_indices = np.where(target_waypoint == i)[0]
        if len(wp_indices) > 0:
            start_idx = wp_indices[0]
            end_idx = wp_indices[-1]
            ax.axhline(y=wp_z, xmin=time[start_idx]/time[-1], xmax=time[end_idx]/time[-1], 
                      color='red', linestyle='--', alpha=0.7, 
                      label=f'Waypoint {i+1} Altitude' if i == 0 else None)
    
    # Configure axes
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.grid(True)
    
    # Add title and legend
    plt.title('Drone Altitude Profile')
    
    # Handle legend - combine same labels
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    # Save figure if output directory is specified
    if output_dir:
        ensure_dir(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Altitude profile saved: {filepath}")
    
    return fig

def plot_velocity_profile(flight_data, output_dir=None, filename_prefix='velocity_profile'):
    """Plot velocity profile over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Extract flight data
    time = flight_data['time'].values
    vx = flight_data['velocity_x'].values
    vy = flight_data['velocity_y'].values
    vz = flight_data['velocity_z'].values
    speed = flight_data['speed'].values
    states = flight_data['state'].values
    
    # Plot velocity components
    ax1.plot(time, vx, 'r-', linewidth=1, label='X-Velocity')
    ax1.plot(time, vy, 'g-', linewidth=1, label='Y-Velocity')
    ax1.plot(time, vz, 'b-', linewidth=1, label='Z-Velocity')
    
    # Plot total speed
    ax2.plot(time, speed, 'k-', linewidth=2, label='Total Speed')
    
    # Highlight different flight states for both plots
    state_colors = {
        'TAKEOFF': 'orange',
        'NAVIGATE': 'blue',
        'HOVER': 'green',
        'LAND': 'purple',
        'EMERGENCY': 'red',
        'FINISHED': 'black'
    }
    
    # Create background color regions for different states
    prev_state = states[0]
    segment_start = 0
    
    for i in range(1, len(states)):
        if states[i] != prev_state or i == len(states) - 1:
            end_idx = i if states[i] != prev_state else i + 1
            color = state_colors.get(prev_state, 'gray')
            
            # Add to both plots
            ax1.axvspan(time[segment_start], time[end_idx-1], alpha=0.2, color=color, 
                       label=f'State: {prev_state}' if prev_state not in [s for s in flight_data['state'].values[:segment_start]] else None)
            ax2.axvspan(time[segment_start], time[end_idx-1], alpha=0.2, color=color)
            
            segment_start = i
            prev_state = states[i]
    
    # Configure axes
    ax1.set_ylabel('Velocity (m/s)')
    ax1.grid(True)
    ax1.set_title('Drone Velocity Components')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (m/s)')
    ax2.grid(True)
    ax2.set_title('Drone Total Speed')
    
    # Handle legend - combine same labels for ax1
    handles, labels = ax1.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax1.legend(*zip(*unique))
    
    # Legend for ax2
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure if output directory is specified
    if output_dir:
        ensure_dir(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Velocity profile saved: {filepath}")
    
    return fig

def calculate_performance_metrics(flight_data, waypoints):
    """Calculate performance metrics for the flight"""
    metrics = {}
    
    # Total flight time
    metrics['total_flight_time'] = flight_data['time'].max()
    
    # Average speed
    metrics['average_speed'] = flight_data['speed'].mean()
    metrics['max_speed'] = flight_data['speed'].max()
    
    # Time spent in each state
    state_times = {}
    states = flight_data['state'].values
    time = flight_data['time'].values
    
    prev_state = states[0]
    state_start_time = time[0]
    
    for i in range(1, len(states)):
        if states[i] != prev_state or i == len(states) - 1:
            end_time = time[i]
            state_duration = end_time - state_start_time
            
            if prev_state in state_times:
                state_times[prev_state] += state_duration
            else:
                state_times[prev_state] = state_duration
            
            state_start_time = end_time
            prev_state = states[i]
    
    metrics['state_times'] = state_times
    
    # Navigation efficiency (straight line distance / actual distance traveled)
    x = flight_data['x'].values
    y = flight_data['y'].values
    z = flight_data['z'].values
    
    # Calculate actual distance traveled
    actual_distance = 0
    for i in range(1, len(x)):
        segment_distance = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2 + (z[i] - z[i-1])**2)
        actual_distance += segment_distance
    
    # Calculate straight line distance between waypoints
    straight_distance = 0
    for i in range(1, len(waypoints)):
        wp_distance = np.sqrt((waypoints[i][0] - waypoints[i-1][0])**2 + 
                              (waypoints[i][1] - waypoints[i-1][1])**2 + 
                              (waypoints[i][2] - waypoints[i-1][2])**2)
        straight_distance += wp_distance
    
    metrics['actual_distance'] = actual_distance
    metrics['straight_distance'] = straight_distance
    
    if actual_distance > 0:
        metrics['path_efficiency'] = straight_distance / actual_distance
    else:
        metrics['path_efficiency'] = 0
    
    # Waypoint accuracy
    # For each waypoint, find the minimum distance the drone got to it
    waypoint_accuracy = []
    for waypoint in waypoints:
        distances = np.sqrt((x - waypoint[0])**2 + (y - waypoint[1])**2 + (z - waypoint[2])**2)
        min_distance = np.min(distances)
        waypoint_accuracy.append(min_distance)
    
    metrics['waypoint_accuracy'] = waypoint_accuracy
    metrics['average_waypoint_accuracy'] = np.mean(waypoint_accuracy)
    metrics['max_waypoint_error'] = np.max(waypoint_accuracy)
    
    # Altitude stability (standard deviation of altitude error during NAVIGATE state)
    navigate_indices = flight_data['state'] == 'NAVIGATE'
    if np.any(navigate_indices):
        navigate_data = flight_data[navigate_indices]
        target_waypoint_indices = navigate_data['target_waypoint'].values
        
        altitude_errors = []
        for i, row in navigate_data.iterrows():
            if row['target_waypoint'] >= 0 and row['target_waypoint'] < len(waypoints):
                target_altitude = waypoints[int(row['target_waypoint'])][2]
                altitude_error = abs(row['z'] - target_altitude)
                altitude_errors.append(altitude_error)
        
        if altitude_errors:
            metrics['altitude_stability'] = np.std(altitude_errors)
            metrics['average_altitude_error'] = np.mean(altitude_errors)
            metrics['max_altitude_error'] = np.max(altitude_errors)
        else:
            metrics['altitude_stability'] = None
            metrics['average_altitude_error'] = None
            metrics['max_altitude_error'] = None
    else:
        metrics['altitude_stability'] = None
        metrics['average_altitude_error'] = None
        metrics['max_altitude_error'] = None
    
    # Time between waypoints
    target_waypoint = flight_data['target_waypoint'].values
    waypoint_times = {}
    
    for i in range(len(waypoints)):
        # Find first and last time this waypoint was targeted
        wp_indices = np.where(target_waypoint == i)[0]
        if len(wp_indices) > 0:
            start_idx = wp_indices[0]
            end_idx = wp_indices[-1]
            waypoint_times[i] = time[end_idx] - time[start_idx]
    
    metrics['waypoint_times'] = waypoint_times
    
    return metrics

def display_performance_metrics(metrics, output_dir=None, filename_prefix='performance_metrics'):
    """Display and save performance metrics"""
    # Create a figure to display metrics
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Format the metrics for display
    metrics_text = "Flight Performance Metrics:\n\n"
    metrics_text += f"Total Flight Time: {metrics['total_flight_time']:.2f} seconds\n"
    metrics_text += f"Total Distance Traveled: {metrics['actual_distance']:.2f} meters\n"
    metrics_text += f"Straight-Line Distance: {metrics['straight_distance']:.2f} meters\n"
    metrics_text += f"Path Efficiency: {metrics['path_efficiency']:.2%}\n\n"
    
    metrics_text += f"Average Speed: {metrics['average_speed']:.2f} m/s\n"
    metrics_text += f"Maximum Speed: {metrics['max_speed']:.2f} m/s\n\n"
    
    metrics_text += "Time Spent in Each State:\n"
    for state, time in metrics['state_times'].items():
        metrics_text += f"  - {state}: {time:.2f} seconds ({time/metrics['total_flight_time']:.1%})\n"
    
    metrics_text += "\nWaypoint Accuracy:\n"
    metrics_text += f"  - Average closest approach: {metrics['average_waypoint_accuracy']:.2f} meters\n"
    metrics_text += f"  - Maximum error: {metrics['max_waypoint_error']:.2f} meters\n"
    
    for i, accuracy in enumerate(metrics['waypoint_accuracy']):
        metrics_text += f"  - Waypoint {i+1}: {accuracy:.2f} meters\n"
    
    if metrics['altitude_stability'] is not None:
        metrics_text += f"\nAltitude Stability: {metrics['altitude_stability']:.2f} meters (std dev)\n"
        metrics_text += f"Average Altitude Error: {metrics['average_altitude_error']:.2f} meters\n"
        metrics_text += f"Maximum Altitude Error: {metrics['max_altitude_error']:.2f} meters\n"
    
    metrics_text += "\nTime at Each Waypoint:\n"
    for wp, wp_time in metrics['waypoint_times'].items():
        metrics_text += f"  - Waypoint {wp+1}: {wp_time:.2f} seconds\n"
    
    # Display metrics
    ax.text(0.05, 0.95, metrics_text, va='top', ha='left', fontsize=12, family='monospace')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.05, 0.02, f"Generated: {timestamp}", fontsize=10)
    
    plt.tight_layout()
    
    # Save metrics if output directory is specified
    if output_dir:
        ensure_dir(output_dir)
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics as PNG
        filename_png = f"{filename_prefix}_{timestamp_file}.png"
        filepath_png = os.path.join(output_dir, filename_png)
        plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
        print(f"Performance metrics image saved: {filepath_png}")
        
        # Also save metrics as text file
        filename_txt = f"{filename_prefix}_{timestamp_file}.txt"
        filepath_txt = os.path.join(output_dir, filename_txt)
        with open(filepath_txt, 'w') as f:
            f.write(metrics_text)
        print(f"Performance metrics text saved: {filepath_txt}")
    
    return fig

def main():
    # Set up output directory
    output_dir = args.output_dir
    ensure_dir(output_dir)
    
    # Determine which data file to use
    data_file = None
    if args.file:
        data_file = args.file
    elif args.latest:
        data_file = find_latest_data(args.data_dir)
    else:
        data_file = find_latest_data(args.data_dir)
    
    if not data_file:
        print("No flight data file found. Exiting.")
        return
    
    # Load flight data
    flight_data = load_flight_data(data_file)
    if flight_data is None:
        print("Failed to load flight data. Exiting.")
        return
    
    # Extract waypoints
    waypoints = extract_waypoints_from_world(args.world_file)
    print(f"Extracted {len(waypoints)} waypoints from world file")
    
    # Extract obstacles
    obstacles = extract_obstacles_from_world(args.world_file)
    print(f"Extracted {len(obstacles)} obstacles from world file")
    
    # Generate plots
    fig_3d = plot_3d_path(flight_data, waypoints, obstacles, output_dir)
    fig_top = plot_top_view(flight_data, waypoints, obstacles, output_dir)
    fig_altitude = plot_altitude_profile(flight_data, waypoints, output_dir)
    fig_velocity = plot_velocity_profile(flight_data, output_dir)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(flight_data, waypoints)
    fig_metrics = display_performance_metrics(metrics, output_dir)
    
    # Show plots if not save_only
    if not args.save_only:
        plt.show()
    else:
        plt.close('all')
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 