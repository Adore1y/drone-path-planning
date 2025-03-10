#!/usr/bin/env python3

"""
Webots Drone Environment Wrapper for Reinforcement Learning
Provides a standardized gym-like interface for deep reinforcement learning algorithms
"""

import numpy as np
import os
import time
import gym
from gym import spaces
import threading
import csv
from datetime import datetime
import subprocess
import signal
import sys
import json
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Union

# Conditional import to handle running with or without Webots
try:
    from controller import Robot, Supervisor, Node
    WEBOTS_AVAILABLE = True
    print("Successfully imported Webots controller module.")
except ImportError:
    WEBOTS_AVAILABLE = False
    print("Warning: Webots controller module not found. Running in simulation-only mode.")


class WebotsMAVDroneEnv(gym.Env):
    """
    Webots Mavic Drone Environment for Path Planning RL
    
    This environment wrapper provides a gym-like interface for the Webots drone simulation,
    allowing deep reinforcement learning algorithms to be trained for waypoint navigation
    and obstacle avoidance tasks.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                config_file=None, 
                headless=False, 
                real_time_factor=1.0, 
                max_episode_steps=1000,
                time_step=32,  # Webots timestep in ms
                reward_type='dense',
                sim_seed=None):
        """
        Initialize the Webots MAV Drone Environment
        
        Args:
            config_file: Path to the configuration file
            headless: Run Webots in headless mode
            real_time_factor: Speed factor for simulation
            max_episode_steps: Maximum number of steps per episode
            time_step: Webots time step in milliseconds
            reward_type: Type of reward function to use
            sim_seed: Random seed for the simulation
        """
        super(WebotsMAVDroneEnv, self).__init__()
        
        # Import DroneEnergyModel only when needed
        try:
            from algorithms.drl.energy_model import DroneEnergyModel
            self.energy_model = DroneEnergyModel()
            self.has_energy_model = True
        except ImportError:
            print("Warning: DroneEnergyModel not found, energy-based rewards will be limited")
            self.has_energy_model = False
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Environment parameters
        self.headless = headless
        self.real_time_factor = real_time_factor
        self.max_episode_steps = max_episode_steps
        self.time_step = time_step
        self.reward_type = reward_type
        self.sim_seed = sim_seed if sim_seed is not None else np.random.randint(0, 10000)
        
        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Webots objects
        self.robot = None
        self.supervisor = None
        self.webots_process = None
        
        # Waypoints
        self.waypoints = self._load_waypoints(self.config.get('waypoints_file', None))
        self.current_waypoint_idx = 0
        
        # Obstacles
        self.obstacles = self._load_obstacles(self.config.get('obstacles_file', None))
        
        # Initialize spaces
        self._setup_action_observation_spaces()
        
        # Initialize sensors and actuators
        self.sensors = {}
        self.actuators = {}
        
        # For rendering
        self.rendering_initialized = False
        self.viewer = None
        
        # Data logging
        self.log_dir = self.config.get('log_dir', 'drl_training_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = None
        self.csv_writer = None
        
        # Start Webots if running in real simulation mode
        if WEBOTS_AVAILABLE and not self.config.get('sim_only', False):
            self._start_webots()
            self._init_robot()
    
    def _load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            'sim_only': False,
            'world_file': 'worlds/drone_rl_training.wbt',
            'waypoints_file': None,
            'obstacles_file': None,
            'log_dir': 'drl_training_logs',
            'reward_weights': {
                'waypoint_reached': 100.0,
                'distance_improvement': 1.0,
                'collision': -100.0,
                'action_smoothness': -0.1,
                'energy_efficiency': -0.01
            },
            'done_on_collision': True,
            'state_normalization': True
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Merge configs
                for key, value in user_config.items():
                    if key == 'reward_weights' and 'reward_weights' in default_config:
                        # Merge reward weights
                        for rw_key, rw_value in value.items():
                            default_config['reward_weights'][rw_key] = rw_value
                    else:
                        default_config[key] = value
        
        return default_config
    
    def _setup_action_observation_spaces(self):
        """Define action and observation spaces"""
        # Action space: [roll, pitch, yaw_rate, thrust]
        # Each in range [-1, 1] which will be scaled to appropriate ranges
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Observation space components
        obs_components = {
            # Drone state
            'position': spaces.Box(low=-100.0, high=100.0, shape=(3,), dtype=np.float32),
            'orientation': spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),  # quaternion
            'linear_velocity': spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),
            'angular_velocity': spaces.Box(low=-3.0, high=3.0, shape=(3,), dtype=np.float32),
            
            # Target information
            'target_direction': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'target_distance': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            
            # Sensor readings
            'lidar_ranges': spaces.Box(low=0.0, high=10.0, shape=(16,), dtype=np.float32),
            
            # Progress information
            'waypoint_idx': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'progress': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        }
        
        # Combine all components into a Dict space
        self.observation_space = spaces.Dict(obs_components)
        
        # Flatten observation space for algorithms that require it
        self.flat_observation_space = self._flatten_observation_space()
    
    def _flatten_observation_space(self):
        """Convert Dict observation space to a flattened Box space"""
        # Calculate total size of flattened observation
        total_size = sum([np.prod(space.shape) for space in self.observation_space.spaces.values()])
        
        # Create flattened observation space
        return spaces.Box(low=-float('inf'), high=float('inf'), shape=(int(total_size),), dtype=np.float32)
    
    def _flatten_observation(self, obs_dict):
        """Flatten a dictionary observation into a single array"""
        return np.concatenate([obs_dict[key].flatten() for key in sorted(obs_dict.keys())])
    
    def _load_waypoints(self, waypoints_file):
        """Load waypoints from file or generate defaults"""
        if waypoints_file and os.path.exists(waypoints_file):
            waypoints = []
            with open(waypoints_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.strip().startswith('#'):
                        coords = line.strip().split()
                        if len(coords) >= 3:
                            try:
                                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                                waypoints.append([x, y, z])
                            except ValueError:
                                pass
            if waypoints:
                return np.array(waypoints)
        
        # Default waypoints if none loaded
        return np.array([
            [0, 0, 1],
            [2, 2, 1.5],
            [4, 0, 2],
            [6, -2, 1.5],
            [8, 0, 1]
        ])
    
    def _load_obstacles(self, obstacles_file):
        """Load obstacles from file or generate defaults"""
        if obstacles_file and os.path.exists(obstacles_file):
            obstacles = []
            with open(obstacles_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.strip().startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 6:  # x, y, z, width, height, depth
                            try:
                                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                                width, height, depth = float(parts[3]), float(parts[4]), float(parts[5])
                                obstacles.append({
                                    'position': [x, y, z],
                                    'size': [width, height, depth]
                                })
                            except ValueError:
                                pass
            if obstacles:
                return obstacles
        
        # Default obstacles if none loaded (empty list - no obstacles)
        return []
    
    def _start_webots(self):
        """Start Webots process with the specified world file"""
        if self.webots_process is not None:
            print("Webots is already running.")
            return
        
        world_path = os.path.abspath(self.config['world_file'])
        if not os.path.exists(world_path):
            raise FileNotFoundError(f"World file not found: {world_path}")
        
        # Construct Webots command
        cmd = ['webots']
        if self.headless:
            cmd.append('--batch')
        cmd.append('--mode=fast')
        cmd.append(f'--stdout')
        cmd.append(f'--stderr')
        cmd.append(f'--no-rendering')
        cmd.append(world_path)
        
        # Start Webots process
        try:
            self.webots_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            # Wait for Webots to start
            time.sleep(5)
            print("Webots started successfully.")
        except Exception as e:
            print(f"Failed to start Webots: {e}")
            self.webots_process = None
    
    def _init_robot(self):
        """Initialize robot controller and sensors"""
        if not WEBOTS_AVAILABLE:
            return
        
        try:
            # Initialize robot and supervisor
            self.robot = Supervisor()
            self.supervisor = self.robot
            
            # Set time step
            self.robot.setTime(self.time_step)
            
            # Initialize sensors
            self._init_sensors()
            
            # Initialize actuators
            self._init_actuators()
            
            print("Robot initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize robot: {e}")
            self.robot = None
            self.supervisor = None
    
    def _init_sensors(self):
        """Initialize robot sensors"""
        if not self.robot:
            return
        
        try:
            # GPS
            self.sensors['gps'] = self.robot.getDevice('gps')
            self.sensors['gps'].enable(self.time_step)
            
            # IMU (inertial unit)
            self.sensors['imu'] = self.robot.getDevice('inertial unit')
            self.sensors['imu'].enable(self.time_step)
            
            # Gyro
            self.sensors['gyro'] = self.robot.getDevice('gyro')
            self.sensors['gyro'].enable(self.time_step)
            
            # Accelerometer
            self.sensors['accelerometer'] = self.robot.getDevice('accelerometer')
            self.sensors['accelerometer'].enable(self.time_step)
            
            # Compass
            self.sensors['compass'] = self.robot.getDevice('compass')
            self.sensors['compass'].enable(self.time_step)
            
            # Camera
            self.sensors['camera'] = self.robot.getDevice('camera')
            self.sensors['camera'].enable(self.time_step)
            
            # Distance sensors (lidar simulation)
            for i in range(1, 17):  # Assuming 16 distance sensors
                sensor_name = f'distance_sensor{i}'
                try:
                    self.sensors[sensor_name] = self.robot.getDevice(sensor_name)
                    self.sensors[sensor_name].enable(self.time_step)
                except:
                    print(f"Warning: {sensor_name} not found")
            
            print("Sensors initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize sensors: {e}")
    
    def _init_actuators(self):
        """Initialize robot actuators"""
        if not self.robot:
            return
        
        try:
            # Motors
            for i in range(1, 5):  # Assuming 4 motors for a quadcopter
                motor_name = f'motor{i}'
                try:
                    self.actuators[motor_name] = self.robot.getDevice(motor_name)
                    self.actuators[motor_name].setPosition(float('inf'))  # Velocity control mode
                    self.actuators[motor_name].setVelocity(0.0)
                except:
                    print(f"Warning: {motor_name} not found")
            
            print("Actuators initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize actuators: {e}")
    
    def _get_observation(self):
        """Get current observation from the environment"""
        if not WEBOTS_AVAILABLE or not self.robot:
            # In simulation-only mode, return a synthetic observation
            return self._get_synthetic_observation()
        
        try:
            # Get drone position
            position = np.array(self.sensors['gps'].getValues())
            
            # Get drone orientation (as rotation matrix, convert to quaternion)
            rot_matrix = np.array(self.sensors['imu'].getRollPitchYaw())
            # Convert to quaternion (simplified, should use proper conversion in production)
            orientation = np.array([np.cos(rot_matrix[0]/2), np.sin(rot_matrix[0]/2), 0, 0])
            
            # Get linear velocity (from supervisor API)
            drone_node = self.supervisor.getSelf()
            linear_velocity = np.array(drone_node.getVelocity()[:3])
            
            # Get angular velocity
            angular_velocity = np.array(self.sensors['gyro'].getValues())
            
            # Get target waypoint information
            if self.current_waypoint_idx < len(self.waypoints):
                target_waypoint = self.waypoints[self.current_waypoint_idx]
                
                # Direction to target (normalized)
                target_vector = target_waypoint - position
                target_distance = np.linalg.norm(target_vector)
                target_direction = target_vector / (target_distance + 1e-10)  # Avoid division by zero
                
                # Normalize waypoint index for progress indication
                normalized_waypoint_idx = self.current_waypoint_idx / max(1, len(self.waypoints) - 1)
                
                # Overall mission progress
                progress = normalized_waypoint_idx
            else:
                # If all waypoints completed
                target_direction = np.zeros(3)
                target_distance = np.array([0.0])
                normalized_waypoint_idx = np.array([1.0])
                progress = np.array([1.0])
            
            # Get LiDAR-like readings from distance sensors
            lidar_ranges = np.zeros(16)
            for i in range(1, 17):
                sensor_name = f'distance_sensor{i}'
                if sensor_name in self.sensors:
                    lidar_ranges[i-1] = self.sensors[sensor_name].getValue()
                    
            # Combine all components into the observation dictionary
            obs_dict = {
                'position': position,
                'orientation': orientation,
                'linear_velocity': linear_velocity,
                'angular_velocity': angular_velocity,
                'target_direction': target_direction,
                'target_distance': np.array([target_distance]),
                'lidar_ranges': lidar_ranges,
                'waypoint_idx': np.array([normalized_waypoint_idx]),
                'progress': np.array([progress])
            }
            
            # Apply state normalization if configured
            if self.config.get('state_normalization', True):
                obs_dict = self._normalize_observation(obs_dict)
            
            return obs_dict
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            return self._get_synthetic_observation()
    
    def _get_synthetic_observation(self):
        """Generate a synthetic observation for testing or simulation-only mode"""
        # Generate a basic synthetic observation with zeros
        obs_dict = {
            'position': np.zeros(3),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            'linear_velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'target_direction': np.array([1.0, 0.0, 0.0]),  # Forward direction
            'target_distance': np.array([1.0]),
            'lidar_ranges': np.ones(16) * 10.0,  # Maximum range
            'waypoint_idx': np.array([0.0]),
            'progress': np.array([0.0])
        }
        return obs_dict
    
    def _normalize_observation(self, obs_dict):
        """Normalize observation values to improve learning stability"""
        # Clone the observation to avoid modifying the original
        normalized_obs = {k: v.copy() for k, v in obs_dict.items()}
        
        # Normalize values based on expected ranges
        # Position is already normalized relative to workspace in spaces definition
        # Quaternion orientation is already normalized by nature
        
        # Normalize velocities
        normalized_obs['linear_velocity'] /= 10.0  # Assuming max velocity is 10 m/s
        normalized_obs['angular_velocity'] /= 3.0  # Assuming max angular velocity is 3 rad/s
        
        # Target direction is already normalized
        # Target distance is normalized by dividing by expected maximum distance
        normalized_obs['target_distance'] /= 100.0
        
        # Normalize lidar ranges
        normalized_obs['lidar_ranges'] /= 10.0  # Assuming max range is 10m
        
        # Waypoint index and progress are already normalized between 0-1
        
        return normalized_obs
    
    def _apply_action(self, action):
        """Apply the action to the drone motors"""
        if not WEBOTS_AVAILABLE or not self.robot:
            return  # In simulation-only mode, just return
        
        try:
            # Scale actions from [-1, 1] to appropriate ranges
            # Assuming quadcopter control mapping:
            # action[0] = roll control (-1 to 1) -> differential thrust between left and right motors
            # action[1] = pitch control (-1 to 1) -> differential thrust between front and back motors
            # action[2] = yaw rate control (-1 to 1) -> differential thrust in clockwise/counterclockwise pairs
            # action[3] = thrust control (-1 to 1) -> collective thrust of all motors
            
            # Base speed for all motors (scaled from thrust action)
            base_speed = 100.0 * (action[3] * 0.5 + 0.5)  # Maps from [-1,1] to [0,100]
            
            # Calculate motor speeds based on control inputs
            roll_diff = action[0] * 50.0  # Differential for roll
            pitch_diff = action[1] * 50.0  # Differential for pitch
            yaw_diff = action[2] * 30.0  # Differential for yaw
            
            # Calculate motor speeds (this is a simplified example, real drones use more complex models)
            # Motor arrangement: 1=front-left, 2=front-right, 3=rear-right, 4=rear-left
            motor_speeds = [
                base_speed - roll_diff + pitch_diff - yaw_diff,  # Motor 1 (front-left)
                base_speed + roll_diff + pitch_diff + yaw_diff,  # Motor 2 (front-right)
                base_speed + roll_diff - pitch_diff - yaw_diff,  # Motor 3 (rear-right)
                base_speed - roll_diff - pitch_diff + yaw_diff   # Motor 4 (rear-left)
            ]
            
            # Apply motor speeds, ensuring they're within valid ranges
            for i, speed in enumerate(motor_speeds):
                motor_name = f'motor{i+1}'
                if motor_name in self.actuators:
                    # Ensure motor speed is within valid range (0 to MAX_SPEED)
                    clamped_speed = max(0, min(1000, speed))
                    self.actuators[motor_name].setVelocity(clamped_speed)
        
        except Exception as e:
            print(f"Error applying action: {e}")
    
    def _check_collision(self):
        """Check if the drone has collided with an obstacle"""
        if not WEBOTS_AVAILABLE or not self.robot:
            return False
        
        try:
            # Get drone position
            position = np.array(self.sensors['gps'].getValues())
            
            # Check collision with obstacles
            for obstacle in self.obstacles:
                obstacle_pos = np.array(obstacle['position'])
                obstacle_size = np.array(obstacle['size'])
                
                # Check if drone is inside the obstacle box (plus a small margin)
                margin = 0.2  # 20cm margin
                
                # Calculate box boundaries
                min_bounds = obstacle_pos - obstacle_size/2 - margin
                max_bounds = obstacle_pos + obstacle_size/2 + margin
                
                # Check if drone position is inside the boundary box
                if (np.all(position >= min_bounds) and np.all(position <= max_bounds)):
                    return True
            
            # If no obstacle collision detected
            return False
            
        except Exception as e:
            print(f"Error checking collision: {e}")
            return False
    
    def _check_waypoint_reached(self):
        """Check if the current waypoint has been reached"""
        if self.current_waypoint_idx >= len(self.waypoints):
            return False
        
        if not WEBOTS_AVAILABLE or not self.robot:
            return False
        
        try:
            # Get drone position
            position = np.array(self.sensors['gps'].getValues())
            
            # Get current target waypoint
            target = self.waypoints[self.current_waypoint_idx]
            
            # Calculate distance to waypoint
            distance = np.linalg.norm(position - target)
            
            # Check if waypoint is reached (within threshold)
            waypoint_threshold = 0.5  # 50cm
            return distance < waypoint_threshold
            
        except Exception as e:
            print(f"Error checking waypoint: {e}")
            return False
    
    def _calculate_reward(self, prev_observation, action, observation, collision, waypoint_reached):
        """
        Calculate reward based on drone's performance
        
        Reward components:
        1. Waypoint reached: Large positive reward
        2. Making progress toward waypoint: Small positive reward based on distance improvement
        3. Collision: Large negative reward
        4. Action smoothness: Small negative reward for jerky actions
        5. Energy efficiency: Small negative reward based on motor usage and energy model
        6. Height penalty: Small negative reward for flying too high or too low
        7. Direction alignment: Small positive reward for facing toward the waypoint
        """
        reward = 0.0
        reward_info = {}
        
        # Get reward weights from config
        weights = self.config.get('reward_weights', {})
        
        # 1. Waypoint reached reward
        if waypoint_reached:
            wp_reward = weights.get('waypoint_reached', 100.0)
            reward += wp_reward
            reward_info['waypoint_reached'] = wp_reward
        
        # 2. Distance improvement reward
        if not waypoint_reached and self.current_waypoint_idx < len(self.waypoints):
            # Calculate previous distance to waypoint
            prev_position = prev_observation['position']
            prev_distance = np.linalg.norm(prev_position - self.waypoints[self.current_waypoint_idx])
            
            # Calculate current distance to waypoint
            curr_position = observation['position']
            curr_distance = np.linalg.norm(curr_position - self.waypoints[self.current_waypoint_idx])
            
            # Reward for getting closer to waypoint
            distance_improvement = prev_distance - curr_distance
            dist_reward = distance_improvement * weights.get('distance_improvement', 1.0)
            reward += dist_reward
            reward_info['distance_improvement'] = dist_reward
        
        # 3. Collision penalty
        if collision:
            coll_penalty = weights.get('collision', -100.0)
            reward += coll_penalty
            reward_info['collision'] = coll_penalty
        
        # 4. Action smoothness penalty
        # Penalize large changes in actions for smoother control
        if hasattr(self, 'prev_action'):
            action_diff = np.sum(np.abs(action - self.prev_action))
            smoothness_penalty = action_diff * weights.get('action_smoothness', -0.1)
            reward += smoothness_penalty
            reward_info['action_smoothness'] = smoothness_penalty
        
        # 5. Energy efficiency penalty
        # 5.1 基本能量消耗 - 基于推力的简单模型
        basic_energy_penalty = (np.abs(action[3]) - 0.5) * weights.get('energy_efficiency', -0.01)
        reward += basic_energy_penalty
        reward_info['basic_energy'] = basic_energy_penalty
        
        # 5.2 高级能量消耗 - 如果提供了能量模型
        if self.has_energy_model:
            # 构建状态字典
            state_dict = {
                'position': observation['position'],
                'linear_velocity': observation.get('linear_velocity', np.zeros(3)),
                'angular_velocity': observation.get('angular_velocity', np.zeros(3)),
                'orientation': observation.get('orientation', np.zeros(3))
            }
            
            # 计算能量消耗
            energy_consumption = self.energy_model.calculate_energy_consumption(state_dict, action)
            
            # 添加能量奖励（负值，因为我们希望最小化能量消耗）
            energy_reward = -energy_consumption * weights.get('advanced_energy', -0.005)
            reward += energy_reward
            reward_info['advanced_energy'] = energy_reward
            
            # 记录累计能量消耗
            if not hasattr(self, 'total_energy_consumption'):
                self.total_energy_consumption = 0.0
            self.total_energy_consumption += energy_consumption
            
            # 记录飞行距离
            if hasattr(self, 'prev_position'):
                distance = np.linalg.norm(observation['position'] - self.prev_position)
                if not hasattr(self, 'total_distance'):
                    self.total_distance = 0.0
                self.total_distance += distance
            
            # 更新上一个位置
            self.prev_position = observation['position']
        
        # 6. 高度惩罚 - 鼓励无人机在合理高度飞行
        target_height = 1.0  # 目标高度
        height_diff = abs(observation['position'][2] - target_height)
        height_penalty = height_diff * weights.get('height_penalty', -0.01)
        reward += height_penalty
        reward_info['height_penalty'] = height_penalty
        
        # 7. 方向对齐奖励 - 鼓励无人机朝向目标
        if self.current_waypoint_idx < len(self.waypoints) and 'orientation' in observation:
            # 计算当前到目标的向量
            curr_to_waypoint = self.waypoints[self.current_waypoint_idx] - observation['position']
            curr_to_waypoint_norm = curr_to_waypoint / np.linalg.norm(curr_to_waypoint)
            
            # 获取无人机前向向量（假设使用四元数表示方向）
            drone_forward = self._get_forward_vector(observation['orientation'])
            
            # 计算方向对齐度（点积）
            alignment = np.dot(curr_to_waypoint_norm, drone_forward)
            
            # 添加方向奖励
            alignment_reward = alignment * weights.get('alignment', 0.1)
            reward += alignment_reward
            reward_info['alignment'] = alignment_reward
        
        # Store current action for next step comparison
        self.prev_action = action
        
        return reward, reward_info
    
    def _get_forward_vector(self, orientation):
        """将方向（四元数或欧拉角）转换为前向向量"""
        if len(orientation) == 4:  # 四元数 [w, x, y, z]
            w, x, y, z = orientation
            # 前向向量的计算（根据四元数旋转 [1, 0, 0]）
            forward_x = 1 - 2 * (y*y + z*z)
            forward_y = 2 * (x*y + w*z)
            forward_z = 2 * (x*z - w*y)
        else:  # 欧拉角 [roll, pitch, yaw]
            roll, pitch, yaw = orientation
            # 前向向量的计算（根据欧拉角）
            forward_x = np.cos(yaw) * np.cos(pitch)
            forward_y = np.sin(yaw) * np.cos(pitch)
            forward_z = -np.sin(pitch)
        
        # 归一化前向向量
        forward = np.array([forward_x, forward_y, forward_z])
        return forward / np.linalg.norm(forward)
    
    def _setup_logging(self):
        """Set up logging for training data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.log_dir, f"training_log_{timestamp}.csv")
        
        self.log_file = open(log_filename, 'w', newline='')
        fieldnames = [
            'episode', 'step', 'total_reward', 'reward', 
            'waypoint_reached', 'collision', 'waypoint_idx',
            'position_x', 'position_y', 'position_z',
            'target_x', 'target_y', 'target_z',
            'distance_to_target', 'action_roll', 'action_pitch',
            'action_yaw', 'action_thrust'
        ]
        
        self.csv_writer = csv.DictWriter(self.log_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
    
    def _log_step(self, observation, action, reward, reward_info, waypoint_reached, collision):
        """Log step data to CSV file"""
        if self.csv_writer is None:
            self._setup_logging()
        
        # Current target waypoint
        if self.current_waypoint_idx < len(self.waypoints):
            target_wp = self.waypoints[self.current_waypoint_idx]
        else:
            target_wp = np.zeros(3)
        
        # Log data to CSV
        self.csv_writer.writerow({
            'episode': self.current_episode,
            'step': self.current_step,
            'total_reward': self.total_reward,
            'reward': reward,
            'waypoint_reached': 1 if waypoint_reached else 0,
            'collision': 1 if collision else 0,
            'waypoint_idx': self.current_waypoint_idx,
            'position_x': observation['position'][0],
            'position_y': observation['position'][1],
            'position_z': observation['position'][2],
            'target_x': target_wp[0],
            'target_y': target_wp[1],
            'target_z': target_wp[2],
            'distance_to_target': observation['target_distance'][0],
            'action_roll': action[0],
            'action_pitch': action[1],
            'action_yaw': action[2],
            'action_thrust': action[3]
        })
        self.log_file.flush()  # Ensure data is written immediately
    
    def reset(self):
        """Reset the environment to start a new episode"""
        self.current_step = 0
        self.total_reward = 0.0
        self.current_episode += 1
        self.current_waypoint_idx = 0
        self.prev_action = np.zeros(4)
        
        if WEBOTS_AVAILABLE and self.robot:
            try:
                # Reset robot position
                drone_node = self.supervisor.getSelf()
                starting_pose = [0, 0, 0.5, 0, 0, 0, 1]  # [x, y, z, q1, q2, q3, q4] at 0.5m height
                drone_node.getField('translation').setSFVec3f(starting_pose[:3])
                drone_node.getField('rotation').setSFRotation(starting_pose[3:])
                
                # Reset physics
                drone_node.resetPhysics()
                
                # Step simulation to stabilize
                for _ in range(10):
                    self.robot.step(self.time_step)
            
            except Exception as e:
                print(f"Error resetting environment: {e}")
        
        # Get initial observation
        observation = self._get_observation()
        
        # Create an info dictionary
        info = {
            'episode': self.current_episode,
            'waypoint_idx': self.current_waypoint_idx
        }
        
        # Return flattened observation if needed by the algorithm
        if hasattr(self, 'flat_observation') and self.flat_observation:
            return self._flatten_observation(observation), info
        return observation, info
    
    def step(self, action):
        """Take a step in the environment with the given action"""
        self.current_step += 1
        
        # Store previous observation for reward calculation
        prev_observation = self._get_observation()
        
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        if WEBOTS_AVAILABLE and self.robot:
            self.robot.step(self.time_step)
        
        # Get new observation
        observation = self._get_observation()
        
        # Check for collision
        collision = self._check_collision()
        
        # Check if waypoint reached
        waypoint_reached = self._check_waypoint_reached()
        if waypoint_reached and self.current_waypoint_idx < len(self.waypoints):
            print(f"Waypoint {self.current_waypoint_idx + 1} reached!")
            self.current_waypoint_idx += 1
        
        # Calculate reward
        reward, reward_info = self._calculate_reward(prev_observation, action, observation, collision, waypoint_reached)
        self.total_reward += reward
        
        # Determine if episode is terminated
        terminated = False
        
        # Terminated if all waypoints reached
        if self.current_waypoint_idx >= len(self.waypoints):
            terminated = True
            print(f"All waypoints reached! Episode {self.current_episode} completed successfully!")
        
        # Terminated if collision detected and config specifies to end on collision
        if collision and self.config.get('done_on_collision', True):
            terminated = True
            print(f"Collision detected! Episode {self.current_episode} failed.")
        
        # Truncated if maximum steps reached
        truncated = self.current_step >= self.max_episode_steps
        if truncated:
            print(f"Maximum steps reached. Episode {self.current_episode} terminated.")
        
        # Log step data
        self._log_step(observation, action, reward, reward_info, waypoint_reached, collision)
        
        # If episode is done, record episode results
        if terminated or truncated:
            self.episode_rewards.append(self.total_reward)
            self.episode_lengths.append(self.current_step)
            
            # Print episode summary
            print(f"Episode {self.current_episode} completed:")
            print(f"  Total reward: {self.total_reward}")
            print(f"  Steps: {self.current_step}")
            print(f"  Waypoints reached: {self.current_waypoint_idx}/{len(self.waypoints)}")
        
        # Additional info for debugging and visualization
        info = {
            'waypoint_reached': waypoint_reached,
            'collision': collision,
            'waypoint_idx': self.current_waypoint_idx,
            'reward_info': reward_info
        }
        
        # Return flattened observation if needed by the algorithm
        if hasattr(self, 'flat_observation') and self.flat_observation:
            return self._flatten_observation(observation), reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the environment (not essential for Webots which has its own rendering)"""
        if mode == 'human':
            # Webots has its own rendering, nothing to do here if not headless
            if not self.headless:
                return None
            
            # For headless mode, we could implement a simple visualization here
            # (not implemented in this example)
            pass
        
        elif mode == 'rgb_array':
            # Return camera image if available
            if WEBOTS_AVAILABLE and self.robot and 'camera' in self.sensors:
                try:
                    camera = self.sensors['camera']
                    image = camera.getImage()
                    width = camera.getWidth()
                    height = camera.getHeight()
                    
                    # Convert image data to numpy array
                    # (This is a simple and potentially inefficient implementation)
                    img = np.zeros((height, width, 3), dtype=np.uint8)
                    for y in range(height):
                        for x in range(width):
                            idx = (y * width + x) * 4  # RGBA format
                            img[y, x, 0] = image[idx]    # R
                            img[y, x, 1] = image[idx+1]  # G
                            img[y, x, 2] = image[idx+2]  # B
                    
                    return img
                except:
                    # Return empty image if camera not available
                    return np.zeros((240, 320, 3), dtype=np.uint8)
            
            # Default empty image
            return np.zeros((240, 320, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources"""
        # Close log file if open
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.csv_writer = None
        
        # Terminate Webots process if running
        if self.webots_process:
            print("Terminating Webots process...")
            try:
                self.webots_process.terminate()
                self.webots_process.wait(timeout=5)
            except:
                print("Forcing Webots process to terminate...")
                self.webots_process.kill()
            
            self.webots_process = None
        
        # Clean up robot resources
        if self.robot:
            self.robot = None
            self.supervisor = None
            self.sensors.clear()
            self.actuators.clear()


# Simulated Environment for development without Webots
class SimulatedDroneEnv(gym.Env):
    """
    Simulated Drone Environment for development and testing without Webots.
    This provides the same interface as WebotsMAVDroneEnv but works without the actual simulator.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config_file=None):
        """
        Initialize the Simulated Drone Environment
        
        Args:
            config_file: Path to the configuration file
        """
        super(SimulatedDroneEnv, self).__init__()
        
        # Import DroneEnergyModel only when needed
        try:
            from algorithms.drl.energy_model import DroneEnergyModel
            self.energy_model = DroneEnergyModel()
            self.has_energy_model = True
        except ImportError:
            print("Warning: DroneEnergyModel not found, energy-based rewards will be limited")
            self.has_energy_model = False
        
        # Load configuration
        self.config_file = config_file
        self.config = self._load_config(config_file)
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Set flat observation flag (defaults to True)
        self.flat_observation = True
        
        # Initialize state
        self.reset()
    
    def _load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            'sim_only': True,
            'world_file': None,
            'waypoints_file': None,
            'obstacles_file': None,
            'reward_weights': {
                'progress': 10.0,
                'distance': -1.0,
                'collision': -100.0,
                'completion': 100.0,
                'altitude': -0.1,
                'stability': -0.05,
                'energy': -0.01
            },
            'done_on_collision': True,
            'state_normalization': True,
            'max_episode_steps': 1000,
            'time_step': 32
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge configs
                    for key, value in user_config.items():
                        if key == 'reward_weights' and 'reward_weights' in default_config:
                            # Merge reward weights
                            for rw_key, rw_value in value.items():
                                default_config['reward_weights'][rw_key] = rw_value
                        else:
                            default_config[key] = value
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        return default_config
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Define observation space components
        obs_components = {
            'position': spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            'rotation': spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
            'velocity': spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            'waypoint': spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            'distance_to_waypoints': spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
            'lidar': spaces.Box(low=0, high=100, shape=(16,), dtype=np.float32),
        }
        
        # Create Dict observation space
        self.observation_space = spaces.Dict(obs_components)
        
        # Create flat observation space
        self.flat_observation_space = self._flatten_observation_space()
        
        # Define action space (continuous)
        # [roll, pitch, yaw_rate, thrust]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0]), 
            high=np.array([1, 1, 1, 1]), 
            dtype=np.float32
        )
    
    def _flatten_observation_space(self):
        """Flatten Dict observation space to Box for easier use with algorithms"""
        total_size = sum([np.prod(space.shape) for space in self.observation_space.spaces.values()])
        return spaces.Box(low=-np.inf, high=np.inf, shape=(int(total_size),), dtype=np.float32)
    
    def _flatten_observation(self, obs_dict):
        """Flatten dictionary observation into a single array"""
        return np.concatenate([obs_dict[k].flatten() for k in sorted(obs_dict.keys())])
    
    def _get_observation(self):
        """Generate simulated observation"""
        # Simulate drone state
        obs = {
            'position': np.array([self.drone_position[0], self.drone_position[1], self.drone_position[2]], dtype=np.float32),
            'rotation': np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'velocity': np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'waypoint': np.array([self.current_waypoint[0], self.current_waypoint[1], self.current_waypoint[2]], dtype=np.float32),
            'distance_to_waypoints': np.array([np.linalg.norm(np.array(self.drone_position) - np.array(self.current_waypoint))], dtype=np.float32),
            'lidar': np.full(16, 100.0, dtype=np.float32),  # Simulated lidar with no obstacles
        }
        
        # Apply noise to make it more realistic
        for key in obs:
            obs[key] += np.random.normal(0, 0.01, obs[key].shape).astype(np.float32)
        
        return obs
    
    def reset(self):
        """Reset environment to initial state"""
        # Initialize waypoints (simple path for simulation)
        self.waypoints = [
            [0, 0, 1],   # Takeoff
            [5, 0, 3],   # First waypoint
            [5, 5, 5],   # Second waypoint
            [0, 5, 3],   # Third waypoint
            [0, 0, 1]    # Landing
        ]
        
        # Initialize state
        self.drone_position = [0, 0, 0]
        self.waypoint_index = 0
        self.current_waypoint = self.waypoints[0]
        self.step_count = 0
        
        # Get observation
        obs_dict = self._get_observation()
        
        # Return observation and empty info dict to comply with Gym v0.26+ API
        if self.flat_observation:
            return self._flatten_observation(obs_dict), {}
        return obs_dict, {}
    
    def step(self, action):
        """Take a step in environment with given action"""
        # Increment step counter
        self.step_count += 1
        
        # Apply action to move drone (simplified simulation)
        move_scale = 0.1  # Scale action to movement
        
        # Extract action components
        roll = action[0]      # Rotate around x-axis (move in y)
        pitch = action[1]     # Rotate around y-axis (move in x)
        yaw_rate = action[2]  # Not used in this simple simulation
        thrust = action[3]    # Move in z
        
        # Update position based on action
        self.drone_position[0] += pitch * move_scale
        self.drone_position[1] -= roll * move_scale
        self.drone_position[2] += (thrust - 0.5) * move_scale
        
        # Check waypoint reached
        dist_to_waypoint = np.linalg.norm(np.array(self.drone_position) - np.array(self.current_waypoint))
        waypoint_reached = dist_to_waypoint < 0.5
        
        # Update waypoint if reached
        if waypoint_reached and self.waypoint_index < len(self.waypoints) - 1:
            self.waypoint_index += 1
            self.current_waypoint = self.waypoints[self.waypoint_index]
        
        # Calculate reward
        reward = -0.1  # Small negative reward for each step
        reward -= dist_to_waypoint * 0.1  # Distance penalty
        
        if waypoint_reached:
            reward += 10.0  # Reward for reaching waypoint
        
        # Check if task is complete (reached final waypoint)
        terminated = self.waypoint_index == len(self.waypoints) - 1 and waypoint_reached
        
        # Check if episode is done due to step limit
        truncated = self.step_count >= self.config['max_episode_steps']
        
        # Get observation
        obs_dict = self._get_observation()
        
        # Additional info
        info = {
            'waypoint_reached': waypoint_reached,
            'waypoint_index': self.waypoint_index,
            'distance': dist_to_waypoint
        }
        
        # Return observation, reward, terminated, truncated, and info to comply with Gym v0.26+ API
        if self.flat_observation:
            return self._flatten_observation(obs_dict), reward, terminated, truncated, info
        return obs_dict, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the environment (not implemented for simulation)"""
        print(f"Drone position: {self.drone_position}, Waypoint: {self.current_waypoint}, Reward: {-np.linalg.norm(np.array(self.drone_position) - np.array(self.current_waypoint)) * 0.1}")
        return None
    
    def close(self):
        """Close the environment"""
        pass


# Utility functions for environment management

def create_env(config_file=None, headless=False, real_time_factor=1.0, flat_observation=True):
    """Factory function to create and configure a WebotsMAVDroneEnv or SimulatedDroneEnv instance"""
    # Use simulated environment if Webots is not available
    if not WEBOTS_AVAILABLE:
        print("Using simulated environment.")
        env = SimulatedDroneEnv(config_file=config_file)
        env.flat_observation = flat_observation
        return env
    
    # Use actual Webots environment
    env = WebotsMAVDroneEnv(
        config_file=config_file,
        headless=headless,
        real_time_factor=real_time_factor
    )
    
    # Set flat observation flag
    env.flat_observation = flat_observation
    
    return env

if __name__ == "__main__":
    # Simple test to verify environment functionality
    
    # Create environment
    env = create_env(headless=False)
    
    # Reset environment
    observation = env.reset()
    
    # Run for a few steps
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        
        # Take step
        observation, reward, done, info = env.step(action)
        
        print(f"Step {step}, Reward: {reward}, Done: {done}")
        
        if done:
            print("Episode finished early")
            break
    
    # Close environment
    env.close() 