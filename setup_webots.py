#!/usr/bin/env python3
"""
设置Webots仿真环境的脚本
- 创建Webots世界文件
- 初始化必要的目录结构
- 复制控制器文件
"""

import os
import shutil
import json
import argparse

def create_directory_structure():
    """创建必要的目录结构"""
    directories = [
        "webots/controllers/drone_controller",
        "webots/controllers/supervisor",
        "webots/worlds",
        "webots/protos",
        "algorithms/drl",
        "algorithms/common",
        "utils",
        "models/gatdrl",
        "models/ppo",
        "models/dqn",
        "models/td3"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Initialize package\n")
    
    print("Directory structure created.")

def create_webots_world_files():
    """创建Webots世界文件"""
    scenario_templates = {
        "sparse": {
            "num_buildings": 10,
            "max_height": 40,
            "description": "Sparse Urban Environment"
        },
        "mixed": {
            "num_buildings": 20,
            "max_height": 60,
            "description": "Mixed Urban Environment"
        },
        "dense": {
            "num_buildings": 30,
            "max_height": 100,
            "description": "Dense Urban Environment"
        }
    }
    
    # 基础世界文件模板
    base_template = """#VRML_SIM R2023a utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023a/projects/objects/floors/protos/Floor.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023a/projects/robots/dji/mavic/protos/Mavic2Pro.proto"

WorldInfo {
  info [
    "UAV Path Planning Simulation"
    "$description"
  ]
  title "UAV Path Planning"
  basicTimeStep 16
  FPS 30
}

Viewpoint {
  orientation 0.38 0.08 -0.92 0.8
  position -10 -10 15
  follow "Mavic2Pro"
}

TexturedBackground {
}

TexturedBackgroundLight {
}

Floor {
  size 200 200
  appearance PBRAppearance {
    baseColorMap ImageTexture {
      url [
        "textures/ground.jpg"
      ]
    }
    roughness 1
    metalness 0
  }
}

Mavic2Pro {
  translation 10 10 0.2
  rotation 0 0 1 0
  name "QUADCOPTER"
  controller "drone_controller"
  cameraSlot [
    Camera {
      width 400
      height 240
      near 0.2
    }
  ]
}

DEF SUPERVISOR Robot {
  children [
    Emitter {
      channel 1
    }
  ]
  supervisor TRUE
  controller "supervisor"
}

# Buildings and waypoints will be dynamically added in runtime
"""
    
    for scenario, config in scenario_templates.items():
        # 使用字符串替换而不是format
        file_content = base_template.replace("$description", config["description"])
        
        file_path = f"webots/worlds/{scenario}_scenario.wbt"
        with open(file_path, 'w') as f:
            f.write(file_content)
        
        print(f"Created Webots world file: {file_path}")

def copy_controller_files():
    """复制控制器文件到Webots目录"""
    # 无人机控制器
    drone_controller_path = "webots/controllers/drone_controller/drone_controller.py"
    drone_controller_content = """#!/usr/bin/env python3
\"\"\"
无人机控制器基础实现
\"\"\"
from controller import Robot, GPS, Compass, InertialUnit, Camera, Gyro, Motor, Accelerometer
import struct
import numpy as np
import sys
import os

# 添加算法目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../algorithms'))

# 导入算法模块
from drl.gatdrl import GATDRL
from drl.ppo import PPO
from drl.dqn import DQN
from drl.td3 import TD3

# 仿真参数
TIME_STEP = 32  # 毫秒，仿真时间步长

class DroneController:
    \"\"\"无人机控制器基类\"\"\"
    
    def __init__(self):
        # 初始化机器人
        self.robot = Robot()
        
        # 获取设备
        self.init_devices()
        
        # 获取环境信息
        self.algorithm = os.environ.get('ALGORITHM', 'GAT-DRL')
        self.scenario = os.environ.get('SCENARIO', 'mixed')
        
        print(f"Initializing drone controller with algorithm: {self.algorithm}")
        
        # 初始化路径规划
        self.init_path_planning()
    
    def init_devices(self):
        \"\"\"初始化设备\"\"\"
        # GPS
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(TIME_STEP)
        
        # 罗盘
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(TIME_STEP)
        
        # IMU
        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(TIME_STEP)
        
        # 陀螺仪
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(TIME_STEP)
        
        # 加速度计
        self.accelerometer = self.robot.getDevice("accelerometer")
        self.accelerometer.enable(TIME_STEP)
        
        # 相机
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(TIME_STEP)
        
        # 电机
        self.motors = []
        for i in range(4):
            motor_name = f"m{i+1}" if self.robot.getDevice(f"m{i+1}") else f"motor{i+1}"
            motor = self.robot.getDevice(motor_name)
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)
            self.motors.append(motor)
    
    def init_path_planning(self):
        \"\"\"初始化路径规划\"\"\"
        # 这里将根据算法类型加载相应的路径规划模块
        # 实际实现时需要根据具体算法进行调整
        pass
    
    def run(self):
        \"\"\"运行控制循环\"\"\"
        while self.robot.step(TIME_STEP) != -1:
            # 获取传感器数据
            gps_values = self.gps.getValues()
            compass_values = self.compass.getValues()
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            
            # 在此处实现路径规划和控制逻辑
            
            # TODO: 实现具体的控制算法
            
            # 测试用：简单的悬停
            for i in range(4):
                self.motors[i].setVelocity(4.0)

# 主函数
controller = DroneController()
controller.run()
"""

    with open(drone_controller_path, 'w') as f:
        f.write(drone_controller_content)
    
    # 监督者控制器
    supervisor_path = "webots/controllers/supervisor/supervisor.py"
    supervisor_content = """#!/usr/bin/env python3
\"\"\"
监督者控制器实现
用于管理仿真环境、建筑物和必经点
\"\"\"
from controller import Supervisor
import os
import sys
import json
import random
import numpy as np
import math

# 添加算法目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../algorithms'))

# 仿真参数
TIME_STEP = 32  # 毫秒，仿真时间步长

class SimulationSupervisor:
    \"\"\"仿真监督者\"\"\"
    
    def __init__(self):
        # 初始化监督者
        self.supervisor = Supervisor()
        
        # 获取环境信息
        self.scenario = os.environ.get('SCENARIO', 'mixed')
        self.num_waypoints = int(os.environ.get('NUM_WAYPOINTS', '5'))
        
        print(f"Initializing supervisor for {self.scenario} scenario with {self.num_waypoints} waypoints")
        
        # 生成环境
        self.generate_environment()
    
    def generate_environment(self):
        \"\"\"生成环境（建筑物和必经点）\"\"\"
        # 加载或生成建筑物数据
        buildings_data = self.load_buildings_data()
        
        # 创建建筑物
        self.create_buildings(buildings_data)
        
        # 创建必经点
        self.create_waypoints()
    
    def load_buildings_data(self):
        \"\"\"加载建筑物数据\"\"\"
        # 尝试从文件加载
        data_file = f"worlds/{self.scenario}_city_data.json"
        
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                data = json.load(f)
                return data.get('buildings', [])
        
        # 如果文件不存在，生成随机建筑物
        return self.generate_random_buildings()
    
    def generate_random_buildings(self):
        \"\"\"生成随机建筑物\"\"\"
        if self.scenario == 'sparse':
            num_buildings = 10
            max_height = 40
        elif self.scenario == 'dense':
            num_buildings = 30
            max_height = 100
        else:  # mixed
            num_buildings = 20
            max_height = 60
        
        buildings = []
        area_size = 150  # 区域大小
        
        for i in range(num_buildings):
            x = random.uniform(-area_size/2 + 20, area_size/2 - 20)
            y = random.uniform(-area_size/2 + 20, area_size/2 - 20)
            width = random.uniform(5, 15)
            length = random.uniform(5, 15)
            height = random.uniform(10, max_height)
            
            buildings.append({
                'position': [x, y, height/2],
                'size': [width, length, height]
            })
        
        return buildings
    
    def create_buildings(self, buildings_data):
        \"\"\"在仿真中创建建筑物\"\"\"
        root = self.supervisor.getRoot()
        children_field = root.getField('children')
        
        for i, building in enumerate(buildings_data):
            pos = building['position']
            size = building['size']
            
            # 创建建筑物节点
            building_node = children_field.importMFNodeFromString(
                f'''
                DEF BUILDING_{i} Solid {{
                    translation {pos[0]} {pos[1]} {pos[2]}
                    children [
                        Shape {{
                            appearance PBRAppearance {{
                                baseColor {random.random()} {random.random()} {random.random()}
                                roughness 0.5
                                metalness 0
                            }}
                            geometry Box {{
                                size {size[0]} {size[1]} {size[2]}
                            }}
                        }}
                    ]
                    boundingObject Box {{
                        size {size[0]} {size[1]} {size[2]}
                    }}
                    physics Physics {{}}
                }}
                '''
            )
    
    def create_waypoints(self):
        \"\"\"创建必经点\"\"\"
        # 基于场景类型和建筑物位置生成合适的必经点
        waypoints = self.generate_waypoints()
        
        root = self.supervisor.getRoot()
        children_field = root.getField('children')
        
        for i, waypoint in enumerate(waypoints):
            x, y, z = waypoint
            
            # 创建可视化标记
            waypoint_node = children_field.importMFNodeFromString(
                f'''
                DEF WAYPOINT_{i} Solid {{
                    translation {x} {y} {z}
                    children [
                        Shape {{
                            appearance PBRAppearance {{
                                baseColor 1 0 0
                                emissiveColor 1 0 0
                            }}
                            geometry Sphere {{
                                radius 1
                                subdivision 2
                            }}
                        }}
                    ]
                }}
                '''
            )
    
    def generate_waypoints(self):
        \"\"\"生成必经点\"\"\"
        waypoints = []
        area_size = 150
        
        # 获取现有建筑物位置，避免碰撞
        buildings = []
        root = self.supervisor.getRoot()
        children_field = root.getField('children')
        n = children_field.getCount()
        
        for i in range(n):
            node = children_field.getMFNode(i)
            if 'BUILDING' in node.getDef():
                pos = node.getField('translation').getSFVec3f()
                size = node.getField('children').getMFNode(0).getField('geometry').getSFNode().getField('size').getSFVec3f()
                buildings.append({'position': pos, 'size': size})
        
        # 在网格中生成waypoints
        grid_size = int(math.sqrt(self.num_waypoints))
        if grid_size * grid_size < self.num_waypoints:
            grid_size += 1
        
        cell_size = area_size / grid_size
        
        for i in range(self.num_waypoints):
            row = i // grid_size
            col = i % grid_size
            
            # 在网格单元中随机生成点
            x_min = -area_size/2 + col * cell_size
            x_max = x_min + cell_size
            y_min = -area_size/2 + row * cell_size
            y_max = y_min + cell_size
            
            # 设置位置
            for attempt in range(20):  # 尝试20次找到无碰撞位置
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                z = random.uniform(10, 50)  # 在10-50米高度
                
                # 检查与建筑物的碰撞
                collision = False
                for building in buildings:
                    b_pos = building['position']
                    b_size = building['size']
                    
                    if (abs(x - b_pos[0]) < b_size[0]/2 + 5 and
                        abs(y - b_pos[1]) < b_size[1]/2 + 5 and
                        abs(z - b_pos[2]) < b_size[2]/2 + 5):
                        collision = True
                        break
                
                if not collision:
                    waypoints.append([x, y, z])
                    break
            
            if attempt == 19:  # 如果尝试了所有次数仍然失败
                # 随机找一个位置
                x = random.uniform(-area_size/2, area_size/2)
                y = random.uniform(-area_size/2, area_size/2)
                z = random.uniform(max([b['position'][2] + b['size'][2]/2 for b in buildings], default=0) + 10, 80)
                waypoints.append([x, y, z])
        
        return waypoints
    
    def run(self):
        \"\"\"运行监督者控制器\"\"\"
        while self.supervisor.step(TIME_STEP) != -1:
            # 持续监控仿真状态
            pass

# 主函数
supervisor = SimulationSupervisor()
supervisor.run()
"""

    with open(supervisor_path, 'w') as f:
        f.write(supervisor_content)
    
    print("Controller files created.")

def copy_utility_files():
    """创建实用工具文件"""
    # 创建utils目录下的实用工具
    robotics_utils_path = "utils/robotics_utils.py"
    robotics_utils_content = """#!/usr/bin/env python3
\"\"\"
机器人学相关的实用工具函数
\"\"\"
import numpy as np
import math

def euler_to_quaternion(roll, pitch, yaw):
    \"\"\"欧拉角转四元数\"\"\"
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return [w, x, y, z]

def quaternion_to_euler(q):
    \"\"\"四元数转欧拉角\"\"\"
    # 提取四元数分量
    w, x, y, z = q
    
    # 计算欧拉角
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    
    return [roll, pitch, yaw]

def rotation_matrix(roll, pitch, yaw):
    \"\"\"计算旋转矩阵\"\"\"
    # 计算三角函数值
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    
    # 构建旋转矩阵
    R = np.array([
        [cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
        [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
        [-sp, sr*cp, cr*cp]
    ])
    
    return R

def calculate_distance(point1, point2):
    \"\"\"计算两点之间的欧几里得距离\"\"\"
    return np.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

def calculate_energy(velocity, time_step):
    \"\"\"估算能量消耗\"\"\"
    # 简化模型：能量与速度平方成正比
    return sum([v**2 for v in velocity]) * time_step

def check_collision(position, obstacles, safety_margin=1.0):
    \"\"\"检查给定位置是否与障碍物碰撞\"\"\"
    for obstacle in obstacles:
        obs_pos = obstacle['position']
        obs_size = obstacle['size']
        
        # 检查碰撞
        if (abs(position[0] - obs_pos[0]) < obs_size[0]/2 + safety_margin and
            abs(position[1] - obs_pos[1]) < obs_size[1]/2 + safety_margin and
            abs(position[2] - obs_pos[2]) < obs_size[2]/2 + safety_margin):
            return True
    
    return False
"""

    with open(robotics_utils_path, 'w') as f:
        f.write(robotics_utils_content)
    
    # 创建数据处理工具
    data_processor_path = "utils/data_processor.py"
    data_processor_content = """#!/usr/bin/env python3
\"\"\"
数据处理相关的实用工具函数
\"\"\"
import numpy as np
import pandas as pd
import json
import os

def load_trajectory_data(file_path):
    \"\"\"加载轨迹数据\"\"\"
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path).values.tolist()
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_trajectory_data(data, file_path, format='csv'):
    \"\"\"保存轨迹数据\"\"\"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if format == 'csv':
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        df.to_csv(file_path, index=False)
    elif format == 'json':
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

def calculate_path_metrics(path):
    \"\"\"计算路径的各种指标\"\"\"
    if not path or len(path) < 2:
        return {'length': 0, 'avg_altitude': 0, 'max_altitude': 0}
    
    # 计算路径长度
    length = 0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        segment_length = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
        length += segment_length
    
    # 计算高度指标
    altitudes = [p[2] for p in path]
    avg_altitude = sum(altitudes) / len(altitudes)
    max_altitude = max(altitudes)
    
    return {
        'length': length,
        'avg_altitude': avg_altitude,
        'max_altitude': max_altitude
    }

def merge_metrics_data(metrics_files):
    \"\"\"合并多个指标数据文件\"\"\"
    merged_data = []
    
    for file_path in metrics_files:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            merged_data.append(df)
    
    if merged_data:
        return pd.concat(merged_data, ignore_index=True)
    else:
        return pd.DataFrame()
"""

    with open(data_processor_path, 'w') as f:
        f.write(data_processor_content)
    
    print("Utility files created.")

def update_requirements():
    """更新requirements.txt文件"""
    requirements = """# 基础科学计算
numpy==1.23.5
pandas==1.5.3
matplotlib==3.7.1
scipy==1.10.1

# 深度学习
torch==2.0.0
torchvision==0.15.1

# 图处理
networkx==3.0

# 可视化
plotly==5.14.1
seaborn==0.12.2

# 杂项
tqdm==4.65.0
"""

    with open("requirements.txt", 'w') as f:
        f.write(requirements)
    
    print("Updated requirements.txt")

def update_readme():
    """更新README.md文件"""
    readme = """# 无人机城市环境路径规划研究

## 项目概述

本项目实现了多种路径规划算法在城市环境中的无人机路径规划研究，包括传统算法（A*、RRT*）和深度强化学习算法（GAT-DRL、PPO、DQN、TD3）。项目支持两种仿真模式：

1. **快速模拟**：使用Python直接生成模拟数据和可视化结果，适合快速实验和算法比较。
2. **Webots物理仿真**：使用Webots机器人仿真器进行更真实的物理仿真，适合论文发表的严谨研究。

## 环境要求

- Python 3.8+
- Webots R2023a（用于物理仿真模式）
- PyTorch 2.0+（用于DRL算法）

## 安装与设置

1. 克隆仓库
```
git clone https://github.com/username/drone-path-planning.git
cd drone-path-planning
```

2. 安装依赖
```
pip install -r requirements.txt
```

3. 设置Webots环境（对于物理仿真模式）
```
python setup_webots.py
```

## 使用方法

### 快速模拟模式

运行快速模拟以获取算法性能比较：

```
python run_simulation.py --mode mock --algorithm GAT-DRL --scenario mixed --num_waypoints 5
```

### Webots物理仿真模式

运行Webots物理仿真：

```
python run_simulation.py --mode webots --algorithm GAT-DRL --scenario dense --num_waypoints 6
```

## 支持的算法

- **GAT-DRL**：结合图注意力网络的深度强化学习
- **PPO**：近端策略优化
- **DQN**：深度Q网络
- **TD3**：双延迟深度确定性策略梯度
- **A***：传统A*搜索算法
- **RRT***：随机快速扩展树*

## 支持的场景

- **sparse**：稀疏城市环境（10栋建筑物）
- **mixed**：混合城市环境（20栋建筑物）
- **dense**：密集城市环境（30栋建筑物）

## 项目结构

- `algorithms/`: 路径规划算法实现
- `webots/`: Webots仿真相关文件
- `utils/`: 工具函数
- `models/`: 预训练模型
- `results/`: 结果和可视化
- `worlds/`: 场景数据

## 引用格式

如果您在研究中使用了本项目，请按以下格式引用：

```
@article{author2023drone,
  title={Drone Path Planning in Urban Environments: A Comparative Study of DRL and Classical Algorithms},
  author={Author, A.},
  journal={Journal of Intelligent Systems},
  year={2023}
}
```

## 许可证

MIT
"""

    with open("README.md", 'w') as f:
        f.write(readme)
    
    print("Updated README.md")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Set up Webots environment for UAV path planning simulation')
    parser.add_argument('--force', action='store_true', help='Force overwrite existing files')
    args = parser.parse_args()
    
    print("Setting up the UAV path planning simulation environment...")
    
    # 创建目录结构
    create_directory_structure()
    
    # 创建Webots世界文件
    create_webots_world_files()
    
    # 复制控制器文件
    copy_controller_files()
    
    # 创建实用工具文件
    copy_utility_files()
    
    # 更新requirements.txt
    update_requirements()
    
    # 更新README.md
    update_readme()
    
    print("Setup completed successfully!")
    print("You can now run the simulation using:")
    print("  python run_simulation.py --mode webots --algorithm GAT-DRL --scenario mixed")

if __name__ == "__main__":
    main() 