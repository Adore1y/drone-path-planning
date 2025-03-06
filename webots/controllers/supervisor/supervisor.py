#!/usr/bin/env python3
"""
监督者控制器实现
用于管理仿真环境、建筑物和必经点
"""
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
    """仿真监督者"""
    
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
        """生成环境（建筑物和必经点）"""
        # 加载或生成建筑物数据
        buildings_data = self.load_buildings_data()
        
        # 创建建筑物
        self.create_buildings(buildings_data)
        
        # 创建必经点
        self.create_waypoints()
    
    def load_buildings_data(self):
        """加载建筑物数据"""
        # 尝试从文件加载
        data_file = f"worlds/{self.scenario}_city_data.json"
        
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                data = json.load(f)
                return data.get('buildings', [])
        
        # 如果文件不存在，生成随机建筑物
        return self.generate_random_buildings()
    
    def generate_random_buildings(self):
        """生成随机建筑物"""
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
        """在仿真中创建建筑物"""
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
        """创建必经点"""
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
        """生成必经点"""
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
        """运行监督者控制器"""
        while self.supervisor.step(TIME_STEP) != -1:
            # 持续监控仿真状态
            pass

# 主函数
supervisor = SimulationSupervisor()
supervisor.run()
