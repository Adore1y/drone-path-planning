#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Webots无人机监督控制器
负责模拟环境管理、数据记录和与外部学习系统的交互
"""

# 尝试导入Webots控制器，如果不可用，创建模拟版本
try:
    from controller import Supervisor
except ImportError:
    # 创建模拟的Supervisor类用于非Webots环境
    class Supervisor:
        def __init__(self):
            print("警告: 使用模拟的Supervisor类，仅用于测试")
            
        def getFromDef(self, def_name):
            return SimulatedNode(def_name)
            
        def step(self, time_step):
            return 0
            
    class SimulatedNode:
        def __init__(self, def_name):
            self.def_name = def_name
            
        def getField(self, field_name):
            return SimulatedField(field_name)
            
        def getPosition(self):
            return [0, 0, 0]
            
    class SimulatedField:
        def __init__(self, field_name):
            self.field_name = field_name
            
        def getSFVec3f(self):
            return [0, 0, 0]
            
        def setSFVec3f(self, value):
            pass

import sys
import os
import time
import numpy as np
import random

# 添加父目录到Python路径，以便导入其他模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

class UAVSupervisor:
    """无人机监督控制器类"""
    
    def __init__(self):
        """初始化监督控制器"""
        # 创建Supervisor实例
        self.supervisor = Supervisor()
        
        # 获取基本时间步长
        self.time_step = 32  # 默认时间步
        
        # 默认建筑物和边界值（用于非Webots环境）
        self.buildings = []
        self.boundaries = [[-50, 50], [-50, 50]]  # 默认100x100区域
        
        try:
            # 尝试获取时间步（可能在模拟环境中不可用）
            if hasattr(self.supervisor, 'getBasicTimeStep'):
                self.time_step = int(self.supervisor.getBasicTimeStep())
            
            # 获取无人机和目标点的引用
            self.uav_node = self.supervisor.getFromDef("UAV")
            if self.uav_node:
                # 获取无人机的平移和旋转字段
                self.uav_translation = self.uav_node.getField("translation")
            else:
                print("警告: 未找到UAV节点，使用模拟数据")
                self.uav_node = None
                self.uav_translation = None
        except Exception as e:
            print(f"初始化监督控制器时发生错误: {e}")
            print("使用模拟数据继续")
            self.uav_node = None
            self.uav_translation = None
        
        # 初始化目标位置
        self.target_position = [0, 0, 0]
        
        # 实验数据记录
        self.experiment_data = {
            "trajectory": [],
            "timestamps": [],
            "energy_consumption": 0,
            "collisions": 0,
            "completion_time": 0,
            "success": False
        }
        
    def create_target(self):
        """在场景中创建目标点"""
        root_node = self.supervisor.getRoot()
        children_field = root_node.getField("children")
        children_field.importMFNodeFromString(-1, """
        DEF TARGET Solid {
          translation 0 0 0
          children [
            Shape {
              appearance PBRAppearance {
                baseColor 1 0 0
                metalness 0
                roughness 0.3
              }
              geometry Sphere {
                radius 0.5
                subdivision 2
              }
            }
          ]
          name "target"
        }
        """)
        
        self.target_node = self.supervisor.getFromDef("TARGET")
        
    def load_buildings(self, scenario="dense"):
        """
        加载特定场景的建筑物数据
        
        参数:
            scenario: 场景类型，可选值为 'dense', 'sparse', 'mixed'
        
        返回值:
            成功加载返回True，否则返回False
        """
        try:
            if scenario == "dense":
                try:
                    from worlds.dense_city_buildings import BUILDINGS, BOUNDARIES
                except ImportError:
                    print(f"Warning: Could not import dense_city_buildings, using mock data")
                    # 生成密集城市的模拟建筑物数据
                    BUILDINGS = []
                    for i in range(50):  # 密集城市有50个建筑物
                        x = (random.random() - 0.5) * 80
                        z = (random.random() - 0.5) * 80
                        width = random.uniform(5, 15)
                        height = random.uniform(10, 50)
                        length = random.uniform(5, 15)
                        BUILDINGS.append({
                            'position': [x, height/2, z],
                            'width': width,
                            'length': length,
                            'height': height
                        })
                    BOUNDARIES = [-50, -50, 50, 50]
            elif scenario == "sparse":
                try:
                    from worlds.sparse_city_buildings import BUILDINGS, BOUNDARIES
                except ImportError:
                    print(f"Warning: Could not import sparse_city_buildings, using mock data")
                    # 生成稀疏城市的模拟建筑物数据
                    BUILDINGS = []
                    for i in range(15):  # 稀疏城市有15个建筑物
                        x = (random.random() - 0.5) * 120
                        z = (random.random() - 0.5) * 120
                        width = random.uniform(10, 25)
                        height = random.uniform(15, 60)
                        length = random.uniform(10, 25)
                        BUILDINGS.append({
                            'position': [x, height/2, z],
                            'width': width,
                            'length': length,
                            'height': height
                        })
                    BOUNDARIES = [-75, -75, 75, 75]
            else:  # mixed
                try:
                    from worlds.mixed_city_buildings import BUILDINGS, BOUNDARIES
                except ImportError:
                    print(f"Warning: Could not import mixed_city_buildings, using mock data")
                    # 生成混合城市的模拟建筑物数据
                    BUILDINGS = []
                    # 中心区域密集的建筑物
                    for i in range(30):
                        x = (random.random() - 0.5) * 50
                        z = (random.random() - 0.5) * 50
                from worlds.mixed_city_buildings import BUILDINGS, BOUNDARIES
                
            self.buildings = BUILDINGS
            self.boundaries = BOUNDARIES
            return True
        except ImportError:
            print(f"Warning: Could not load buildings for {scenario} scenario")
            return False
            
    def reset(self, start_pos=None, target_pos=None, random_positions=True):
        """
        重置模拟环境
        
        参数:
            start_pos: 无人机的起始位置 [x, y, z]
            target_pos: 目标点位置 [x, y, z]
            random_positions: 是否随机生成起始和目标位置
        """
        # 重置实验数据
        self.experiment_data["trajectory"] = []
        self.experiment_data["timestamps"] = []
        self.experiment_data["energy_consumption"] = 0
        self.experiment_data["collisions"] = 0
        self.experiment_data["completion_time"] = 0
        self.experiment_data["success"] = False
        
        # 设置起始位置
        if start_pos is None:
            if random_positions:
                # 随机生成起始位置，避开建筑物
                start_pos = self.generate_random_position(min_height=1.0, max_height=5.0)
            else:
                # 默认起始位置
                start_pos = [0, 0.5, 0]
        
        # 设置目标位置
        if target_pos is None:
            if random_positions:
                # 随机生成目标位置，避开建筑物，与起始位置保持一定距离
                min_dist = 20.0  # 与起始位置的最小距离
                max_tries = 50   # 最大尝试次数
                
                for _ in range(max_tries):
                    target_pos = self.generate_random_position(min_height=1.0, max_height=20.0)
                    
                    # 检查与起始位置的距离
                    dist = np.sqrt((target_pos[0] - start_pos[0])**2 + 
                                   (target_pos[2] - start_pos[2])**2)
                    
                    if dist >= min_dist:
                        break
                        
                if not target_pos:
                    # 如果未能生成有效的目标位置，使用默认位置
                    target_pos = [30, 5, 30]
            else:
                # 默认目标位置
                target_pos = [30, 5, 30]
        
        # 重置无人机位置
        if self.uav_translation:
            self.uav_translation.setSFVec3f([start_pos[0], start_pos[1], start_pos[2]])
        
        # 重置目标位置
        if self.target_node:
            target_translation = self.target_node.getField("translation")
            target_translation.setSFVec3f([target_pos[0], target_pos[1], target_pos[2]])
        
        self.target_position = target_pos
        
        # 记录初始轨迹点
        self.experiment_data["trajectory"].append(start_pos)
        self.experiment_data["timestamps"].append(self.supervisor.getTime())
        
        # 重置物理引擎
        self.supervisor.simulationResetPhysics()
        
        return start_pos, target_pos
        
    def generate_random_position(self, min_height=0.5, max_height=10.0):
        """
        生成随机有效位置（不与建筑物碰撞）
        
        参数:
            min_height: 最小高度
            max_height: 最大高度
            
        返回:
            position: [x, y, z] 位置
        """
        if not self.buildings:
            # 如果没有建筑物信息，在固定范围内随机生成
            x = np.random.uniform(-50, 50)
            z = np.random.uniform(-50, 50)
            y = np.random.uniform(min_height, max_height)
            return [x, y, z]
        
        max_tries = 100
        
        for _ in range(max_tries):
            # 在边界范围内随机生成位置
            min_x, min_z, max_x, max_z = self.boundaries
            x = np.random.uniform(min_x, max_x)
            z = np.random.uniform(min_z, max_z)
            y = np.random.uniform(min_height, max_height)
            
            # 检查是否与任何建筑物碰撞
            valid_position = True
            for building in self.buildings:
                bx, bz, width, height, length = building
                
                # 检查水平面上是否在建筑物内
                if (bx - width/2 <= x <= bx + width/2) and (bz - length/2 <= z <= bz + length/2):
                    # 检查垂直方向是否与建筑物相交
                    if y <= height:
                        valid_position = False
                        break
            
            if valid_position:
                return [x, y, z]
        
        # 如果无法找到有效位置，返回默认位置
        print("Warning: Could not find valid random position, using default")
        return [0, min_height, 0]
        
    def is_collision_with_buildings(self, position):
        """
        检查给定位置是否与建筑物碰撞
        
        参数:
            position: [x, y, z] 位置
            
        返回:
            collision: 是否碰撞
        """
        if not self.buildings:
            return False
            
        x, y, z = position
        
        for building in self.buildings:
            bx, bz, width, height, length = building
            
            # 检查水平面上是否在建筑物内
            if (bx - width/2 <= x <= bx + width/2) and (bz - length/2 <= z <= bz + length/2):
                # 检查垂直方向是否与建筑物相交
                if y <= height:
                    return True
        
        return False
        
    def run_step(self):
        """
        执行一个模拟步骤
        
        返回:
            done: 是否结束（碰撞或到达目标）
            info: 附加信息
        """
        # 推进模拟
        self.supervisor.step(self.time_step)
        
        # 获取无人机当前位置
        position = self.uav_translation.getSFVec3f() if self.uav_translation else [0, 0, 0]
        
        # 记录轨迹
        self.experiment_data["trajectory"].append(position)
        self.experiment_data["timestamps"].append(self.supervisor.getTime())
        
        # 检查是否与建筑物碰撞
        if self.is_collision_with_buildings(position):
            self.experiment_data["collisions"] += 1
            self.experiment_data["completion_time"] = self.supervisor.getTime()
            return True, {"collision": True, "success": False}
        
        # 检查是否到达目标
        if self.target_position:
            dist_to_target = np.sqrt((position[0] - self.target_position[0])**2 + 
                                     (position[1] - self.target_position[1])**2 + 
                                     (position[2] - self.target_position[2])**2)
            
            if dist_to_target < 1.0:  # 1米以内视为到达目标
                self.experiment_data["completion_time"] = self.supervisor.getTime()
                self.experiment_data["success"] = True
                return True, {"collision": False, "success": True}
        
        # 检查是否超出边界
        if self.boundaries:
            min_x, min_z, max_x, max_z = self.boundaries
            if position[0] < min_x or position[0] > max_x or position[2] < min_z or position[2] > max_z:
                self.experiment_data["completion_time"] = self.supervisor.getTime()
                return True, {"collision": True, "success": False, "boundary": True}
        
        # 检查高度是否过低或过高
        if position[1] < 0.2:  # 过低，地面碰撞
            self.experiment_data["collisions"] += 1
            self.experiment_data["completion_time"] = self.supervisor.getTime()
            return True, {"collision": True, "success": False, "ground": True}
            
        if position[1] > 100:  # 过高，超出安全范围
            self.experiment_data["completion_time"] = self.supervisor.getTime()
            return True, {"collision": False, "success": False, "too_high": True}
        
        return False, {}
        
    def get_experiment_data(self):
        """
        获取实验数据
        
        返回:
            data: 包含轨迹和统计信息的字典
        """
        duration = self.experiment_data["completion_time"] - self.experiment_data["timestamps"][0] if self.experiment_data["completion_time"] else self.supervisor.getTime() - self.experiment_data["timestamps"][0]
        
        # 计算路径长度
        path_length = 0
        for i in range(1, len(self.experiment_data["trajectory"])):
            x1, y1, z1 = self.experiment_data["trajectory"][i-1]
            x2, y2, z2 = self.experiment_data["trajectory"][i]
            segment_length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            path_length += segment_length
        
        # 计算平均高度和高度变化
        heights = [point[1] for point in self.experiment_data["trajectory"]]
        avg_height = np.mean(heights) if heights else 0
        height_var = np.var(heights) if len(heights) > 1 else 0
        
        return {
            "trajectory": self.experiment_data["trajectory"],
            "path_length": path_length,
            "duration": duration,
            "collision_count": self.experiment_data["collisions"],
            "avg_height": avg_height,
            "height_var": height_var
        }
        
    def cleanup(self):
        """清理资源"""
        pass

def main():
    """主函数"""
    supervisor = UAVSupervisor()
    
    # 加载建筑物信息（根据当前世界文件自动选择）
    world_name = supervisor.supervisor.getWorldPath().split('/')[-1]
    scenario = None
    
    if "dense" in world_name:
        scenario = "dense"
    elif "sparse" in world_name:
        scenario = "sparse"
    else:
        scenario = "mixed"
        
    supervisor.load_buildings(scenario)
    
    # 随机设置起始位置和目标位置
    start_pos, target_pos = supervisor.reset(random_positions=True)
    print(f"Start position: {start_pos}")
    print(f"Target position: {target_pos}")
    
    # 主循环
    done = False
    step_count = 0
    
    while supervisor.supervisor.step(supervisor.time_step) != -1:
        done, info = supervisor.run_step()
        step_count += 1
        
        if done:
            print(f"Episode ended after {step_count} steps")
            print(f"Info: {info}")
            
            # 获取实验数据
            data = supervisor.get_experiment_data()
            print(f"Path length: {data['path_length']:.2f} m")
            print(f"Duration: {data['duration']:.2f} s")
            print(f"Average height: {data['avg_height']:.2f} m")
            
            # 重置环境，开始新的episodes
            start_pos, target_pos = supervisor.reset(random_positions=True)
            print(f"New start position: {start_pos}")
            print(f"New target position: {target_pos}")
            step_count = 0
            
    # 清理资源
    supervisor.cleanup()

if __name__ == "__main__":
    main() 