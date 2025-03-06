#!/usr/bin/env python3

"""
障碍物生成脚本
生成多种障碍物以创建复杂的无人机飞行环境
"""

import random
import math

def generate_box(name, x, y, z, width, length, height, color=(1, 0, 0), physics=True):
    """生成一个盒子障碍物的VRML代码"""
    r, g, b = color
    physics_str = "  physics Physics {\n  }\n" if physics else ""
    return f"""DEF {name} Solid {{
  translation {x} {y} {z}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {r} {g} {b}
        roughness 0.5
        metalness 0
      }}
      geometry Box {{
        size {width} {length} {height}
      }}
    }}
  ]
  boundingObject Box {{
    size {width} {length} {height}
  }}
{physics_str}}}
"""

def generate_cylinder(name, x, y, z, radius, height, color=(0, 0, 1), physics=True):
    """生成一个圆柱障碍物的VRML代码"""
    r, g, b = color
    physics_str = "  physics Physics {\n  }\n" if physics else ""
    return f"""DEF {name} Solid {{
  translation {x} {y} {z}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {r} {g} {b}
        roughness 0.5
        metalness 0
      }}
      geometry Cylinder {{
        radius {radius}
        height {height}
      }}
    }}
  ]
  boundingObject Cylinder {{
    radius {radius}
    height {height}
  }}
{physics_str}}}
"""

def generate_wall(name, x, y, z, width, height, orientation=0, color=(0.8, 0.8, 0.8), physics=True):
    """生成一面墙的VRML代码"""
    r, g, b = color
    physics_str = "  physics Physics {\n  }\n" if physics else ""
    # 墙的厚度固定为0.2米
    thickness = 0.2
    return f"""DEF {name} Solid {{
  translation {x} {y} {z}
  rotation 0 0 1 {orientation}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {r} {g} {b}
        roughness 0.5
        metalness 0
      }}
      geometry Box {{
        size {width} {thickness} {height}
      }}
    }}
  ]
  boundingObject Box {{
    size {width} {thickness} {height}
  }}
{physics_str}}}
"""

def create_maze(center_x, center_y, size, complexity=0.7):
    """创建一个简单的迷宫结构"""
    walls = []
    # 外墙
    walls.append(generate_wall(f"WALL_N", center_x, center_y - size/2, 1.5, size, 3.0, 1.57))  # 北墙
    walls.append(generate_wall(f"WALL_S", center_x, center_y + size/2, 1.5, size, 3.0, 1.57))  # 南墙
    walls.append(generate_wall(f"WALL_E", center_x + size/2, center_y, 1.5, size, 3.0, 0))    # 东墙
    walls.append(generate_wall(f"WALL_W", center_x - size/2, center_y, 1.5, size, 3.0, 0))    # 西墙
    
    # 内墙
    num_walls = int(complexity * 10)
    for i in range(num_walls):
        # 随机内墙位置
        if random.random() < 0.5:  # 水平墙
            x = center_x + random.uniform(-size/2 + 2, size/2 - 2)
            y = center_y + random.uniform(-size/2 + 2, size/2 - 2)
            length = random.uniform(2, size - 2)
            walls.append(generate_wall(f"INNER_WALL_H_{i}", x, y, 1.5, length, 3.0, 0))
        else:  # 垂直墙
            x = center_x + random.uniform(-size/2 + 2, size/2 - 2)
            y = center_y + random.uniform(-size/2 + 2, size/2 - 2)
            length = random.uniform(2, size - 2)
            walls.append(generate_wall(f"INNER_WALL_V_{i}", x, y, 1.5, length, 3.0, 1.57))
    
    return walls

def generate_obstacles(num_boxes=10, num_cylinders=5, include_maze=True):
    """生成一组障碍物"""
    obstacles = []
    
    # 创建盒子障碍物
    for i in range(num_boxes):
        name = f"BOX_{i}"
        x = random.uniform(-15, 15)
        y = random.uniform(-15, 15)
        
        # 避免在起飞区域放置障碍物
        if abs(x) < 3 and abs(y) < 3:
            continue
            
        z = random.uniform(0, 3)  # 高度变化
        width = random.uniform(0.5, 2)
        length = random.uniform(0.5, 2)
        height = random.uniform(1, 4)
        
        # 随机颜色
        color = (random.random(), random.random(), random.random())
        
        obstacles.append(generate_box(name, x, y, z, width, length, height, color))
    
    # 创建圆柱障碍物
    for i in range(num_cylinders):
        name = f"CYLINDER_{i}"
        x = random.uniform(-15, 15)
        y = random.uniform(-15, 15)
        
        # 避免在起飞区域放置障碍物
        if abs(x) < 3 and abs(y) < 3:
            continue
            
        z = height / 2  # 圆柱的z坐标是中心点
        radius = random.uniform(0.5, 1.5)
        height = random.uniform(2, 6)
        
        # 随机颜色
        color = (random.random(), random.random(), random.random())
        
        obstacles.append(generate_cylinder(name, x, y, z, radius, height, color))
    
    # 创建迷宫
    if include_maze:
        maze_walls = create_maze(10, 10, 8)
        obstacles.extend(maze_walls)
    
    return obstacles

def add_waypoints(num_points=8):
    """生成路径点"""
    waypoints = []
    
    # 起点
    waypoints.append((0, 0, 1.5))
    
    # 中间点 - 创建一个环形路径
    radius = 15
    for i in range(1, num_points):
        angle = 2 * math.pi * i / (num_points - 1)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = random.uniform(1.5, 3.5)  # 随机高度
        waypoints.append((x, y, z))
    
    return waypoints

def generate_waypoints_code(waypoints):
    """生成路径点的VRML代码"""
    code = "# Waypoints\n"
    
    for i, (x, y, z) in enumerate(waypoints):
        code += f"""DEF WAYPOINT_{i} Solid {{
  translation {x} {y} {z}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0 1 0
        roughness 0.2
        metalness 0.5
        transparency 0.5
      }}
      geometry Sphere {{
        radius 0.3
        subdivision 2
      }}
    }}
  ]
  boundingObject Sphere {{
    radius 0.1
  }}
  physics Physics {{
    density -1
    mass 0.001
  }}
}}\n"""
    
    # 添加路径点数据注释，供控制器读取
    code += "\n# Waypoints data, format: x y z (each line is a waypoint)\n"
    code += "# BEGIN_WAYPOINTS\n"
    for x, y, z in waypoints:
        code += f"# {x} {y} {z}\n"
    code += "# END_WAYPOINTS\n"
    
    return code

# 主函数
if __name__ == "__main__":
    # 生成障碍物
    obstacles = generate_obstacles(num_boxes=15, num_cylinders=8, include_maze=True)
    
    # 生成路径点
    waypoints = add_waypoints(10)
    waypoints_code = generate_waypoints_code(waypoints)
    
    # 将所有障碍物合并到一个字符串中
    all_objects = "\n".join(obstacles)
    
    # 添加路径点
    all_objects += "\n" + waypoints_code
    
    # 将结果写入文件
    with open("obstacles_and_waypoints.wbt", "w") as f:
        f.write(all_objects)
    
    print(f"已生成 {len(obstacles)} 个障碍物和 {len(waypoints)} 个路径点")
    print("结果已保存到 obstacles_and_waypoints.wbt 文件中") 