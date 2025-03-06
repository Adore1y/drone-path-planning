#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合城市建筑物数据
定义了一个混合密度的城市环境，中心区域密集，外围区域稀疏
"""

import random
import numpy as np

# 环境边界 [min_x, min_z, max_x, max_z]
BOUNDARIES = [-60, -60, 60, 60]

# 建筑物数据
# 格式: 每个建筑物是一个元组 (x, z, width, height, length)
BUILDINGS = []

# 设置随机种子，确保每次生成的结果相同
random.seed(42)

# 生成中心区域密集的建筑物
for i in range(30):
    # 在中心区域生成较小密集的建筑物
    r = random.uniform(0, 30)
    theta = random.uniform(0, 2 * np.pi)
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    width = random.uniform(5, 15)
    height = random.uniform(10, 50)
    length = random.uniform(5, 15)
    BUILDINGS.append((x, z, width, height, length))

# 生成外围区域稀疏的建筑物
for i in range(10):
    # 在外围区域生成较大分散的建筑物
    r = random.uniform(30, 60)
    theta = random.uniform(0, 2 * np.pi)
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    width = random.uniform(10, 25)
    height = random.uniform(15, 60)
    length = random.uniform(10, 25)
    BUILDINGS.append((x, z, width, height, length))

# 增加一些特定的标志性建筑物
BUILDINGS.extend([
    (0, 0, 25, 70, 25),  # 中心位置的高层建筑
    (40, 40, 35, 30, 35),  # 外围的购物中心
    (-35, 35, 30, 40, 20)  # 外围的办公园区
])

if __name__ == "__main__":
    print(f"已生成 {len(BUILDINGS)} 个建筑物")
    print(f"环境边界: {BOUNDARIES}") 