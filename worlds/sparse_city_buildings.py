#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
稀疏城市建筑物数据
定义了一个稀疏的城市环境，具有较少的分散建筑物
"""

import random
import numpy as np

# 环境边界 [min_x, min_z, max_x, max_z]
BOUNDARIES = [-75, -75, 75, 75]

# 建筑物数据
# 格式: 每个建筑物是一个元组 (x, z, width, height, length)
BUILDINGS = []

# 生成随机建筑物
random.seed(42)  # 设置随机种子，确保每次生成的结果相同
for i in range(15):  # 稀疏城市有15个建筑物
    x = (random.random() - 0.5) * 120
    z = (random.random() - 0.5) * 120
    width = random.uniform(10, 25)
    height = random.uniform(15, 60)
    length = random.uniform(10, 25)
    BUILDINGS.append((x, z, width, height, length))

# 增加一些特定的大型建筑物
BUILDINGS.extend([
    (0, 0, 30, 50, 30),     # 中心位置的建筑
    (50, 50, 40, 60, 20),
    (-50, -50, 35, 40, 35)
])

if __name__ == "__main__":
    print(f"已生成 {len(BUILDINGS)} 个建筑物")
    print(f"环境边界: {BOUNDARIES}") 