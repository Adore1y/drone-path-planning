#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
密集城市建筑物数据
定义了一个密集的城市环境，具有大量高密度建筑物
"""

import random
import numpy as np

# 环境边界 [min_x, min_z, max_x, max_z]
BOUNDARIES = [-50, -50, 50, 50]

# 建筑物数据
# 格式: 每个建筑物是一个元组 (x, z, width, height, length)
BUILDINGS = []

# 生成随机建筑物
random.seed(42)  # 设置随机种子，确保每次生成的结果相同
for i in range(50):  # 密集城市有50个建筑物
    x = (random.random() - 0.5) * 80
    z = (random.random() - 0.5) * 80
    width = random.uniform(5, 15)
    height = random.uniform(10, 50)
    length = random.uniform(5, 15)
    BUILDINGS.append((x, z, width, height, length))

# 增加一些特定的大型建筑物
BUILDINGS.extend([
    (0, 0, 20, 80, 20),     # 中心位置的高层建筑
    (25, 25, 30, 40, 15),
    (-25, -25, 25, 30, 25)
])

if __name__ == "__main__":
    print(f"已生成 {len(BUILDINGS)} 个建筑物")
    print(f"环境边界: {BOUNDARIES}") 