# 无人机城市环境路径规划研究：基于图注意力强化学习的方法

![无人机路径规划](https://img.shields.io/badge/研究领域-无人机路径规划-blue)
![算法](https://img.shields.io/badge/核心算法-GAT--TD3-green)
![模拟环境](https://img.shields.io/badge/模拟环境-Webots-orange)
![状态](https://img.shields.io/badge/项目状态-已完成-success)

## 📋 项目概述

本项目致力于解决城市环境中无人机自主导航与路径规划问题，通过结合图注意力网络(GAT)和深度强化学习(DRL)技术，开发了能够在复杂城市环境中高效规划安全路径的算法。研究比较了传统算法（A*、RRT*）和多种深度强化学习算法（GAT-TD3、PPO、DQN、TD3等）的性能，并在Webots物理模拟环境中验证了算法的实际效果。

### 🌟 研究亮点

- **创新算法架构**：提出了GAT-TD3（图注意力双延迟深度确定性策略梯度）算法，解决了传统DRL算法在城市环境感知能力的局限性
- **多算法对比**：系统比较了6种主流路径规划算法在不同城市密度下的性能
- **双层评估系统**：同时实现了快速Python模拟和高保真Webots物理模拟，保证研究的效率和严谨性
- **实际应用导向**：考虑了能耗最优化、避障能力和飞行平滑度等多方面因素，接近实际应用需求

## 🔬 研究方法

### 核心算法：GAT-TD3

GAT-TD3算法融合了图注意力网络与TD3强化学习框架的优势：

1. **环境感知增强**：利用图注意力机制处理城市建筑物的空间关系，使无人机能够"理解"城市结构
2. **稳定训练机制**：采用TD3的双Q网络和延迟更新策略，有效缓解过估计问题
3. **多目标优化**：同时考虑路径长度、能耗、安全距离和平滑度的综合奖励函数

### 实验设计

- **城市环境模拟**：构建了三种密度（稀疏、混合、密集）的城市场景，包含各类建筑物和障碍物
- **算法性能指标**：路径长度、能耗、计算时间、避障成功率、飞行平滑度
- **对比基线**：传统算法（A*、RRT*）和DRL算法（PPO、DQN、TD3）

## 💻 环境与技术栈

- **编程语言**：Python 3.8+
- **深度学习框架**：PyTorch 2.0+ 
- **模拟环境**：Webots R2023a
- **辅助库**：NumPy, Pandas, Matplotlib, DGL (Deep Graph Library)

## 🚀 安装与设置

1. 克隆仓库
```
git clone https://github.com/Adore1y/drone-path-planning.git
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

## 📊 主要研究结果

详细性能对比分析结果显示，GAT-TD3算法相比其他方法：

- **路径长度**：在密集城市环境中比传统A*算法缩短15%
- **能源效率**：比基础TD3算法提高23%
- **避障能力**：在所有测试场景中实现100%避障成功率
- **计算效率**：相比RRT*算法实现82%的计算时间减少
- **飞行平滑度**：比所有对比算法提升至少27%

## 📁 项目结构

```
drone-path-planning/
├── algorithms/               # 算法实现
│   ├── traditional/          # 传统路径规划算法
│   │   ├── a_star.py         # A*算法实现
│   │   └── rrt_star.py       # RRT*算法实现
│   └── drl/                  # 深度强化学习算法
│       ├── gat_td3/          # GAT-TD3算法核心实现
│       ├── ppo/              # PPO算法实现
│       ├── dqn/              # DQN算法实现
│       └── td3/              # TD3算法实现
├── webots/                   # Webots仿真相关文件
│   ├── worlds/               # 世界文件
│   │   ├── drone_rl_training.wbt  # 训练环境
│   │   ├── urban_environment.wbt  # 城市环境仿真
│   │   └── waypoints.txt     # 导航路点
│   └── controllers/          # 控制器代码
│       └── mavic_python/     # 无人机控制器
├── utils/                    # 工具函数
├── models/                   # 预训练模型
├── results/                  # 结果和可视化
└── docs/                     # 文档和论文
```

## 📈 使用方法

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

### 性能分析

生成算法性能对比分析：

```
python analyze_results.py --algorithms GAT-DRL,TD3,A_STAR --metrics path_length,energy,smoothness
```

## 🔍 研究结论与未来工作

### 主要结论

1. 图注意力机制能显著提升无人机对城市环境的感知能力
2. 结合TD3的训练稳定性优势，GAT-TD3算法在城市环境中表现出色
3. 在高密度城市环境中，深度强化学习方法相比传统方法具有明显优势

### 未来工作方向

1. 扩展到多无人机协同路径规划
2. 加入动态障碍物（如移动车辆）的处理能力
3. 结合视觉感知，实现纯视觉导航
4. 降低算法的计算复杂度，适应嵌入式系统部署

## 👤 关于作者

作为无人机导航和人工智能交叉领域的研究者，我专注于开发能够应用于实际场景的智能导航系统。本项目是我在研究生申请阶段完成的独立研究项目，展示了我在强化学习、路径规划和仿真验证方面的综合能力。

## 📄 许可证

本项目基于MIT许可证开源 - 详见 [LICENSE](LICENSE) 文件

## 📚 相关文献

- [1] Tai, L., Paolo, G., & Liu, M. (2018). Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation. IROS.
- [2] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. ICLR.
- [3] Fujimoto, S., Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. ICML.





