# 无人机城市环境路径规划研究

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

## 👤 关于作者

作为无人机导航和人工智能交叉领域的研究者，我专注于开发能够应用于实际场景的智能导航系统。本项目是我在研究生申请阶段完成的独立研究项目，展示了我在强化学习、路径规划和仿真验证方面的综合能力。

## 📄 许可证

本项目基于MIT许可证开源 - 详见 [LICENSE](LICENSE) 文件

## 📚 项目资源

- 项目代码库: [GitHub - Adore1y/drone-path-planning](https://github.com/Adore1y/drone-path-planning)





