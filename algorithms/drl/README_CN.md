# 无人机路径规划DRL框架

本框架为基于Webots仿真环境的无人机路径规划深度强化学习(DRL)智能体提供训练工具。它支持多种先进的DRL算法，并提供标准化接口用于比较不同的方法。

## 功能特点

- 提供类似gym接口的Webots无人机仿真环境包装器
- 多种DRL算法实现：
  - PPO (近端策略优化算法)
  - DQN (深度Q网络)
- 完整的训练流程，包括日志记录和可视化
- 可配置的环境参数和障碍物
- 易于使用的命令行训练脚本
- 模型保存和加载，支持继续训练和评估
- 支持连续动作空间和离散动作空间

## 文件结构

```
algorithms/drl/
├── env_wrapper.py         # Webots环境包装器，提供gym接口
├── ppo_algorithm.py       # PPO算法实现
├── dqn_algorithm.py       # DQN算法实现
├── train.py               # 主训练脚本
├── README.md              # 英文文档
├── README_CN.md           # 中文文档
└── configs/               # 配置文件
    ├── env_config.json    # 环境配置
    ├── ppo_config.json    # PPO超参数
    └── dqn_config.json    # DQN超参数
```

## 安装

确保已安装以下依赖：

```bash
pip install torch numpy gym matplotlib pandas
```

还需要安装并正确配置Webots。环境包装器假设Webots的Python API可用。

## 使用方法

### 基本训练

使用默认设置开始训练：

```bash
# 使用PPO算法
python train.py --algo ppo

# 使用DQN算法
python train.py --algo dqn
```

### 自定义训练

可以使用各种命令行参数自定义训练过程：

```bash
python train.py --algo ppo \
                --env_config configs/env_config.json \
                --algo_config configs/ppo_config.json \
                --timesteps 1000000 \
                --output_dir my_training_results \
                --exp_name my_experiment \
                --headless
```

### 仅评估模式

评估预训练模型而不进行进一步训练：

```bash
python train.py --algo ppo \
                --load_model models/ppo_final.pt \
                --eval_only \
                --eval_episodes 10
```

### 无头模式

在没有显示屏的服务器或机器上进行更快速的训练，使用`--headless`标志：

```bash
python train.py --algo ppo --headless
```

## 配置

### 环境配置

环境可以使用JSON文件进行配置。环境配置示例：

```json
{
  "world_file": "../../webots/worlds/mixed_scenario.wbt",
  "waypoints_file": "../../webots/worlds/waypoints.txt",
  "obstacles_file": "../../webots/worlds/obstacles.txt",
  "log_dir": "flight_data",
  "reward_weights": {
    "waypoint_reached": 100.0,
    "distance_improvement": 1.0,
    "collision": -100.0,
    "action_smoothness": -0.1,
    "energy_efficiency": -0.01
  }
}
```

### 算法配置

PPO和DQN算法可以使用单独的JSON文件进行配置。PPO配置示例：

```json
{
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_ratio": 0.2,
  "policy_learning_rate": 3e-4,
  "value_learning_rate": 1e-3,
  "target_kl": 0.01,
  "entropy_coef": 0.01,
  "hidden_sizes": [256, 256],
  "total_timesteps": 1000000,
  "steps_per_epoch": 2048
}
```

DQN配置示例：

```json
{
  "learning_rate": 1e-4,
  "batch_size": 64,
  "gamma": 0.99,
  "tau": 0.005,
  "buffer_size": 100000,
  "epsilon_start": 1.0,
  "epsilon_end": 0.1,
  "epsilon_decay": 0.995,
  "hidden_sizes": [256, 256],
  "total_timesteps": 500000
}
```

## 训练输出

训练结果在指定的输出目录中组织如下（默认为`training_results`）：

```
training_results/
└── ppo_20240425_123456/
    ├── models/             # 保存的模型检查点
    ├── logs/               # 训练日志和指标
    │   ├── training_metrics_*.csv
    │   ├── episode_metrics_*.csv
    │   └── training_metrics_*.png
    └── configs/            # 保存的配置文件
        ├── args.json
        └── ppo_config.json
```

## 扩展框架

### 添加新算法

要添加新的DRL算法：

1. 创建新的Python文件（例如`sac_algorithm.py`），遵循现有算法文件的模式
2. 实现算法的智能体类和训练器类
3. 更新`train.py`以包含新算法选项

### 自定义奖励函数

可以通过修改`env_wrapper.py`中`WebotsMAVDroneEnv`类的`_calculate_reward`方法来自定义奖励函数。

### 自定义世界创建

您可以创建具有不同障碍物和航点配置的自定义Webots世界。确保在环境配置中正确指定这些文件的路径。

## 获得良好结果的技巧

- 从较小的训练步骤开始（例如100,000），确保一切正常工作
- 监控训练曲线以尽早发现问题
- 对于PPO，调整`gae_lambda`参数来控制偏差和方差之间的权衡
- 对于DQN，调整`epsilon_decay`和`buffer_size`参数以获得更好的探索
- 对于更复杂的环境，使用更大的神经网络（更多隐藏层）
- 考虑对观察进行标准化以提高训练稳定性

## 引用

如果您在研究中使用此框架，请引用我们：

```
@misc{drone-drl-framework,
  author = {Your Name},
  title = {Drone Path Planning DRL Framework},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/drone-drl-framework}}
}
```

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件 