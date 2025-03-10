# GAT-TD3: 基于图注意力网络和深度强化学习的无人机能量高效路径规划

本项目实现了一种结合图注意力网络(Graph Attention Networks, GAT)和双延迟深度确定性策略梯度(Twin Delayed Deep Deterministic Policy Gradient, TD3)的深度强化学习算法，用于无人机在复杂城市环境中的能量高效路径规划。

## 算法特点

1. **图注意力网络**：利用GAT处理环境的空间结构，捕捉无人机、目标点和障碍物之间的空间关系。
2. **TD3深度强化学习**：采用先进的TD3算法进行策略学习，提高训练稳定性和性能。
3. **能量效率优化**：引入详细的能量消耗模型，优化无人机飞行路径的能量效率。
4. **复杂环境适应性**：专为复杂城市环境设计，能够处理高层建筑、不规则障碍物等挑战。

## 文件结构

- `gat_td3.py`: GAT-TD3算法的核心实现
- `energy_model.py`: 无人机能量消耗模型
- `run_gat_td3.py`: 训练脚本
- `run_gat_td3.sh`: 训练启动脚本
- `configs/gat_td3_config.json`: 算法配置文件

## 安装依赖

确保已安装以下依赖：

```bash
pip install torch numpy gym matplotlib pandas
```

## 使用方法

### 基本训练

```bash
./algorithms/drl/run_gat_td3.sh -t 100000 -e urban
```

### 高级选项

```bash
# 在无头模式下训练50万步
./algorithms/drl/run_gat_td3.sh -t 500000 -e urban -h

# 不使用图表示进行训练（退化为普通TD3）
./algorithms/drl/run_gat_td3.sh -t 100000 -e urban -g

# 不使用能量模型
./algorithms/drl/run_gat_td3.sh -t 100000 -e urban -x

# 评估已训练的模型
./algorithms/drl/run_gat_td3.sh -v -l training_results/gattd3_best_reward
```

### 参数说明

- `-t, --timesteps N`: 训练总步数 (默认: 100000)
- `-e, --env TYPE`: 环境类型 (默认: urban)
- `-c, --config FILE`: GAT-TD3配置文件路径
- `--env-config FILE`: 环境配置文件路径
- `-o, --output DIR`: 输出目录 (默认: training_results)
- `-s, --seed N`: 随机种子
- `-h, --headless`: 使用无头模式运行Webots
- `-n, --no-normalize`: 不对观测进行归一化
- `-g, --no-graph`: 不使用图表示
- `-x, --no-energy`: 不使用能量模型
- `-l, --load FILE`: 加载已训练的模型
- `-v, --eval-only`: 仅进行评估，不训练
- `-p, --episodes N`: 评估时的情节数 (默认: 10)
- `-r, --render`: 渲染评估过程

## 配置文件

可以通过修改`configs/gat_td3_config.json`文件来调整算法参数：

```json
{
    "hidden_sizes": [256, 256],
    "gat_heads": 4,
    "gat_hidden_dim": 128,
    "gat_dropout": 0.2,
    "gat_alpha": 0.2,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "buffer_size": 1000000,
    "min_buffer_size": 1000,
    "batch_size": 128,
    "gamma": 0.99,
    "tau": 0.005,
    "policy_freq": 2,
    "action_noise": 0.1,
    "noise_decay": 0.995,
    "min_noise": 0.05,
    "total_timesteps": 500000,
    "update_freq": 1,
    "save_freq": 10000,
    "log_freq": 1000,
    "eval_freq": 10000,
    "energy_weight": 0.5,
    "use_energy_model": true,
    "max_graph_distance": 50.0,
    "use_graph_representation": true
}
```

## 实验结果

训练结果将保存在`training_results`目录下，包括：

- 训练日志
- 模型权重
- 性能指标图表
- 能量效率分析

## 引用

如果您在研究中使用了本算法，请引用：

```
@article{GAT-DRL,
  title={GAT-DRL: Energy-Efficient UAV Path Planning in Complex Urban Environments via Graph Attention Networks and Deep Reinforcement Learning},
  author={Your Name},
  journal={Your Journal},
  year={2023}
}
``` 