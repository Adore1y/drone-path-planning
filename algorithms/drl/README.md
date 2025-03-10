# Drone Path Planning DRL Framework

This framework provides tools for training deep reinforcement learning (DRL) agents for drone path planning in the Webots simulation environment. It supports multiple state-of-the-art DRL algorithms and provides a standardized interface for comparing different approaches.

## Features

- Environment wrapper for Webots drone simulation with a gym-like interface
- Multiple DRL algorithm implementations:
  - PPO (Proximal Policy Optimization)
  - DQN (Deep Q-Network)
- Comprehensive training pipeline with logging and visualization
- Configurable environment parameters and obstacles
- Easy-to-use training script with command-line arguments
- Model saving and loading for continued training and evaluation
- Support for both continuous and discrete action spaces

## File Structure

```
algorithms/drl/
├── env_wrapper.py         # Webots environment wrapper with gym interface
├── ppo_algorithm.py       # PPO algorithm implementation
├── dqn_algorithm.py       # DQN algorithm implementation
├── train.py               # Main training script
├── README.md              # Documentation
└── configs/               # Configuration files
    ├── env_config.json    # Environment configuration
    ├── ppo_config.json    # PPO hyperparameters
    └── dqn_config.json    # DQN hyperparameters
```

## Installation

Ensure that you have the following dependencies installed:

```bash
pip install torch numpy gym matplotlib pandas
```

Webots should also be installed and properly configured. The environment wrapper assumes that the Webots Python API is available.

## Usage

### Basic Training

To start training with default settings:

```bash
# For PPO algorithm
python train.py --algo ppo

# For DQN algorithm
python train.py --algo dqn
```

### Custom Training

You can customize the training process with various command-line arguments:

```bash
python train.py --algo ppo \
                --env_config configs/env_config.json \
                --algo_config configs/ppo_config.json \
                --timesteps 1000000 \
                --output_dir my_training_results \
                --exp_name my_experiment \
                --headless
```

### Evaluation Only

To evaluate a pre-trained model without further training:

```bash
python train.py --algo ppo \
                --load_model models/ppo_final.pt \
                --eval_only \
                --eval_episodes 10
```

### Headless Mode

For faster training on a server or machine without a display, use the `--headless` flag:

```bash
python train.py --algo ppo --headless
```

## Configuration

### Environment Configuration

The environment can be configured with a JSON file. Example environment configuration:

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

### Algorithm Configuration

PPO and DQN algorithms can be configured with separate JSON files. Example PPO configuration:

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

Example DQN configuration:

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

## Training Outputs

Training results are organized in the specified output directory (or `training_results` by default) as follows:

```
training_results/
└── ppo_20240425_123456/
    ├── models/             # Saved model checkpoints
    ├── logs/               # Training logs and metrics
    │   ├── training_metrics_*.csv
    │   ├── episode_metrics_*.csv
    │   └── training_metrics_*.png
    └── configs/            # Saved configuration files
        ├── args.json
        └── ppo_config.json
```

## Extending the Framework

### Adding New Algorithms

To add a new DRL algorithm:

1. Create a new Python file (e.g., `sac_algorithm.py`) following the pattern of existing algorithm files
2. Implement the algorithm's agent class and trainer class
3. Update `train.py` to include the new algorithm option

### Custom Reward Functions

The reward function can be customized in `env_wrapper.py` by modifying the `_calculate_reward` method in the `WebotsMAVDroneEnv` class.

### Custom World Creation

You can create custom Webots worlds with different obstacles and waypoint configurations. Ensure the paths to these files are correctly specified in the environment configuration.

## Tips for Good Results

- Start with smaller training steps (e.g., 100,000) to make sure everything is working
- Monitor the training curves to detect issues early
- For PPO, adjust the `gae_lambda` parameter to control the trade-off between bias and variance
- For DQN, tune the `epsilon_decay` and `buffer_size` parameters for better exploration
- Use larger neural networks (more hidden layers) for more complex environments
- Consider normalizing observations for better training stability

## Citation

If you use this framework in your research, please cite us:

```
@misc{drone-drl-framework,
  author = {Your Name},
  title = {Drone Path Planning DRL Framework},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/drone-drl-framework}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 