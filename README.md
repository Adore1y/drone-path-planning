# Urban Drone Path Planning Research

## Project Overview

This project implements various path planning algorithms for drone navigation in urban environments, including traditional algorithms (A*, RRT*) and deep reinforcement learning algorithms (GAT-DRL, PPO, DQN, TD3). The project supports two simulation modes:

1. **Rapid Simulation**: Using Python to directly generate simulation data and visualization results, suitable for quick experiments and algorithm comparisons.
2. **Webots Physical Simulation**: Using the Webots robot simulator for more realistic physical simulation, suitable for rigorous research and publication.

## Requirements

- Python 3.8+
- Webots R2023a (for physical simulation mode)
- PyTorch 2.0+ (for DRL algorithms)

## Installation and Setup

1. Clone the repository
```
git clone https://github.com/Adore1y/drone-path-planning.git
cd drone-path-planning
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Set up Webots environment (for physical simulation mode)
```
python setup_webots.py
```

## Usage

### Rapid Simulation Mode

Run rapid simulation to obtain algorithm performance comparisons:

```
python run_simulation.py --mode mock --algorithm GAT-DRL --scenario mixed --num_waypoints 5
```

### Webots Physical Simulation Mode

Run Webots physical simulation:

```
python run_simulation.py --mode webots --algorithm GAT-DRL --scenario dense --num_waypoints 6
```

## Supported Algorithms

- **GAT-DRL**: Deep Reinforcement Learning with Graph Attention Networks
- **PPO**: Proximal Policy Optimization
- **DQN**: Deep Q-Network
- **TD3**: Twin Delayed Deep Deterministic Policy Gradient
- **A***: Traditional A* Search Algorithm
- **RRT***: Rapidly-exploring Random Tree*

## Supported Scenarios

- **sparse**: Sparse urban environment (10 buildings)
- **mixed**: Mixed urban environment (20 buildings)
- **dense**: Dense urban environment (30 buildings)

## Project Structure

- `algorithms/`: Path planning algorithm implementations
- `webots/`: Webots simulation related files
- `utils/`: Utility functions
- `models/`: Pre-trained models
- `results/`: Results and visualizations
- `worlds/`: Scene data

## 📄 License

This project is open-sourced under the MIT License - see the [LICENSE](LICENSE) file for details

## 📚 Project Resources

- Project Repository: [GitHub - Adore1y/drone-path-planning](https://github.com/Adore1y/drone-path-planning)





