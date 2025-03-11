# Drone Simulation Project Structure

## Directory Structure

```
drones/
├── run_experiment.sh                # Main experiment launch script
├── Simulation_Guide.md              # User operation guide
├── Project_Structure.md             # This document
├── webots/
│   ├── worlds/
│   │   └── mixed_scenario.wbt       # Simulation world file
│   └── controllers/
│       └── mavic_python/
│           ├── mavic_python.py      # Basic controller
│           ├── mavic_waypoints_controller.py  # Waypoint navigation controller
│           ├── analyze_flight_data.py   # Basic data analysis script
│           ├── analyze_waypoints.py     # Waypoint analysis script
│           ├── generate_dl_path.py      # Deep learning path generation script
│           ├── compare_paths.py         # Path comparison script
│           └── requirements.txt         # Python dependencies
└── scripts/
    └── generate_obstacles.py        # Obstacle and waypoint generation script
```

## Component Relationships

### 1. Simulation Environment

- **mixed_scenario.wbt**: Webots world file defining the simulation environment, drone, and obstacles
- **generate_obstacles.py**: Generates random obstacles and waypoints, creating the `obstacles_and_waypoints.wbt` file

### 2. Drone Control

- **mavic_python.py**: Basic controller implementing simple takeoff, hovering, and forward movement
- **mavic_waypoints_controller.py**: Advanced controller implementing waypoint navigation and obstacle avoidance

### 3. Data Analysis

- **analyze_flight_data.py**: Analyzes basic flight data, generates simple statistics and charts
- **analyze_waypoints.py**: Analyzes waypoint navigation flight data, including path comparison and performance evaluation
- **generate_dl_path.py**: Generates simulated deep learning path data, supporting multiple model types
- **compare_paths.py**: Compares performance of multiple paths, generating comparative charts and metrics

### 4. User Interface

- **run_experiment.sh**: Main launch script guiding users through the entire experiment process
- **Simulation_Guide.md**: Detailed user operation guide

## Data Flow

1. **Environment Preparation**: `generate_obstacles.py` generates obstacles and waypoints, writes to world file
2. **Simulation Execution**: Webots loads world file, runs controller (basic or waypoint navigation)
3. **Data Collection**: Controller saves flight data as CSV files
4. **Path Generation**: `generate_dl_path.py` generates simulated deep learning path data
5. **Data Analysis**: Analysis scripts process collected data, generate charts and metrics
6. **Result Comparison**: `compare_paths.py` compares performance of different flight paths

## Extension Points

The project is designed with a modular architecture that can be extended in the following areas:

1. **Control Algorithms**: Add new controllers implementing different navigation strategies
2. **Deep Learning Integration**: Replace simulated deep learning paths with actual deep learning algorithms
3. **Environment Complexity**: Add more types of obstacles and environmental conditions
4. **Multi-Drone Coordination**: Extend to multi-drone cooperative tasks
5. **Task Scenarios**: Add specific task scenarios such as search, tracking, or monitoring 