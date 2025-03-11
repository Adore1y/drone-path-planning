# Webots Drone Simulation Guide

## 1. Running the Simulation

To start the Webots simulation environment, run the following command in the terminal:

```bash
open -a Webots.app "$(pwd)/webots/worlds/mixed_scenario.wbt"
```

This will open Webots and load our created simulation world.

## 2. Simulation Controls

After Webots starts:

1. Click the **Play** button (▶️) at the top of the interface to start the simulation
2. Use the **Pause** button (⏸) to pause the simulation
3. Use the **Speed up** button to increase simulation speed
4. Use the **Reset** button (⟳) to restart the simulation

## 3. Controller Selection and Functions

We provide two controllers:

### Basic Controller (mavic_python.py)
- Simple three-state control: takeoff, hover, and forward movement
- Basic data recording functionality
- No waypoint navigation functionality

To use this controller, set the robot's controller to "mavic_python".

### Waypoint Navigation Controller (mavic_waypoints_controller.py)
- Navigation based on preset waypoints
- PID controller for precise navigation
- Simple obstacle avoidance functionality
- Comprehensive data recording for subsequent analysis

To use this controller, set the robot's controller to "mavic_waypoints_controller" (default setting).

## 4. Simulation Data Collection

When the simulation runs, the controller automatically collects the following data:
- Drone position (X, Y, Z)
- Drone angles (roll, pitch, yaw)
- Drone velocity
- Motor speeds
- Flight state
- Target waypoint information (waypoint navigation controller)
- Obstacle avoidance status (waypoint navigation controller)

This data is saved in CSV files in the `webots/controllers/mavic_python/flight_data/` directory. Each simulation run generates a new data file with a timestamp.

## 5. Data Analysis

### Basic Data Analysis

After running the basic controller, you can use the analysis script to visualize the data:

```bash
cd webots/controllers/mavic_python && python analyze_flight_data.py
```

### Waypoint Navigation Analysis and Comparison

After running the waypoint navigation controller, use the waypoint analysis script for advanced analysis:

```bash
cd webots/controllers/mavic_python && python analyze_waypoints.py
```

This script provides the following analyses:
- 3D comparison of actual path versus ideal path
- Flight path deviation statistics
- Flight speed and altitude analysis
- Comparison with simulated deep learning paths
- Performance metrics analysis (path length, smoothness, obstacle avoidance count, etc.)

All analysis results and charts are saved in the `webots/controllers/mavic_python/flight_data/comparison_results/` directory.

## 6. Modifying the Simulation

To modify the simulation environment or drone behavior:

- Edit the `webots/worlds/mixed_scenario.wbt` file to change the simulation environment (add/remove objects, modify terrain, etc.)
- Edit the `scripts/generate_obstacles.py` to regenerate random obstacles and waypoints
- Edit the controller files to change the drone's control logic:
  - `webots/controllers/mavic_python/mavic_python.py` - basic controller
  - `webots/controllers/mavic_python/mavic_waypoints_controller.py` - waypoint navigation controller

## 7. Waypoint Definition

Waypoints are defined in the world file in the following format:

```
# BEGIN_WAYPOINTS
# x y z
# x y z
# ...
# END_WAYPOINTS
```

You can manually edit these waypoints or use the `scripts/generate_obstacles.py` script to generate new waypoints.

## 8. Common Issues

1. **Issue**: Simulation runs slowly
   **Solution**: Reduce rendering quality or increase simulation speed

2. **Issue**: Drone is unstable or crashes
   **Solution**: Adjust PID parameters or base throttle value in the controller code

3. **Issue**: Analysis scripts throw errors
   **Solution**: Ensure required Python libraries are installed
   ```bash
   pip install -r webots/controllers/mavic_python/requirements.txt
   ```

4. **Issue**: Drone cannot reach certain waypoints
   **Solution**: Check if waypoints are blocked by obstacles or adjust waypoint coordinates

## 9. Extended Functions

Here are some functions that can be extended:

- Implement more complex obstacle avoidance algorithms
- Integrate real deep learning algorithms for path planning
- Add multi-drone collaborative operation functions
- Implement dynamic waypoint re-planning
- Add environmental disturbance factors such as wind effects 