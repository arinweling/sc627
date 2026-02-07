# Assignment 1 Week 2 - Path Planning with A* and GVD
## Demo videos present at 
https://drive.google.com/drive/folders/1IKHoOeiIJZ6rUMLL74V-qUrAEhWeYKcm?usp=drive_link


ROS 2 path planning implementation featuring A* and Generalized Voronoi Diagram (GVD) planners with autonomous path following for TurtleBot3 in Gazebo simulation.
## Work Distribution
1. Arin Weling (22b1230) : Integrating path planner with Gazebo and Rviz2 for testing and visualization
2. Saumya Shah (22b1238) : Implemented the Astar algorithm
3. Ayush Prasad (22b0674) : Implemented the GVD algorithm
4. Rishabh Parwal (24b1212) : Tuned PID loops and developed the path planner code
## Overview

This package provides:
- **A* Planner**: Grid-based shortest path planner with robot footprint collision avoidance
- **GVD Planner**: Skeleton-based planner that maximizes clearance from obstacles
- **Path Follower**: Proportional controller for autonomous waypoint navigation
- **Unified Planner Server**: Combined node running both planners simultaneously
- **Gazebo Integration**: Custom gauntlet world with TurtleBot3 Burger simulation
- **RViz Visualization**: Real-time path and skeleton visualization

## Features

- ✅ Dynamic start pose (uses robot's current position via TF)
- ✅ Robot footprint collision checking (configurable radius)
- ✅ Waypoint density reduction (configurable spacing)
- ✅ Real-time path following with proportional control
- ✅ Dual planner comparison (run both simultaneously)
- ✅ Custom map support with configurable YAML

## Dependencies

### ROS 2 Packages
- `rclpy`
- `geometry_msgs`
- `nav_msgs`
- `visualization_msgs`
- `tf2_ros`
- `turtlebot3_gazebo`
- `turtlebot3_description`
- `gazebo_ros`

### Python Packages
Install Python dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.20.0
- Pillow >= 9.0.0
- scipy >= 1.7.0
- scikit-image >= 0.19.0
- PyYAML >= 5.4.0

## Installation

### 1. Set TurtleBot3 Model Environment Variable
```bash
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
```

### 2. Clone and Build
```bash
cd sc627/ASSIGNMENT_1_TURTLEBOT/ros2_ws/src
# Clone this repository (if not already cloned)

colcon build --symlink-install
source install/setup.bash
```

## Usage

### Launch Complete System
Launches Gazebo, planners, path follower, and RViz:
```bash
ros2 launch assignment_1 gauntlet.launch.py
```

### Set Goal in RViz
1. Click **"2D Goal Pose"** tool in RViz
2. Click and drag on the map to set goal position and orientation
3. Both planners will compute paths from robot's current position
4. Path follower will execute the selected path

### Select Which Planner to Follow
By default, the path follower subscribes to A* planner (`/planned_path`). To switch to GVD:

Edit `config/path_follower_params.yaml`:
```yaml
path_topic: /gvd_path  # or /planned_path for A*
```

## Configuration

### Planner Server (`config/planner_params.yaml`)

**Planner Selection:**
```yaml
planner_type: both  # Options: 'astar', 'gvd', or 'both'
```

**Common Parameters:**
```yaml
diagonal_cost: 1.414        # Cost for diagonal moves
straight_cost: 1.0          # Cost for straight moves
robot_radius: 0.12          # Robot radius in meters (includes safety margin)
waypoint_spacing: 0.2       # Minimum distance between waypoints (meters)
map_yaml: my_gauntlet_map.yaml
```

**A* Specific:**
```yaml
astar_path_line_width: 0.15
astar_path_color_r: 1.0     # Red
astar_path_color_g: 0.0
astar_path_color_b: 0.0
```

**GVD Specific:**
```yaml
skeleton_threshold: 2.0     # Min distance from obstacles for skeleton
search_radius: 50           # Search radius for nearest skeleton point
gvd_path_line_width: 0.15
gvd_path_color_r: 0.0       # Green
gvd_path_color_g: 1.0
gvd_path_color_b: 0.0
```

### Path Follower (`config/path_follower_params.yaml`)
```yaml
linear_speed: 0.3           # Max linear velocity (m/s)
angular_speed: 0.5          # Max angular velocity (rad/s)
k_p: 0.5                    # Proportional gain for linear velocity
goal_tolerance: 0.1         # Distance to waypoint to reach (m)
path_topic: /planned_path   # /planned_path (A*) or /gvd_path (GVD)
```

## Topics

### Published
- `/map` - OccupancyGrid map
- `/planned_path` - A* path (nav_msgs/Path)
- `/path_marker` - A* path visualization (Marker)
- `/gvd_path` - GVD path (nav_msgs/Path)
- `/gvd_path_marker` - GVD path visualization (Marker)
- `/gvd_skeleton` - GVD skeleton visualization (Marker)
- `/cmd_vel` - Velocity commands to robot (Twist)

### Subscribed
- `/goal_pose` - Goal position from RViz (PoseStamped)
- `/planned_path` or `/gvd_path` - Path to follow (Path)

### TF Frames
- `map` → `odom` (static transform)
- `odom` → `base_footprint` (from Gazebo)
- `base_footprint` → sensors (from robot model)

## Architecture

```
┌─────────────────┐
│ Planner Server  │
│  - A* Planner   │──→ /planned_path ──┐
│  - GVD Planner  │──→ /gvd_path       │
└─────────────────┘                    ↓
                                  ┌──────────────┐
        TF: map→base_footprint ──→│Path Follower │──→ /cmd_vel ──→ Robot
                                  └──────────────┘
```

## Algorithms

### A* Planner
- **Search**: Grid-based A* with Euclidean heuristic
- **Connectivity**: 8-directional (diagonal + straight)
- **Collision Checking**: Circular robot footprint validation
- **Output**: Shortest collision-free path

### GVD Planner
- **Distance Transform**: Euclidean distance to nearest obstacle
- **Skeletonization**: Morphological skeleton of safe zones
- **Search**: A* restricted to skeleton points
- **Output**: Path maximizing clearance from obstacles

### Path Follower
- **Control**: Proportional control for linear and angular velocity
- **Waypoint Tracking**: Sequential waypoint following
- **Turn-in-Place**: Rotates when angle error > 0.2 rad
- **Completion**: Stops when all waypoints reached

## File Structure

```
assignment_1/
├── assignment_1/
│   ├── astar_planner.py       # Standalone A* planner
│   ├── gvd_planner.py         # Standalone GVD planner
│   ├── planner_server.py      # Combined planner server
│   └── path_follower.py       # Path following controller
├── config/
│   ├── planner_params.yaml    # Planner server configuration
│   ├── astar_params.yaml      # A* standalone config
│   ├── gvd_params.yaml        # GVD standalone config
│   ├── path_follower_params.yaml
│   └── rviz_config.rviz
├── launch/
│   └── gauntlet.launch.py     # Main launch file
├── maps/
│   ├── my_gauntlet_map.pgm    # Occupancy grid image
│   └── my_gauntlet_map.yaml   # Map metadata
├── worlds/
│   └── gauntlet.world         # Gazebo world file
├── requirements.txt
├── setup.py
└── README.md
```

```

## License

TODO: Add license information

## Authors

- arinweling

## Acknowledgments

- TurtleBot3 packages by ROBOTIS
- ROS 2 community
