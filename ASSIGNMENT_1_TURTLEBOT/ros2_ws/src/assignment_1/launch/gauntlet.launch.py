import os
import signal
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, EmitEvent, DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch_ros.actions import Node

TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')
WORLD_MAP = os.environ.get('WORLD_MAP', 'arms_lab_map_good.yaml')  # Default map file name in maps/ directory


def generate_launch_description():
    # Get the package directory
    pkg_assignment_1 = get_package_share_directory('assignment_1')
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    
    # Get the world file path
    world_file_path = os.path.join(
        pkg_assignment_1,
        'worlds',
        'gauntlet.world'
    )
    
    # Get TurtleBot3 models path
    turtlebot3_models_path = os.path.join(pkg_turtlebot3_gazebo, 'models')
    
    # Set GAZEBO_MODEL_PATH to include TurtleBot3 models
    gazebo_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    if gazebo_model_path:
        gazebo_model_path = f"{turtlebot3_models_path}:{gazebo_model_path}"
    else:
        gazebo_model_path = turtlebot3_models_path

    # Start Gazebo with world file (TurtleBot spawns from world file)
    gazebo_cmd = ExecuteProcess(
        cmd=['gazebo', '--verbose', world_file_path, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen',
        additional_env={'GAZEBO_MODEL_PATH': gazebo_model_path},
        sigterm_timeout='5',
        sigkill_timeout='10'
    )

    # Robot State Publisher
    urdf_content = open(os.path.join(
        get_package_share_directory('turtlebot3_description'),
        'urdf',
        'turtlebot3_burger.urdf'
    )).read()
    
    # Replace ${namespace} with empty string in URDF
    urdf_content = urdf_content.replace('${namespace}', '')
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': urdf_content
        }]
    )

    # Map Corrector - loads map from YAML, corrects orientation, and publishes to /map
    map_file_path = os.path.join(
        pkg_assignment_1,
        'maps',
        WORLD_MAP
    )
    
    map_corrector_config = os.path.join(
        pkg_assignment_1,
        'config',
        'map_corrector_params.yaml'
    )
    
    map_corrector = Node(
        package='assignment_1',
        executable='map_corrector',
        name='map_corrector',
        output='screen',
        parameters=[
            map_corrector_config,
            {'map_yaml_file': map_file_path}
        ]
    )

    # AMCL - provides localization and publishes map->odom transform
    # DISABLED for real robot - using SLAM for localization
    amcl_config = os.path.join(
        pkg_assignment_1,
        'config',
        'amcl_params.yaml'
    )
    
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[amcl_config]
    )

    # Lifecycle Manager for AMCL
    # DISABLED for real robot
    lifecycle_manager_amcl = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_amcl',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['amcl']
        }]
    )

    # Combined Planner Server with config (runs both A* and GVD)
    planner_config = os.path.join(
        pkg_assignment_1,
        'config',
        'planner_params.yaml'
    )
    planner_server = Node(
        package='assignment_1',
        executable='planner_server',
        output='screen',
        name='planner_server',
        parameters=[planner_config]
    )

    # Path Follower with config
    follower_config = os.path.join(
        pkg_assignment_1,
        'config',
        'path_follower_params.yaml'
    )
    path_follower = Node(
        package='assignment_1',
        executable='path_follower',
        output='screen',
        name='path_follower',
        parameters=[follower_config]
    )

    # RViz2 with config
    rviz_config_file = os.path.join(
        pkg_assignment_1,
        'config',
        'rviz_config.rviz'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )
        # Static TF: map -> odom 
    # NOTE: This assumes map and odom are aligned. If the map is flipping or misaligned,
    # you need to either:
    # 1. Run your SLAM node (slam.launch.py) which publishes map->odom transform, OR
    # 2. Manually adjust the transform below to match your robot's starting position
    # For now, using identity transform (no rotation/translation)
    static_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    # Create the launch description
    ld = LaunchDescription()
    # ld.add_action(gazebo_cmd)  # Disabled for real robot
    # ld.add_action(robot_state_publisher)  # Disabled - real robot publishes its own
    ld.add_action(map_corrector)  # Loads map from YAML, corrects orientation, and publishes to /map
    ld.add_action(static_map_to_odom)  # Static map->odom transform
    # ld.add_action(amcl_node)  # Disabled for real robot
    
    # Delay AMCL lifecycle manager to ensure map and TF are available
    # DISABLED for real robot
    # delayed_amcl_lifecycle = TimerAction(
    #     period=2.0,
    #     actions=[lifecycle_manager_amcl]
    # )
    # ld.add_action(delayed_amcl_lifecycle)
    
    # Delay planner and follower to ensure map and TF are available
    delayed_planner = TimerAction(
        period=2.0,  # Reduced delay since no AMCL
        actions=[planner_server]
    )
    
    delayed_follower = TimerAction(
        period=2.0,  # Reduced delay since no AMCL
        actions=[path_follower]
    )
    
    ld.add_action(delayed_planner)
    ld.add_action(delayed_follower)
    ld.add_action(rviz_node)

    return ld

