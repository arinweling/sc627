import os
import signal
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch_ros.actions import Node

TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')


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
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': open(os.path.join(
                get_package_share_directory('turtlebot3_description'),
                'urdf',
                'turtlebot3_burger.urdf'
            )).read()
        }]
    )


    # Static TF: map -> odom (anchors odom to map for visualization)
    static_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
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

    # Create the launch description
    ld = LaunchDescription()
    ld.add_action(gazebo_cmd)
    ld.add_action(robot_state_publisher)
    ld.add_action(static_map_to_odom)
    ld.add_action(planner_server)
    ld.add_action(path_follower)
    ld.add_action(rviz_node)

    return ld

