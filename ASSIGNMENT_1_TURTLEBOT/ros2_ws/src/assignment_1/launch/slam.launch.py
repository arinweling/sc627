import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
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

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Start Gazebo with world file
    gazebo_cmd = ExecuteProcess(
        cmd=['gazebo', '--verbose', world_file_path, 
             '-s', 'libgazebo_ros_init.so', 
             '-s', 'libgazebo_ros_factory.so'],
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
            'use_sim_time': use_sim_time,
            'robot_description': urdf_content
        }]
    )

    # MHD SLAM Node with config
    slam_config = os.path.join(
        pkg_assignment_1,
        'config',
        'slam_params.yaml'
    )
    
    mhd_slam_node = Node(
        package='assignment_1',
        executable='mhd_slam',
        output='screen',
        name='mhd_slam',
        parameters=[slam_config, {'use_sim_time': use_sim_time}]
    )

    # RViz2 with SLAM config
    rviz_config_file = os.path.join(
        pkg_assignment_1,
        'config',
        'slam_rviz.rviz'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        # gazebo_cmd,  # Disabled for real robot
        # robot_state_publisher,  # Disabled - real robot publishes its own
        mhd_slam_node,
        rviz_node,
    ])
