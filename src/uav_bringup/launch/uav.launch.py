import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import (
    get_package_share_path,
    get_package_share_directory,
)


def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_path('uav_bringup'), 'uav_description', 'main.urdf.xacro'
    )

    world_arg = DeclareLaunchArgument('world', default_value='train.world')
    world = PathJoinSubstitution(
        [FindPackageShare('uav_bringup'), 'worlds', LaunchConfiguration('world')]
    )

    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': True}],
        output='screen',
    )

    node_joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    node_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={'world': world}.items(),
    )

    # --- KEY FIX ---
    # 1. Delay the spawn until gzserver has loaded libgazebo_ros_factory.so
    # 2. Add -timeout so it waits for the service instead of failing immediately
    # 3. Use absolute topic name /robot_description
    node_spawn_entity = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-topic', '/robot_description',
                    '-entity', 'drone_gazebo',
                    '-timeout', '120.0',
                ],
                output='screen',
            )
        ],
    )

    return LaunchDescription([
        world_arg,
        node_robot_state_publisher,
        node_joint_state_publisher,
        node_gazebo,
        node_spawn_entity,
    ])