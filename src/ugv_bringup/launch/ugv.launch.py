import os
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_path,get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    controller_config = os.path.join(get_package_share_directory('ugv_bringup'),'config', 'diffdrive_params.yaml')
    urdf_path = os.path.join(get_package_share_path('ugv_bringup'), 'ugv_description', 'main.urdf.xacro')

    world_arg = DeclareLaunchArgument('world', default_value='train.world')
    world = PathJoinSubstitution([FindPackageShare('ugv_bringup'), 'worlds', LaunchConfiguration('world')])

    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}])
    delayed_node_robot_state_publisher = TimerAction(period=10.0, actions=[node_robot_state_publisher])

    node_joint_state_publisher = Node(
       package='joint_state_publisher',
       executable='joint_state_publisher',
       parameters=[{'use_sim_time': True}])
    delayed_joint_state_publisher = TimerAction(period=10.0, actions=[node_joint_state_publisher])

    node_gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
                    launch_arguments={'world': world}.items(),)
    delayed_node_gazebo = TimerAction(period=10.0, actions=[node_gazebo])
    
    node_spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'diffdrive_gazebo_bot'],output='screen')
    delayed_node_spawn_entity = TimerAction(period=10.0, actions=[node_spawn_entity])

    node_controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[{'robot_description': robot_description}, controller_config])
    delayed_controller_manager = TimerAction(period=10.0, actions=[node_controller_manager])

    node_cont_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_cont", "joint_broad"])
    delayed_diff_spawners = TimerAction(period=12.0, actions=[node_cont_spawner])

    return LaunchDescription([
        world_arg,
        delayed_node_robot_state_publisher,
        delayed_joint_state_publisher,
        delayed_node_gazebo,
        delayed_node_spawn_entity,
        delayed_controller_manager,
        delayed_diff_spawners,
    ])
