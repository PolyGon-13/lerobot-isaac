"""
Launch file for Issac_sim_6DOF_ARM.

Usage:
    ros2 launch lerobot_ver2 display.launch.py
    ros2 launch lerobot_ver2 display.launch.py namespace:=robot1
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_dir = get_package_share_directory('lerobot_ver2')

    xacro_file = os.path.join(pkg_dir, 'urdf', 'Issac_sim_6DOF_ARM.urdf.xacro')
    rviz_file = os.path.join(pkg_dir, 'rviz', 'display.rviz')
    controllers_file = os.path.join(pkg_dir, 'config', 'ros2_controllers.yaml')

    ns = LaunchConfiguration('namespace')
    prefix = LaunchConfiguration('prefix')

    # Tunable gripper-joint origin (passed straight through to xacro args of the same name).
    g = {k: LaunchConfiguration(k) for k in ('gx', 'gy', 'gz', 'gR', 'gP', 'gY')}

    # Wrap in ParameterValue(value_type=str) so ROS 2 Humble does NOT try to YAML-parse
    # the xacro-generated URDF XML (otherwise launch fails to parse robot_description).
    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file, ' prefix:=', prefix,
            ' gx:=', g['gx'], ' gy:=', g['gy'], ' gz:=', g['gz'],
            ' gR:=', g['gR'], ' gP:=', g['gP'], ' gY:=', g['gY'],
        ]),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace', default_value='',
            description='ROS 2 namespace for topics and nodes',
        ),
        DeclareLaunchArgument(
            'prefix', default_value='',
            description='URDF link/joint name prefix (passed to xacro)',
        ),

        # Tunable gripper-joint origin (J5 -> gripper). Defaults = original exported values.
        DeclareLaunchArgument('gx', default_value='-0.020000', description='gripper joint origin x'),
        DeclareLaunchArgument('gy', default_value='0.020000', description='gripper joint origin y'),
        DeclareLaunchArgument('gz', default_value='0.022210', description='gripper joint origin z'),
        DeclareLaunchArgument('gR', default_value='0.0', description='gripper joint origin roll'),
        DeclareLaunchArgument('gP', default_value='0.0', description='gripper joint origin pitch'),
        DeclareLaunchArgument('gY', default_value='-3.141593', description='gripper joint origin yaw'),

        GroupAction([
            PushRosNamespace(ns),

            # Robot state publisher
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                parameters=[{'robot_description': robot_description}],
                output='screen',
            ),

            # ros2_control controller manager using generated mock hardware
            Node(
                package='controller_manager',
                executable='ros2_control_node',
                name='controller_manager',
                parameters=[{'robot_description': robot_description}, controllers_file],
                output='screen',
            ),

            Node(
                package='controller_manager',
                executable='spawner',
                name='spawn_joint_state_broadcaster',
                arguments=['joint_state_broadcaster', '--controller-manager', 'controller_manager'],
                output='screen',
            ),

            Node(
                package='controller_manager',
                executable='spawner',
                name='spawn_position_controller',
                arguments=['position_controller', '--controller-manager', 'controller_manager'],
                output='screen',
            ),

            Node(
                package='controller_manager',
                executable='spawner',
                name='spawn_velocity_controller',
                arguments=['velocity_controller', '--controller-manager', 'controller_manager'],
                output='screen',
            ),

            # RViz2
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_file],
                output='screen',
            ),
        ]),
    ])
