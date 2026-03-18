import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("soarm_description")
    urdf_file = os.path.join(pkg_dir, "urdf", "soarm101.urdf")

    with open(urdf_file, "r") as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="true"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            condition=LaunchConfiguration("use_gui"),
        ),
    ])
