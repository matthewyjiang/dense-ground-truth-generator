from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('dense_ground_truth_generator')

    # Path to the configuration file
    config_file = os.path.join(pkg_dir, 'config', 'ground_truth_params.yaml')

    return LaunchDescription([
        Node(
            package='dense_ground_truth_generator',
            executable='ground_truth_server',
            name='ground_truth_server',
            output='screen',
            parameters=[config_file],
            emulate_tty=True
        )
    ])
