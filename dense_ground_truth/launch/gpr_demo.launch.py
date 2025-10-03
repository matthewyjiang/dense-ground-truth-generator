from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('dense_ground_truth')

    # Path to the configuration file
    config_file = os.path.join(pkg_dir, 'config', 'ground_truth_params.yaml')

    # Ground truth server node
    ground_truth_server = Node(
        package='dense_ground_truth',
        executable='ground_truth_server',
        name='ground_truth_server',
        output='screen',
        parameters=[config_file],
        emulate_tty=True
    )

    # GPR visualization node (delayed start to ensure server is ready)
    gpr_visualization = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='dense_ground_truth',
                executable='gpr_visualization.py',
                name='gpr_visualization',
                output='screen',
                parameters=[{
                    'num_training_points': 50,
                    'grid_resolution': 50,
                    'area_min_x': 0.0,
                    'area_max_x': 100.0,
                    'area_min_y': 0.0,
                    'area_max_y': 100.0,
                }],
                emulate_tty=True
            )
        ]
    )

    return LaunchDescription([
        ground_truth_server,
        gpr_visualization
    ])
