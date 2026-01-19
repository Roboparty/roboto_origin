##launch file
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = (
        os.path.join(
            get_package_share_directory("inference"),
            "config",
            "inference.yaml",
        ),
    )

    return LaunchDescription(
        [
            Node(
                package="inference",
                executable="inference_node",
                name="inference_node",
                parameters=[config],
                output="screen",
                # prefix=["xterm -e gdb -ex run --args"],
            ),
        ]
    )
