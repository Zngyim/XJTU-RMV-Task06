from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    camera_node = Node(
        package='board_detection',
        executable='detection_node',
        name='boarddetection_node',
        output='screen',
        parameters=[
            '/home/zngyim/Desktop/XJTU-RMV-Task06/src/board_detection/config/params.yaml'
        ]
    )

    rviz = ExecuteProcess(
        cmd=['rviz2', '-d', '/home/zngyim/Desktop/XJTU-RMV-Task06/src/board_detection/config/detection.rviz'],
        output='screen'
    )

    # 当相机节点退出时，自动关闭 RViz
    exit_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=camera_node,
            on_exit=[rviz]
        )
    )

    return LaunchDescription([camera_node, rviz, exit_handler])
