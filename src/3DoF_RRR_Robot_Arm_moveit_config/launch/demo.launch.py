import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except:
        return {}


def generate_launch_description():
    # Launch arguments
    use_gui = LaunchConfiguration('use_gui', default='false')
    
    # Paths
    moveit_config_pkg = '3DoF_RRR_Robot_Arm_moveit_config'
    robot_desc_pkg = '3DoF_RRR_Robot_Arm_description'
    
    moveit_config_path = get_package_share_directory(moveit_config_pkg)
    robot_desc_path = get_package_share_directory(robot_desc_pkg)
    
    # URDF file
    urdf_file = os.path.join(robot_desc_path, 'urdf', '3DoF_RRR_Robot_Arm.urdf')
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()
    
    robot_description = {'robot_description': robot_description_content}
    
    # SRDF file
    srdf_file = os.path.join(moveit_config_path, 'config', '3DoF_RRR_Robot_Arm.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic_content = f.read()
    
    robot_description_semantic = {'robot_description_semantic': robot_description_semantic_content}
    
    # Kinematics
    kinematics_yaml = load_yaml(moveit_config_pkg, 'config/kinematics.yaml')
    robot_description_kinematics = {'robot_description_kinematics': kinematics_yaml}
    
    # Joint limits
    joint_limits_yaml = load_yaml(moveit_config_pkg, 'config/joint_limits.yaml')
    robot_description_planning = {'robot_description_planning': joint_limits_yaml}
    
    # MoveIt Controllers
    moveit_controllers_yaml = load_yaml(moveit_config_pkg, 'config/moveit_controllers.yaml')
    moveit_controllers = {'moveit_controller_manager': moveit_controllers_yaml.get('moveit_controller_manager', 'moveit_simple_controller_manager/MoveItSimpleControllerManager')}
    moveit_controllers.update(moveit_controllers_yaml)
    
    # OMPL Planning
    ompl_planning_yaml = load_yaml(moveit_config_pkg, 'config/ompl_planning.yaml')
    
    # Convert YAML string adapters to arrays (YAML >- format creates multiline strings)
    if ompl_planning_yaml:
        if 'request_adapters' in ompl_planning_yaml and isinstance(ompl_planning_yaml['request_adapters'], str):
            # Split multiline string into array
            ompl_planning_yaml['request_adapters'] = [
                adapter.strip() 
                for line in ompl_planning_yaml['request_adapters'].split('\n') 
                for adapter in line.split() 
                if adapter.strip()
            ]
        if 'response_adapters' in ompl_planning_yaml and isinstance(ompl_planning_yaml['response_adapters'], str):
            # Split multiline string into array
            ompl_planning_yaml['response_adapters'] = [
                adapter.strip() 
                for line in ompl_planning_yaml['response_adapters'].split('\n') 
                for adapter in line.split() 
                if adapter.strip()
            ]
    
    ompl_planning_pipeline_config = {
        'move_group': {
            'planning_plugins': ['ompl_interface/OMPLPlanner'],
            'request_adapters': [
                'default_planning_request_adapters/ResolveConstraintFrames',
                'default_planning_request_adapters/ValidateWorkspaceBounds',
                'default_planning_request_adapters/CheckStartStateBounds',
                'default_planning_request_adapters/CheckStartStateCollision',
            ],
            'response_adapters': [
                'default_planning_response_adapters/AddTimeOptimalParameterization',
                'default_planning_response_adapters/ValidateSolution',
                'default_planning_response_adapters/DisplayMotionPath',
            ],
        }
    }
    if ompl_planning_yaml:
        ompl_planning_pipeline_config['move_group'].update(ompl_planning_yaml)
    
    # Trajectory execution - We use direct sync script for motor control, but fake controller allows Plan & Execute to work
    trajectory_execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }
    
    # Planning scene monitor
    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }
    
    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )
    
    # Joint State Publisher GUI (optional - disabled by default to avoid conflict with fake action server)
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
        condition=IfCondition(use_gui)
    )
    
    # Move Group Node
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            ompl_planning_pipeline_config,
            trajectory_execution,
            planning_scene_monitor_parameters,
            moveit_controllers,
            {'use_sim_time': False},
        ]
    )
    
    # Fake Joint Trajectory Action Server (allows Plan & Execute to work)
    fake_action_server_node = Node(
        package='3DoF_RRR_Robot_Arm_moveit_config',
        executable='fake_joint_trajectory_action_server.py',
        name='fake_joint_trajectory_action_server',
        output='screen'
    )
    
    # RViz (optional - will start without config if file doesn't exist)
    rviz_config_file = os.path.join(moveit_config_path, 'config', 'moveit.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file] if os.path.exists(rviz_config_file) else [],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
        ]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('use_gui', default_value='false', description='Enable joint state publisher GUI'),
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        fake_action_server_node,
        move_group_node,
        rviz_node,
    ])
