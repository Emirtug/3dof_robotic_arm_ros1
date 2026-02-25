cmake_minimum_required(VERSION 3.0.2)
project(3dof_rrr_robot_arm)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  control_msgs
  actionlib
  actionlib_msgs
  urdf
  xacro
  robot_state_publisher
  joint_state_publisher
  tf
  tf2_ros
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_ros_move_group
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    trajectory_msgs
    control_msgs
    actionlib
    actionlib_msgs
    urdf
    xacro
    robot_state_publisher
    joint_state_publisher
    tf
    tf2_ros
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
    moveit_ros_move_group
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/moveit_direct_sync.py
  scripts/kinematic_3r_location.py
  scripts/fake_joint_trajectory_action_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)

install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
)
