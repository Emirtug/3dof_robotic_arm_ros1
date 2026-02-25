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
  gazebo_ros
  controller_manager
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
    gazebo_ros
    controller_manager
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# Install urdf files
install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)

# Install mesh files
install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
)

# Install scripts
catkin_install_python(PROGRAMS
  scripts/basic_move.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
