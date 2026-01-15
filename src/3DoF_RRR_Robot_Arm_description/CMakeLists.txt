cmake_minimum_required(VERSION 3.8)
project(3DoF_RRR_Robot_Arm_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

# Install URDF files
install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

# Install mesh files
install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS scripts/moveit_direct_sync.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
