cmake_minimum_required(VERSION 3.8)
project(uav_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Build Gazebo plugin
add_library(gazebo_ros_simple_quad SHARED
  src/gazebo_ros_simple_quad.cpp
)

# Include custom header directory
target_include_directories(gazebo_ros_simple_quad PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link ROS 2 and Gazebo dependencies
ament_target_dependencies(gazebo_ros_simple_quad
  gazebo_dev
  gazebo_ros
  rclcpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

# Install plugin library
install(TARGETS gazebo_ros_simple_quad
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install custom headers
install(DIRECTORY include/
  DESTINATION include
)

# Install package resources
install(
  DIRECTORY uav_description launch rviz_config worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(gazebo_ros_simple_quad)
ament_export_dependencies(
  gazebo_dev
  gazebo_ros
  rclcpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

ament_package()