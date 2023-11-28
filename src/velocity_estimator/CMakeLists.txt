cmake_minimum_required(VERSION 3.5)
project(velocity_estimator)


# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)  

add_executable(velocity_estimator_node src/velocity_estimator_node.cpp)
ament_target_dependencies(velocity_estimator_node rclcpp geometry_msgs sensor_msgs)

add_executable(velocity_extractor_node src/velocity_extractor_node.cpp)
ament_target_dependencies(velocity_extractor_node rclcpp geometry_msgs nav_msgs)

# Include directories
target_include_directories(velocity_estimator_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(velocity_estimator_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_include_directories(velocity_extractor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(velocity_extractor_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# Install executables
install(TARGETS
  velocity_estimator_node
  velocity_extractor_node
  DESTINATION lib/${PROJECT_NAME})

# # launch folder
# install(DIRECTORY
#   launch
#   DESTINATION share/${PROJECT_NAME}
# )

ament_package()