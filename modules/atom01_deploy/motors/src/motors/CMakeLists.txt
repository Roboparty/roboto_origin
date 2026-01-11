cmake_minimum_required(VERSION 3.8)
project(motors)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Boost COMPONENTS system)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ControlMotor.srv"
  "srv/ResetMotors.srv"
  "srv/ReadMotors.srv"
  "srv/SetZeros.srv"
  "srv/ClearErrors.srv"
)

add_library(dm_motors SHARED 
  src/dm_motor_driver.cpp
  src/motor_driver.cpp
  src/SocketCAN.cpp
)
set(PUBLIC_DEPENDENCIES
    fmt::fmt spdlog::spdlog ${Boost_LIBRARIES} pthread Eigen3::Eigen)
target_include_directories(dm_motors
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(dm_motors PUBLIC ${PUBLIC_DEPENDENCIES})

add_executable(${PROJECT_NAME}_node 
                src/motors_node.cpp
                src/close_chain_mapping.cpp)
target_include_directories(${PROJECT_NAME}_node
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}_node PUBLIC ${PUBLIC_DEPENDENCIES} dm_motors)
ament_target_dependencies(${PROJECT_NAME}_node PUBLIC rclcpp sensor_msgs rosidl_default_runtime)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_node PUBLIC "${cpp_typesupport_target}")

set_target_properties(${PROJECT_NAME}_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

install(TARGETS dm_motors
  LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rclcpp sensor_msgs rosidl_default_runtime rosidl_default_generators)
ament_export_libraries(dm_motors)
ament_package()
