cmake_minimum_required(VERSION 3.8)
project(inference)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Boost COMPONENTS system)
find_package(Eigen3 REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
set(ONNXRUNTIME_ROOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/onnxruntime-linux-x64-1.21.0)
set(CNPY_ROOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/cnpy-linux-x64)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
set(ONNXRUNTIME_ROOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/onnxruntime-linux-aarch64-1.21.0)
set(CNPY_ROOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/cnpy-linux-aarch64)
endif()

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT_DIR}/lib)

set(CNPY_INCLUDE_DIR ${CNPY_ROOT_DIR}/include)
set(CNPY_LIB_DIR ${CNPY_ROOT_DIR}/lib)

set(PUBLIC_DEPENDENCIES
    ${Boost_LIBRARIES} pthread Eigen3::Eigen)

add_executable(${PROJECT_NAME}_node src/inference_node.cpp src/motion_loader.cpp)
target_include_directories(${PROJECT_NAME}_node
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${CNPY_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME}_node PUBLIC ${PUBLIC_DEPENDENCIES} ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so ${CNPY_LIB_DIR}/libcnpy.so)
ament_target_dependencies(${PROJECT_NAME}_node PUBLIC rclcpp sensor_msgs geometry_msgs)
set_target_properties(${PROJECT_NAME}_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" 
)

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
install(FILES ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1 ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.21.0 ${CNPY_LIB_DIR}/libcnpy.so
        DESTINATION lib)

ament_export_dependencies(rclcpp sensor_msgs geometry_msgs)
ament_export_libraries(${PROJECT_NAME})
ament_package()
