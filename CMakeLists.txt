cmake_minimum_required(VERSION 3.12)
project(imu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

set(PUBLIC_DEPENDENCIES
    fmt::fmt spdlog::spdlog pthread)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

add_subdirectory(src)

add_library(imu SHARED
  src/imu_driver.cpp
)

target_include_directories(imu
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_link_libraries(imu PUBLIC ${PUBLIC_DEPENDENCIES} serial can hipnuc_imu)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY src/ DESTINATION include FILES_MATCHING PATTERN "*.h")

install(TARGETS imu hipnuc_imu can serial
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

pybind11_add_module(imu_py src/pybind_module.cpp)
target_include_directories(imu_py
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_link_libraries(imu_py PUBLIC imu)

install(TARGETS imu_py
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_libraries(imu)
ament_export_include_directories(include)

ament_package()
