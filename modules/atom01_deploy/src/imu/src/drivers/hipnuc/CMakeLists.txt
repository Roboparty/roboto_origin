AUX_SOURCE_DIRECTORY(. SOURCE_LIST_HIPNUC_IMU)
add_library(hipnuc_imu
        STATIC
        ${SOURCE_LIST_HIPNUC_IMU}
)
target_include_directories(hipnuc_imu PUBLIC ./ ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(hipnuc_imu PUBLIC ${PUBLIC_DEPENDENCIES} imu_protocol)