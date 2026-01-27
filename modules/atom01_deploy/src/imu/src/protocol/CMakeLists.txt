file(GLOB_RECURSE PROTOCOL_SRCS *.cpp)

add_library(imu_protocol
        STATIC
        ${PROTOCOL_SRCS}
)
target_include_directories(imu_protocol PUBLIC ./can ./serial)
target_link_libraries(imu_protocol PUBLIC ${PUBLIC_DEPENDENCIES})