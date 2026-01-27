file(GLOB_RECURSE PROTOCOL_SRCS *.cpp)

add_library(motors_protocol
        STATIC
        ${PROTOCOL_SRCS}
)
target_include_directories(motors_protocol PUBLIC ./can)
target_link_libraries(motors_protocol PUBLIC ${PUBLIC_DEPENDENCIES})