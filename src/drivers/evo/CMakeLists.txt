AUX_SOURCE_DIRECTORY(. SOURCE_LIST_EVO_MOTORS)
add_library(evo_motors
        STATIC
        ${SOURCE_LIST_EVO_MOTORS}
)
target_include_directories(evo_motors PUBLIC ./ ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(evo_motors PUBLIC ${PUBLIC_DEPENDENCIES} motors_protocol)
