AUX_SOURCE_DIRECTORY(. SOURCE_LIST_DM_MOTORS)
add_library(dm_motors
        STATIC
        ${SOURCE_LIST_DM_MOTORS}
)
target_include_directories(dm_motors PUBLIC ./ ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(dm_motors PUBLIC ${PUBLIC_DEPENDENCIES} motors_protocol)