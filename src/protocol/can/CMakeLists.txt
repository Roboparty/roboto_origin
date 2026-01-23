AUX_SOURCE_DIRECTORY(. SOURCE_LIST_CAN)
add_library(can
        SHARED
        ${SOURCE_LIST_CAN}
)
target_include_directories(can PUBLIC ./)
target_link_libraries(can PUBLIC ${PUBLIC_DEPENDENCIES})