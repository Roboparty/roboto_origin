AUX_SOURCE_DIRECTORY(. SOURCE_LIST_SERIAL)
add_library(serial
        SHARED
        ${SOURCE_LIST_SERIAL}
)
target_include_directories(serial PUBLIC ./)
target_link_libraries(serial PUBLIC ${PUBLIC_DEPENDENCIES})