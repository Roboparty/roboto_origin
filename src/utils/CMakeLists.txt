AUX_SOURCE_DIRECTORY(. SOURCE_LIST_UTILS)
add_library(utils
        STATIC
        ${SOURCE_LIST_UTILS}
)
target_include_directories(utils PUBLIC ./ ${CNPY_INCLUDE_DIR})
target_link_libraries(utils PUBLIC ${PUBLIC_DEPENDENCIES})