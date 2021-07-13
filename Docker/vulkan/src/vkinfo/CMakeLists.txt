cmake_minimum_required(VERSION 3.16)
project(vkinfo)

set(VULKAN_LIB_LIST "libvulkan.so")

# Set macros
add_definitions(
)

# project
file(
    GLOB_RECURSE
    SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/include/*.cpp
)

# libs
include_directories(
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/include
)
link_directories(
    ${PROJECT_SOURCE_DIR}/libs
)

add_executable(
    ${PROJECT_NAME} 
    ${SOURCES}
)
target_link_libraries(
    ${PROJECT_NAME} 
    ${LINK_DIRECTORIES}
    ${VULKAN_LIB_LIST}
)
target_compile_features(
    ${PROJECT_NAME} PRIVATE
    cxx_std_17
)