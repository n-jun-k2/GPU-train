cmake_minimum_required(VERSION 3.10)

project(vulkan-example VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)


find_package(Vulkan REQUIRED)
include_directories(&Vulkan REQUIRED)
pkg_check_modules(Vulkan REQUIRED vulkan)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ./build)
set(CMAKE_LIBRARY_OUTPUT_DRECTORY ./build)

add_executable( vulakn-example
    main.cpp
)

target_link_libraries(
    vulkan-example
    ${Vulkan_LIBRARIES}
)