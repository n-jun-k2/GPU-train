cmake_minimum_required(VERSION 3.20)
project(warpshfl)

enable_language(CUDA)

# set macros
add_definitions()

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR}/.. ABSOLUTE)

file(
    GLOB_RECURSE
    CUDA_SRC
    ${PROJECT_SOURCE_DIR}/src/*.cu
    ${PROJECT_SOURCE_DIR}/include/*.cu
)

include_directories(
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PARENT_DIR}/utils
)

message("VERSION:"${CUDA_VERSION})

add_executable(
    ${PROJECT_NAME}
    ${CUDA_SRC}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${LINK_DIRECTORIES}
    ${CUDA_LIBRARIES}
)

target_compile_features(
    ${PROJECT_NAME}
    PRIVATE
    cuda_std_17
)