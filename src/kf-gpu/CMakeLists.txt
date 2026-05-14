cmake_minimum_required(VERSION 3.20)

project(kf_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(kf_gpu STATIC kfgpu.cu)
add_executable(kalman_cuda_batch main.cu)

target_link_libraries(kalman_cuda_batch PUBLIC CUDA::cudart)
target_link_libraries(kf_gpu PUBLIC CUDA::cudart)
set_target_properties(kf_gpu PROPERTIES CUDA_ARCHITECTURES "80")
set_target_properties(kalman_cuda_batch PROPERTIES CUDA_ARCHITECTURES "80")

target_include_directories(kf_gpu PUBLIC 
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include>)

install(TARGETS kf_gpu EXPORT kf_gpu_tar)
install(EXPORT kf_gpu_tar
    FILE kf_gpu.cmake
    DESTINATION lib/cmake/kf_gpu
)