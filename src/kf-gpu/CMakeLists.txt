cmake_minimum_required(VERSION 3.20)

project(kf_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(kalman_cuda_batch main.cu)
target_link_libraries(kalman_cuda_batch PRIVATE CUDA::cudart)

install(TARGETS kalman_cuda_batch EXPORT kf_gpu)
install(EXPORT kf_gpu
    FILE kf_gpu.cmake
    DESTINATION lib/cmake/kf_gpu
)