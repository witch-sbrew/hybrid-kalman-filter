cmake_minimum_required(VERSION 3.4)
project(KFSchedule CXX)

FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(cnpy)

add_executable(main hscheduler.cpp)

find_package(OpenMP REQUIRED) 
find_package(CUDAToolkit REQUIRED)

target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX kf_cpu kf_gpu cnpy CUDA::cudart)
target_include_directories(main PRIVATE ${cnpy_SOURCE_DIR})
