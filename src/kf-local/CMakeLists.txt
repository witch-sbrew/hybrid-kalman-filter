cmake_minimum_required(VERSION 3.4)
project(KFcpu)

set(EIGEN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eigen-3.3.0")
set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/external/eigen-3.3.0/build")

add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}"
)
# set(Eigen3_VERSION "3.3.0")
# set(Eigen3_VERSION_MAJOR "3")
# set(Eigen3_VERSION_MINOR "3")
#add_subdirectory(external/eigen-5.0.0/Eigen)
add_subdirectory(external/OpenKF)


add_executable(main main.cpp)

target_link_libraries(main PUBLIC Eigen3::Eigen OpenKF)