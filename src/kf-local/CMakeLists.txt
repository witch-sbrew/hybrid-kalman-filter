cmake_minimum_required(VERSION 3.4)
project(KFcpu)

# Need to give it the raw eigen lib with the cmake files I added or KF doesn't work!!
set(EIGEN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/eigen-3.3.0")
set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/eigen-3.3.0/build")

find_package(OpenMP REQUIRED) 
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
     INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}"
 )

include(FetchContent)

FetchContent_Declare(
  OpenKF
  GIT_REPOSITORY https://github.com/Al-khwarizmi-780/OpenKF.git
  GIT_TAG      main
)

FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG        master
)

FetchContent_MakeAvailable(OpenKF cnpy)

# set_target_properties(Eigen3::Eigen PROPERTIES
#     INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}"
# )
# # set(Eigen3_VERSION "3.3.0")
# # set(Eigen3_VERSION_MAJOR "3")
# # set(Eigen3_VERSION_MINOR "3")
# #add_subdirectory(external/eigen-5.0.0/Eigen)
# add_subdirectory(external/OpenKF)


add_executable(main main.cpp)
add_executable(mainP mainP.cpp)

target_link_libraries(main PUBLIC Eigen3::Eigen OpenKF cnpy)
target_link_libraries(mainP PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen OpenKF cnpy)

target_include_directories(main PUBLIC ${cnpy_SOURCE_DIR})
target_include_directories(mainP PUBLIC 
$<BUILD_INTERFACE:${cnpy_SOURCE_DIR}>
$<INSTALL_INTERFACE:include>)

install(TARGETS mainP EXPORT kf_cpu)
install(EXPORT kf_cpu
    FILE kf_cpu.cmake
    DESTINATION lib/cmake/kf_cpu
)