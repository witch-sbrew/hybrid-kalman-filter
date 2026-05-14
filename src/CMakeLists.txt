cmake_minimum_required(VERSION 3.4)
project(KFcpu CXX C CUDA)

add_subdirectory(kf-local)

add_subdirectory(kf-gpu)
add_subdirectory(hscheduler)