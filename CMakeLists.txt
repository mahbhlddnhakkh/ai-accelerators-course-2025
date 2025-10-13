cmake_minimum_required(VERSION 3.22)

project(ai-accelerators-course)

option(ENABLE_CPU "Enable building CPU-based tasks" OFF)
option(ENABLE_CUDA "Enable building CUDA-based tasks" OFF)
option(ENABLE_ASCEND "Enable building Ascend-based tasks" OFF)

add_subdirectory(tasks)
