cmake_minimum_required(VERSION 3.22...3.30)

project(CudaProject LANGUAGES CXX CUDA)

# 全局标准
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD         17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找 CUDA（放根级）
find_package(CUDAToolkit 13.2 REQUIRED)

# 如果你坚持用 clang++，可以在这里设置（但测试时建议先用 g++）
# set(CMAKE_CXX_COMPILER clang++)

add_subdirectory(src/cuda)
add_subdirectory(src/hello)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)