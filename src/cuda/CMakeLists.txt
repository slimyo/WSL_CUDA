# 子目录 CMakeLists.txt（保持简单，继承根级的 find_package）

add_executable(heat_sim compute-io-overlap.cu)   # 建议改成 .cu

set_target_properties(heat_sim PROPERTIES
    CXX_STANDARD              17
    CXX_STANDARD_REQUIRED     ON
    CUDA_STANDARD             17
    CUDA_STANDARD_REQUIRED    ON
    CUDA_SEPARABLE_COMPILATION ON
)

# 包含路径：当前目录（ach.h） + CUDA include + cccl 子目录（关键！）
target_include_directories(heat_sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}               # /usr/local/cuda/include
    ${CUDAToolkit_INCLUDE_DIRS}/cccl          # 关键！解决 thrust/nv/target not found
    /usr/local/cuda/include
    /usr/local/cuda/include/cccl
    # ${CUDAToolkit_INCLUDE_DIRS}/cccl     # 这行解决 thrust/fill.h not found
    # ${CUDAToolkit_INCLUDE_DIRS}/nv               # 显式加 nv/target 目录（如果在子目录）
)

# 链接（只保留 cudart，CCCL 不需要链接，因为是 header-only 在 device 侧）
target_link_libraries(heat_sim PRIVATE
    CUDA::cudart
    # 删除 CUDA::CCCL 这一行！
)

# 编译选项（保持）
target_compile_options(heat_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=native>
    # 如果你用 clang++ 作为 host compiler，可加：
    # $<$<COMPILE_LANGUAGE:CUDA>:-ccbin clang++>
)

# 可选：定义 CUB_VERSION（保险）
target_compile_definitions(heat_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:CUB_VERSION=320000>
)