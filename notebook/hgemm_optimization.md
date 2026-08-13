# HGEMM (Half Precision GEMM) 优化全流程笔记
> 目标：从基础 WMMA 实现出发，通过多级流水线、双缓冲、Swizzle 等技术，最终达到 cuBLAS 98% 的性能。
> 硬件假设：NVIDIA Ampere Architecture (SM80+, e.g., A100, RTX 3090)。
---
## 1. 优化路线图
这是 HGEMM 性能优化的标准“通关”路线。每一步都是为了解决特定的瓶颈：
| 阶段 | 技术手段 | 解决的核心瓶颈 | 预期提升 |
| :--- | :--- | :--- | :--- |
| **Baseline** | WMMA API | 基础 Tensor Core 利用 | 基准 |
| **Step 1** | **Multi-Stage Pipeline** (cp.async) | 隐藏 Global Memory 访问延迟 | 巨大 (消除 SM 空转) |
| **Step 2** | **Register Double Buffer** | 隐藏 Shared Memory 访问延迟 | 显著 (减少 stall) |
| **Step 3** | **Shared Memory Padding** | 解决 Bank Conflict | 中等 (提升带宽利用率) |
| **Step 4** | **Block Swizzle** | 提高 L2 Cache 命中率 | 微小但关键 (最后冲刺 98%) |
---
## 2. Multi-Stage Pipeline (cp.async)
### 2.1 核心问题：延迟隐藏
在 Baseline 版本中，代码执行流是线性的：
1. `__syncthreads()`
2. 从 Global Memory 拷贝数据到 Shared Memory (高延迟，~300-600 cycles)
3. 从 Shared Memory 加载数据到寄存器
4. 计算
在这个过程中，计算单元 在第 2 步是完全闲置的。
### 2.2 解决方案：软件流水线
我们将内存访问（Producer）和计算（Consumer）重叠。利用 `cp.async` 指令，可以在 SM 进行计算的同时，通过专用的拷贝单元在后台搬运下一块数据。
### 2.3 关键指令：cp.async
*   `cp.async`: 异步发起 Global Memory -> Shared Memory 的拷贝。
*   `cp.async.commit_group`: 将一组 cp.async 操作提交到一个“组”中。
*   `cp.async.wait_group<N>`: 等待第 `N` 个之前的组都完成。
### 2.4 代码逻辑解析 (4-Stage Pipeline)
假设我们有 4 块 Shared Memory 缓冲区 (`stage 0` ~ `stage 3`)。
**阶段一：预热**
刚进入 Kernel 时，没有任何数据，必须先同步加载初始数据。我们一次性触发前 3 个 stage 的加载。
```cpp
// 手动触发前 3 个 stage 的异步拷贝
cp.async(&smem_A[0], &A[...]); cp.async(&smem_B[0], &B[...]);
cp.async.commit_group(); // Group 0
cp.async(&smem_A[1], &A[...]); cp.async(&smem_B[1], &B[...]);
cp.async.commit_group(); // Group 1
cp.async(&smem_A[2], &A[...]); cp.async(&smem_B[2], &B[...]);
cp.async.commit_group(); // Group 2
```
**阶段二：流水线循环**
在主计算循环中，我们需要维护“当前计算的 stage”和“正在加载的 stage”。为了保证数据就绪，我们通常保持 3 个 stage 的差距（即计算 stage `k` 时，确保 stage `k+3` 已经加载完毕）。
```cpp
int stage = 0;
for (int tile_k = 0; tile_k < K_tiles; ++tile_k) {
    // 1. 等待当前 stage 的数据就绪
    // wait_group<3> 意味着：等到当前正在提交的组的前 3 个组都完成。
    // 如果我们维护 4 个 buffer，这确保了数据一定已经写好。
    cp.async.wait_group<3>();
    // 2. 从 smem 加载到寄存器 并计算
    // 这里加载的是 smem[stage]
    load_fragment(smem_A[stage], smem_B[stage]);
    mma_sync(); // 使用寄存器数据进行计算
    // 3. 异步加载下一轮数据
    // 准备放入下一个 stage buffer
    int next_stage = (stage + 3) % 4;
    if (next_stage < K_tiles) { // 边界检查
        cp.async(&smem_A[next_stage], &A[...]);
        cp.async(&smem_B[next_stage], &B[...]);
        cp.async.commit_group();
    }
    // 4. 更新 stage 索引
    stage = (stage + 1) % 4;
}
```
---
## 3. Register Double Buffer (寄存器双缓冲)
### 3.1 问题
即使在 Multi-Stage Pipeline 下，计算核心内部仍有停顿：
```text
Cycle 1: LDG (Lds -> Reg)  (流水线充满)
Cycle 2: MMA (计算)
```
如果指令级并行度（ILP）不足，SM 可能会因为等待数据从 Shared Memory 加载到 Register 而停顿。
### 3.2 解决方案
在 Register 层面也做双缓冲。每个 Warp 维护两份 Accumulator/Fragments：`frag_cur` 和 `frag_next`。
### 3.3 执行流程
```cpp
// 初始化：加载第一块数据到 frag_next
ldmatrix(&frag_next);
for (int k = 1; k < K; ++k) {
    // 1. 交换指针，使得 frag_next 变成当前的 frag_cur
    swap(frag_cur, frag_next);
    
    // 2. 对当前数据进行计算
    mma_sync(acc, frag_cur, ...);
    
    // 3. 异步加载下一块数据到 frag_next
    // 此时计算正在进行，加载由内存单元并行处理
    ldmatrix_async(&frag_next); 
}
// 处理最后一块
mma_sync(acc, frag_next, ...);
```
这样，计算 `k` 和加载 `k+1` 在指令层面完全重叠。
---
## 4. Shared Memory Padding
### 4.1 问题：Bank Conflict
Shared Memory 被分为 32 个 Bank。同一个 Warp 内的线程如果访问同一个 Bank 的不同地址，就会发生 Bank Conflict，导致串行访问。
在 GEMM 中，`B` 矩阵通常是 `K x N` 的布局。
Warp 访问模式通常是：线程 `t` 访问 `B[k][t]`, `B[k][t+32]` 等等。
如果不加 Padding，`B[k][0]` 和 `B[k][32]` 可能会映射到同一个 Bank（取决于维度 `BLOCK_N` 是否是 32 的倍数）。即使 `BLOCK_N` 对齐，跨行访问时也可能因 stride 导致冲突。
### 4.2 解决方案
在 Shared Memory 定义维度上手动加 1（或根据具体的 warp 访问模式加 padding）。
```cpp
// 原始定义
// __shared__ half smem_B[BLOCK_K][BLOCK_N]; 
// 优化定义：增加 Padding，错开 Bank 映射
// 使得每行的起始地址不在同一个 Bank 上
__shared__ half smem_B[BLOCK_K][BLOCK_N + 8]; // 常用 +8 或 +1
```
注意：Padding 会增加 SMEM 占用，需要确保不超出每 SM 有限的 Shared Memory 容量（通常 192KB 或 100KB+ configurable）。
---
## 5. Block Swizzle
### 5.1 问题：L2 Cache & DRAM Thrashing
默认情况下，CUDA Grid 中的 Block 是按行优先顺序分配给 SM 的。
*   Block (0,0) 处理矩阵的左上角。
*   Block (1,0) 处理正下方的块。
这在空间局部性上看起来没问题，但在硬件底层的内存控制器上，会导致大量的 L2 Cache 冲突，或者对同一个 DRAM Bank 产生过大的压力。另外，Grid 边缘的 Block 可能计算量不均（如果 M/N 不能被 BLOCK_SIZE 整除），导致部分 SM 提前空闲。
### 5.2 解决方案：Xor Swizzle
通过对 Block ID `(bid_x, bid_y)` 进行位运算重排，改变 Block 的执行顺序和数据访问模式。
### 5.3 算法示例
利用经典的 Morton Order (Z-order) 或简单的 Xor 模式。
```cpp
uint32_t getSwizzledIndex(uint32_t x, uint32_t y) {
    // 简单的 Xor swizzle 逻辑，使得相邻的 block 在物理内存上不再连续，
    // 从而分散压力，提高 L2 Cache 的复用率。
    // 具体的 mask 数值取决于 Block 的维度大小。
    return (x ^ (y >> 1)); 
}
// Kernel Launch 或 Grid Stride Loop 中使用
int swizzled blockIdx_x = blockIdx.x;
int swizzled blockIdx_y = getSwizzledIndex(blockIdx.x, blockIdx.y);
```
**效果**：虽然数据访问地址没有变，但访问的*时间*和*并发模式*变了，使得 L2 Cache 的命中率大幅提升，从而增加了实际有效的内存带宽。
---
## 6. 性能测量与分析
每完成一个步骤，必须使用 Nsight Compute (`ncu`) 进行测量。
### 6.1 测量命令
```bash
# 关键指标：SM 吞吐、内存吞吐、L2 命中率
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics lts__t_sector_hit_rate.pct \
    ./hgemm
```
### 6.2 瓶颈诊断指南
1.  **SM Throughput 低 (< 50%)**
    *   **Memory Latency**: 检查 `Long Scoreboard` 或 `Memory Dependency`。说明流水线没排满，需要 Multi-Stage 或 Double Buffer。
    *   **Occupancy 低**: 检查 `smsp__occupancy.avg.pct_of_peak_sustained_elapsed`。如果低，可能是每个 Block 占用寄存器/SMEM 太多，需要减小 Tile Size。
2.  **DRAM Throughput 低但 SM 利用率高**
    *   说明算力充分利用，但数据供不上。可能是 L2 Cache Miss 太高（需要 Swizzle）或者 Shared Memory Bandwidth 瓶颈（需要 Padding）。
3.  **Memory Throughput 高，但 TFLOPS 低**
    *   可能是指令发射效率低，或者使用了不合适的数据类型（例如过多的 float32 累加而非 Tensor Core 内部的累加）。
---
## 7. 推荐阅读与参考
1.  **Cutlass 源码**: `cutlass/include/cutlass/gemm/threadblock/` 下的 `multistage_mma_base.h` 是工业级 Multi-Stage Pipeline 的标准实现。
2.  **NVIDIA GTC 2020**: "Optimizing CUDA Graphs and Tensor Cores" (包含大量关于 CP.Async 的细节)。
3.  **kernels/swizzle/**: 目录中关于 Block Swizzle 的实现展示了如何处理不同的 Tile 维度。
4.  **kernels/hgemm/**: 对比 `hgemm_baseline.cu` 和 `hgemm_tma.cu` (H100特性) 或 `hgemm_cpasync.cu` (A100特性) 的代码行数差异，理解软件复杂度的增加。
