# Shared Memory Bank Conflict 学习笔记 (优化版)
> 对应文件: `src/Puzzle/bank_conflict.cu`
> 前置知识: Puzzle 8/11 (shared memory reduce), `reduce_warp_learning.md`
> 面试重点: Bank 冲突的产生原因、Padding 原理、如何判断
---
## 1. Shared Memory Bank 结构
每个 SM 的 Shared Memory 被物理划分为 **32 个 Bank**，每个 Bank 的宽度为 **4 Bytes (32 bits)**。这意味着每个 Bank 每个周期能服务一个 32-bit 的访问请求。
```
物理 Bank 布局:
Bank:      0       1       2      ...     31
         [4B]    [4B]    [4B]   ...    [4B]
         [4B]    [4B]    [4B]   ...    [4B]
         [4B]    [4B]    [4B]   ...    [4B]
         ... (更多行)
```
**地址映射公式:**
CUDA 将线性字节地址映射到 Bank 的方式如下：
```cuda
bank_id = (byte_address / 4) % 32;
// 等价于: bank_id = (word_address) % 32;  (假设以 4-byte 为单位寻址)
```
**连续地址特性:**
由于取模运算的存在，**连续的 32 个 4-byte 元素会完美地映射到 32 个不同的 Bank 上**，下一个元素绕回 Bank 0。
```cpp
float data[32];
// data[0] -> word_addr 0  -> Bank 0
// data[1] -> word_addr 1  -> Bank 1
// ...
// data[31]-> word_addr 31 -> Bank 31
// data[32]-> word_addr 32 -> Bank 0 (循环)
```
---
## 2. Bank Conflict 的定义
**什么是 Bank Conflict？**
当 **同一个 Warp 内的多个线程** 访问 **同一个 Bank** 的 **不同地址** 时，就会发生 Bank Conflict。
- **无冲突 (1-way):** Warp 内每个线程访问不同的 Bank（或同一个地址）。耗时 **1 cycle**。
- **n-way Conflict:** Warp 内有 `n` 个线程访问同一个 Bank 的不同地址。硬件会将这 `n` 个请求**串行化**执行，耗时 **n cycles**。
---
## 3. 经典场景：连续访问 vs 跨步访问
> **注意：** 下文假设 Shared Memory 为行主序存储 `smem[Row][Col]`。
### 3.1 场景一：连续访问 (No Conflict) —— 读取列
这是最高效的访问模式。
```cuda
__shared__ float smem[32][32];
// Warp 内 32 个线程，每个线程读取同一行中连续的 4 个元素
// 或者：读取一列（线程索引对应列索引）
int tid = threadIdx.x;
float val = smem[0][tid];  // 线程 0 读 [0][0], 线程 1 读 [0][1] ...
```
**地址分析:**
- 线程 `k` 访问 `smem[0][k]`。
- 线性地址 = `(0 * 32 + k) * 4` bytes。
- **Bank ID** = `(0 * 32 + k) % 32` = `k`。
**结论:** 32 个线程分别访问 Bank 0~31，每个 Bank 只被访问 1 次。**无冲突，1 cycle 完成。**
---
### 3.2 场景二：跨步访问 (Conflict) —— 读取行
这是最常见的性能杀手。
```cuda
__shared__ float smem[32][32];
// Warp 内 32 个线程，每个线程读取不同行的同一列
int tid = threadIdx.x;
float val = smem[tid][0];  // 线程 0 读 [0][0], 线程 1 读 [1][0] ...
```
**地址分析:**
- 线程 `k` 访问 `smem[k][0]`。
- 线性地址 = `(k * 32 + 0) * 4` = `k * 128` bytes。
- **Bank ID** = `(k * 32 + 0) % 32` = `0`。
**结论:** 32 个线程全部访问 **Bank 0** 的不同地址。硬件需要串行处理这 32 个请求。**32-way Conflict，耗时 32 cycles（性能暴跌 32 倍）。**
---
## 3.3 代码实战：Bank Conflict 修正对比
以下三个 Kernel 演示了如何通过 Padding 消除跨步访问带来的冲突。
#### Kernel A: Baseline (列访问，无冲突)
```cuda
__global__ void smem_read_no_conflict(float *output) {
    __shared__ float smem[32][32];
    int tid = threadIdx.x;
    // 写入：列访问，无冲突
    for (int i = 0; i < 32; i++) smem[i][tid] = tid * 32 + i;
    __syncthreads();
    
    float sum = 0.0f;
    // 读取：列访问，无冲突
    // smem[i][tid] -> Bank = (i*32 + tid) % 32 = tid (各线程不同)
    for (int i = 0; i < 32; i++) sum += smem[i][tid];
    output[tid] = sum;
}
```
#### Kernel B: Conflict (行访问，全冲突)
```cuda
__global__ void smem_read_conflict(float *output) {
    __shared__ float smem[32][32];
    int tid = threadIdx.x;
    // 写入：行访问，这里写入有冲突，但我们主要关注读取
    for (int i = 0; i < 32; i++) smem[tid][i] = tid * 32 + i;
    __syncthreads();
    
    float sum = 0.0f;
    // 读取：行访问，严重冲突！
    // smem[tid][i] -> Bank = (tid*32 + i) % 32 = i (所有线程访问同一列)
    for (int i = 0; i < 32; i++) sum += smem[tid][i];
    output[tid] = sum;
}
```
*注：Kernel B 中，每次循环迭代 `i`，所有线程都会同时访问 Bank `i`，导致严重的 32-way conflict。*
#### Kernel C: Fixed with Padding (行访问，无冲突)
```cuda
__global__ void smem_read_padding(float *output) {
    // 关键：列数从 32 变为 33
    __shared__ float smem[32][33]; 
    int tid = threadIdx.x;
    
    // 写入：行 stride 变成了 33 (奇数)
    for (int i = 0; i < 32; i++) smem[tid][i] = tid * 33 + i;
    __syncthreads();
    
    float sum = 0.0f;
    // 读取：行访问，但因为 stride 是奇数，Bank 被错开
    // smem[tid][i] -> Bank = (tid*33 + i) % 32 = (tid + i) % 32
    // 对于固定的 i，当 tid 从 0 变到 31，Bank 也会从 0 遍历到 31
    for (int i = 0; i < 32; i++) sum += smem[tid][i];
    output[tid] = sum;
}
```
---
## 4. 为什么 Padding 能消除冲突？
**核心原理：打破 32 的整除关系。**
在 Kernel B (无 Padding) 中：
- Row Stride = 32。
- 跨行跳转地址 = `tid * 32`。
- 由于 `32 % 32 = 0`，无论 `tid` 是多少，跨行的地址偏移量都落在 Bank 0 上。
在 Kernel C (Padding) 中：
- Row Stride = 33 (手动扩了一列)。
- 跨行跳转地址 = `tid * 33`。
- `33 % 32 = 1`。这导致每一行的起始地址相对于上一行，在 Bank 环中偏移了 1 位。
- 第 0 行起始 Bank: `0`
- 第 1 行起始 Bank: `(0 + 1) % 32 = 1`
- ...
- 第 `tid` 行起始 Bank: `(tid * 1) % 32 = tid`。
- 结果：32 个线程访问 32 个不同的 Bank。**冲突消除！**
---
## 5. 特例：Broadcast (广播)
**Broadcast 并不是 Conflict。**
如果一个 Warp 中的所有 32 个线程读取 **同一个 Bank 的同一个地址**，这被称为广播。硬件支持这种操作，通常只需要 **1 cycle**（或者极低的开销）。
```cuda
__shared__ float smem[32];
float val = smem[0]; // 所有线程都读 smem[0]
```
**总结区分：**
- **同 Bank + 不同地址** → Conflict (串行，慢)。
- **同 Bank + 同地址** → Broadcast (并行，快)。
- **不同 Bank** → No Conflict (并行，快)。
---
## 6. 实战诊断：使用 NCU 测量
使用 NVIDIA Nsight Compute (NCU) 的 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld_sum` 指标可以直接观测到 Shared Memory 的 Bank Conflict 次数。
```bash
# 编译 (SM80/A100)
nvcc -arch=sm_80 -O2 -o bank_conflict bank_conflict.cu
# 运行 NCU 采集 Shared Memory Bank Conflicts 指标
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld_sum ./bank_conflict
```
**预期结果对比：**
| Kernel | 访问模式 | Bank Conflicts 次数 (近似值) | 性能 |
|--------|---------|:---:|---|
| smem_read_no_conflict | 列访问 (Stride 1) | ~0 | 最快 |
| smem_read_conflict | 行访问 (Stride 32) | ~32 * 32 * N (巨大) | 最慢 |
| smem_read_padding | 行访问 (Stride 33) | ~0 | 快 (略浪费空间) |
---
## 7. 常见冲突场景与解决方案速查
| 场景 | 访问模式示例 | 是否 Conflict | 解决方案 |
| :--- | :--- | :---: | :--- |
| **向量/数组加载** | `smem[0][tid]` | ❌ No | 无需处理 |
| **矩阵转置 (写)** | `smem[y][x] = val` | ❌ No | 无需处理 (连续写入) |
| **矩阵转置 (读)** | `val = smem[x][y]` | ✅ **Yes** | Padding 或 Bank Swizzling |
| **多维数组 Reduce** | `smem[tid][i]` (行累加) | ✅ **Yes** | 改为列累加 `smem[i][tid]` 或 Padding |
### Padding 策略总结
如果你需要以 `32` (或 32 的倍数) 为步长跨越行（例如矩阵转置中的读取），必须进行 Padding。
```cpp
// 通用 Padding 模板
// 定义 Shared Memory 时，列数多开 1
constexpr int TILE_DIM = 32;
__shared__ float smem[TILE_DIM][TILE_DIM + 1]; // +1 是关键
```
| 数据类型 | Bank 宽度 | 冲突步长 | Padding 建议 |
|---|---:|---|---|
| float / int (4B) | 32 | 32, 64, 96... | COLS + 1 |
| double / long long (8B) | 32 (每次访问2个bank或对齐) | 16, 32... | COLS + 1 (逻辑列数+1) |
---
## 8. 学习检查清单
- [ ] 能画出 32 Banks 的物理结构图。
- [ ] 能默写 `bank_id = (word_addr) % 32` 公式。
- [ ] 能解释为什么 `smem[tid][0]` 会导致 Conflict，而 `smem[0][tid]` 不会。
- [ ] 理解 Padding 是通过改变 Row Stride 为奇数来错开 Bank 的。
- [ ] 知道 Broadcast 和 Conflict 的本质区别。
- [ ] 能使用 NCU 指标验证 Kernel 是否存在 Bank Conflict。
---
## 9. 参考阅读
1.  `src/Puzzle/bank_conflict.cu` — 亲自运行感受性能差异。
2.  LeetCUDA `layer_norm.cu` — 观察 `__shared__ float shared[NUM_WARPS]` 的使用，为何不需要 Padding（因为访问模式通常是连续的）。
3.  CUDA C++ Best Practices Guide: "Shared Memory" 章节。
