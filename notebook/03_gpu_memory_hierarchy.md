# 03 GPU 内存层级: 从寄存器到 HBM

> 对象: CUDA / GPU 零基础
> 前置: 01_gpu_hardware_architecture.md, 02_cuda_programming_model.md
> 目标: 面试能说清楚每种内存的位置、速度、大小、用途、scope 和 lifetime

---

## 1. 内存层级全景图

```
                ┌──────────────────────────────────────────┐
                │             SM (车间)                     │
                │                                          │
   速度最快 ←   │  寄存器 (Register File)                   │
   容量最小     │  每 SM 65536 个 32-bit 寄存器 (A100)       │
   延迟 ~0     │  每 thread 私有的寄存器 → 编译器分配         │
                │                                          │
                │  ┌────────────────────────────────────┐   │
                │  │  Shared Memory + L1 Cache (SRAM)   │   │
   延迟 ~5-30   │  │  每 SM 最大 164 KB (A100)           │   │
   cycles       │  │  block 内所有 thread 共享            │   │
                │  │  可配置 split (如 100KB SMEM +64KB L1)│  │
                │  └────────────────────────────────────┘   │
                │                                          │
                │  Constant Memory (cache)                  │
                │  Texture Memory (cache)                   │
                └──────────────────────────────────────────┘

                ┌─────────────┐      ┌─────────────────────┐
  速度较慢 ←    │  L2 Cache   │ ←──→ │   Global Memory     │
  容量较大      │  A100: 40MB │      │   (HBM / VRAM)      │
  延迟 ~200+   │  H100: 50MB │      │   A100: 80GB        │
  cycles       │  全 GPU 共享 │      │   带宽: 2.0 TB/s    │
                └─────────────┘      └─────────────────────┘

                ┌─────────────────────┐
                │  Host Memory (CPU)  │  ← 最慢, 需 PCIe/NVLink
                │  延迟 ~1000+ cycles │     带宽: PCIe 4.0 ~32GB/s
                └─────────────────────┘
```

---

## 2. 六种内存逐一详解

### 2.1 寄存器 (Registers) — 最快, 最珍贵

| 属性 | 值 |
|------|------|
| 位置 | SM 内 Register File |
| 延迟 | ~0 cycle (指令操作数直接来自寄存器) |
| 容量 | 每 SM 65536 个 32-bit (A100), 每 thread 最多 255 个 |
| Scope | 单个 thread |
| Lifetime | thread 存活期间 |
| 如何用 | 局部变量 (编译器自动分配) |

**面试关键计算:**

```
blockDim = 256, 每 thread 用 128 个寄存器
→ block 用 256×128 = 32768 寄存器
→ 每 SM 65536/32768 = 2 ← 最多驻留 2 个 block

register spilling: 寄存器不够用时, 编译器把变量"溢出"到 local memory
  → 实际存在 global memory (L1 cache), 延迟 ~200+ cycles
  → 用 --ptxas-options=-v 查看寄存器使用量
```

**寄存器用超了怎么办?**
```bash
nvcc --ptxas-options=-v kernel.cu
# 输出示例:
# Used 128 registers, 0 bytes smem, 0 bytes lmem  ← 没溢出
# Used 255 registers, 0 bytes smem, 64 bytes lmem ← 溢出了 64 bytes
```

`lmem` = local memory, 实际上在 global memory 中, 延迟剧增。

### 2.2 Shared Memory — Block 内共享的 SRAM

| 属性 | 值 |
|------|------|
| 位置 | SM 内 SRAM, 和 L1 Cache 共享同一块物理 SRAM |
| 延迟 | ~5 cycles (no conflict), ~30 cycles (bank conflict) |
| 容量 | 每 SM 最大 164 KB (A100, 可配置), 每 block 最多 48 KB (static) |
| Scope | 单个 block 内所有 thread |
| Lifetime | block 存活期间 |
| 如何用 | `__shared__` 关键字 |

**声明:**
```cuda
__global__ void kernel() {
    __shared__ float smem[256];           // 静态分配
    extern __shared__ float dynamic[];    // 动态分配 (大小由 launch 第三个参数决定)
}

// 动态 shared memory 用法:
kernel<<<grid, block, shared_mem_bytes>>>();
```

**Shared Memory 数量 vs 性能:**

```
每个 block 用 shared memory 越多 → 每个 SM 能驻留的 block 越少 → occupancy 越低
需要在 shared memory 大小和 occupancy 之间权衡

例子:
  每 block 48KB smem, A100 164KB/SM → 最多 3 blocks/SM
  每 block 16KB smem → 最多 10 blocks/SM (但 block 数通常被寄存器先限制住)
```

**Shared Memory Bank 冲突** → 详见 `bank_conflict_learning.md`

### 2.3 Global Memory (HBM / VRAM) — 最大的池子

| 属性 | 值 |
|------|------|
| 位置 | GPU 板载 HBM (A100: 80GB) |
| 延迟 | ~200-800 cycles |
| 容量 | 40/80 GB (A100), 80 GB (H100) |
| 带宽 | A100: 2.0 TB/s, H100: 3.35 TB/s |
| Scope | 整个 grid, 所有 kernel |
| Lifetime | 整个程序 |
| 如何用 | `cudaMalloc`, `cudaMemcpy` |

**Global Memory 的访问粒度:**

```cuda
// Global memory 访问按 cache line (128 bytes) 为单位
// 如果有 32 个 thread 各读 4 bytes, 且地址连续:
// → 128 bytes contiguous → 1 次 memory transaction (coalesced access)

// √ 合并访问 (coalesced):
float val = data[threadIdx.x];  // tid 0→data[0], tid 1→data[1], ...
// → 合并成 1 次 128B transaction

// × 非合并访问 (strided):
float val = data[threadIdx.x * 1024];  // 跨大步访问
// → 需要 32 次 32B transaction (最坏情况)
```

**Coalesced Access 原理图:**
```
合并访问:
  T0读addr0  T1读addr4  T2读addr8  ...  T31读addr124
  → 打包成一次 128B 读 → 带宽利用率 100%

跨步访问:
  T0读addr0  T1读addr4096  T2读addr8192  ...
  → 每个访问独立加载 cache line → 带宽利用率 ~3%
```

### 2.4 Constant Memory — 只读 + 广播

| 属性 | 值 |
|------|------|
| 位置 | 在 Global Memory 中分配, 通过专用 Constant Cache 访问 |
| 延迟 | ~0 cycle (cache hit) |
| 容量 | 64 KB (所有 SM 共享同一个 64 KB pool) |
| Scope | 整个 grid |
| Lifetime | 整个程序 |
| 如何用 | `__constant__` 关键字, host 端用 `cudaMemcpyToSymbol` |

```cuda
__constant__ float coeffs[1024];  // 最大 64 KB

__global__ void apply_coeffs(float *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) data[idx] *= coeffs[idx % 1024];
}

// host 端:
float h_coeffs[1024];
cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(h_coeffs));
```

**何时用**: 所有 thread 读相同地址的数据 (broadcast → 1 cycle)。例如: 卷积核系数、查找表。



### 2.5 向量化访存：float4 与 128 位加载

> 关键优化面试题：如何通过向量化数据类型（float4 / half2 / int4）提高显存带宽利用率、减少指令数和线程数。

#### A. 提高显存带宽利用率 (Memory Coalescing 与 128 位对齐)

```
GPU 的 global memory 访问以 memory transaction 为单位（32 字节或 128 字节）。
一个 warp (32 threads) 的请求能否被合并成少量 transaction，取决于地址对齐和访问模式。

不使用 float4：
  每个线程读 1 个 float (4 字节)
  warp 32 线程 = 32 * 4 = 128 字节 (对齐在 128B 边界上) -> 可以合并为 1 次 128B transaction
  但地址必须是 128B 对齐的：若线程 0 读 addr0，线程 1 读 addr1，线程 2 读 addr2 ...
  此时 addr = base + tid * 4，stride=4 字节 -> 地址连续 -> 合并 OK

使用 float4：
  每个线程读 4 个 float (16 字节)
  warp 32 线程 = 32 * 16 = 512 字节 = 4 次 128B transaction
  每个线程的地址天然 16 字节对齐（因为 sizeof(float4) = 16）
  一个 warp 的 4 次 128B transaction 被完美拆分，无跨 segment 访问

  // float4 版本
  float4 val = ((float4*)arr)[idx];  // 1 条 LDG.128 指令
  // 等价于但更高效
  float v0 = arr[idx*4+0];           // 1 条 LDG
  float v1 = arr[idx*4+1];           // 1 条 LDG
  float v2 = arr[idx*4+2];           // 1 条 LDG
  float v3 = arr[idx*4+3];           // 1 条 LDG

关键：仅仅"地址连续"不够——还需要"地址对齐到 128B 边界"。
  若 base address % 128 != 0，即使连续访问也会跨 cache line 边界，
  一次 128B transaction 变成 2 次 128B transaction (带宽利用率减半)。
  使用 float4 强制 16B 对齐，有利于编译器生成对齐的 LDG.128 指令。
```

#### B. 减少指令数量

```
不使用 float4: 编译器可能生成 4 条 LDG (32-bit load) 指令
  或幸运时 fuse 成 1 条 LDS.128/SMEM load——但 global memory 上无保证。

使用 float4: 编译器保证生成 1 条 LDG.128 (128-bit vector load) 指令
  PTX 层面:
    ld.global.v4.f32  {%r0, %r1, %r2, %r3}, [%rd1];   // 1 条向量化加载
  SASS 层面:
    LDG.E.128  R0, [R6];                               // 1 条 128-bit load

指令数减少的收益：
  - 指令 cache (I-Cache) 压力减小（更少的指令占据 I-Cache）
  - 取指/解码带宽节省（解码单元可更专注于其他指令）
  - warp scheduler 的 issue slot 更宽松（1 条 vs 4 条指令占 1 个 cycle vs 4 cycles）

  同理适用于 store:
    float4 val = make_float4(a, b, c, d);
    ((float4*)output)[idx] = val;   // 1 条 STG.128

    vs:
    output[idx*4+0] = a;            // 1 条 STG
    output[idx*4+1] = b;            // 1 条 STG
    output[idx*4+2] = c;            // 1 条 STG
    output[idx*4+3] = d;            // 1 条 STG
```

#### C. 减少启动的线程数量

```
处理 N 个 float 元素：

  naive 方案: N 个线程，每个处理 1 个元素
  float4 方案: N/4 个线程，每个处理 4 个元素

  收益：
    - 线程块调度开销减少（~1/4 的 block 数）
    - __syncthreads() 同步开销减少（同步次数不变但参与的 warp 更少）
    - 寄存器压力降低（虽然每个线程多用 4 个寄存器，但总线程数减少 > 4 倍？并不）
      注意: 每个线程的寄存器占用会升高（因为需要 4 个值），能否省寄存器看具体实现。

实际使用的典型模式：

  // --- 标准写法 ---
  __global__ void add_kernel(const float *a, const float *b, float *c, int N) {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx < N) c[idx] = a[idx] + b[idx];
  }

  // --- float4 优化版 ---
  __global__ void add_kernel_float4(const float *a, const float *b, float *c, int N) {
      int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;      // ×4
      if (idx + 3 < N) {  // 同时检查 4 个元素
          float4 va = ((float4*)a)[idx/4];
          float4 vb = ((float4*)b)[idx/4];
          float4 vc;
          vc.x = va.x + vb.x;
          vc.y = va.y + vb.y;
          vc.z = va.z + vb.z;
          vc.w = va.w + vb.w;
          ((float4*)c)[idx/4] = vc;
      }
      // 处理尾部不足 4 个的元素
      for (int i = idx + (idx + 3 < N ? 4 : 0); i < N; i++)
          c[i] = a[i] + b[i];
  }
```

#### D. float4 的适用场景与局限

```
适用场景：
  - Elementwise ops: add, mul, silu, relu, gelu 等（读入→计算→写回）
  - Memory-bound 算子（占绝大多数 fuse-able elementwise 场景）
  - 数据连续排列（AOS 不行，SOA 可以）

不适用场景：
  - Reduce / softmax / layernorm（需要跨元素归约，float4 打包后归约反而更复杂）
  - 随机访存（索引不连续，float4 加载无用数据浪费带宽）
  - 数据类型不是 4 字节的倍数（如 char/short 无法直接用 float4）
  - GPU 架构较老（Maxwell/Kepler 之前 LDG.128 支持不好）

其他向量化数据类型：
  float4  (16B): 4 个 float, 最常用
  float2  (8B):  2 个 float, 宽度不够但比 scalar 好
  half2   (4B):  2 个 half, fp16 场景常用（LeetCUDA gelu/elu/relu 大量使用）
  int4    (16B): 4 个 int, INT8/INT32 计算用
  double2 (16B): 2 个 double, double precision 场景
```

> **面试题：什么叫 vectorized memory access？为什么 float4 比 4 次 float load 更快？**
> 标准回答：减少指令数（LDG.128 代替 4x LDG）、保证 128-bit 对齐提高 coalescing 效率、
> 减少取指/解码成本、减少线程数/block 调度开销。注意尾部不足 4 个元素的处理，
> 以及不适用于 reduce/softmax 等归约操作。

### 2.6 Texture Memory — 2D 空间局部性

| 属性 | 值 |
|------|------|
| 位置 | 在 Global Memory 中分配, 通过 Texture Cache 访问 |
| 特性 | 硬件插值、边界处理、2D 空间局部性 cache |
| Scope | 整个 grid |
| 何时用 | 图像处理、2D 数据访问 |

在现代 CUDA 中, Texture Memory 用得越来越少 (因为 L1/L2 cache 越来越智能), 面试了解概念即可。

### 2.7 Local Memory — 寄存器的"备胎"

编译器把放不进寄存器的变量"溢出"到 local memory, 物理上在 Global Memory (由 L1 缓存):

```cuda
__global__ void kernel() {
    // 如果这个大数组不 fit 进寄存器, 就去 local memory
    float big_array[1024];  // 很可能溢出
}
```

**local memory 不用手动声明, 是编译器自动的决定。**

---

## 3. L1 Cache 与 Shared Memory 的物理关系

**L1 Cache 和 Shared Memory 共享同一块 SRAM 物理资源**, 大小可配置:

| 配置 | Shared Memory | L1 Cache | 适用场景 |
|------|:---:|:---:|------|
| `cudaFuncCachePreferShared` | 最大 100 KB | 剩余 | block 内协作多 (reduce, scan) |
| `cudaFuncCachePreferL1` | 最大 28 KB | 剩余 | 访存模式不规律, 依赖 cache |

```cuda
cudaFuncSetCacheConfig(my_kernel, cudaFuncCachePreferShared);
```

---

## 4. 每种内存的速度数量级对比

| 内存类型 | 延迟 (cycle) | 延迟 (ns @1.4GHz) | 带宽 | 每 SM 容量 |
|------|:---:|:---:|------|:---:|
| Register | ~0 | ~0 | - | 256 KB (65536×4B) |
| Shared Memory | ~5-30 | ~3.5-21 ns | ~128 B/clk per SM | 最大 164 KB |
| Constant Cache | ~0 (hit) | ~0 | 取决于 broadcast | 64 KB (全 GPU) |
| L1 Cache | ~30 | ~21 ns | - | 共享 SMEM 池 |
| L2 Cache | ~200 | ~140 ns | ~3-4 TB/s | 40 MB (A100) |
| Global (HBM) | ~200-800 | ~140-570 ns | 2 TB/s (A100) | 40/80 GB |
| Host (CPU DDR) | ~1000+ | ~700+ ns | 32 GB/s (PCIe 4.0) | 数百 GB |

**规律: 容量每大一级, 延迟约 ×10, 带宽约低 10×。**

---

## 5. 数据搬运策略：为什么要把数据搬到 Shared Memory

```cuda
// 反例: 反复读 global memory
for (int i = 0; i < 1000; i++) {
    sum += global_data[idx];  // 每次 ~300 cycles
}
// 总延迟: 1000 × 300 = 300,000 cycles

// 正例: 先搬到 register / shared memory
float val = global_data[idx];  // 1 次 300 cycles
for (int i = 0; i < 1000; i++) {
    sum += val;  // 每次 ~0 cycles (寄存器)
}
// 总延迟: 300 + 1000×0 ≈ 300 cycles — 快 1000×
```

**Matrix Multiply 的标准做法:**
```
1. 把 A_tile 和 B_tile 从 global memory 搬到 shared memory
2. 在寄存器上计算 C_tile = A_tile × B_tile
3. 把 C_tile 写回 global memory

每个 global memory byte 被复用多次 (O(K) 次) → 最大化带宽利用率
```

---

## 6. 面试高频问题

### Q1: GPU 有哪几种内存? 按速度排序。
**答:** Register > Shared Memory ≈ Constant Cache > L1 > L2 > Global (HBM) > Host (CPU)。寄存器最快 (~0 cycle), shared memory ~5-30 cycles, global memory ~200-800 cycles。

### Q2: 什么是 Coalesced Memory Access?
**答:** 一个 warp 的 32 个线程在同一个时钟周期访问 global memory 时, 如果地址连续且在 128-byte 对齐的范围内, 硬件会把它们合并成一次 memory transaction。非合并访问会显著降低有效带宽。

### Q3: Shared Memory 和 L1 Cache 的关系?
**答:** 共享同一块片上 SRAM, 大小可配。`cudaFuncCachePreferShared` 给 SMEM 更多空间, 反之给 L1 更多。

### Q4: 寄存器 spilling 是什么? 怎么查?
**答:** 当编译器发现寄存器不够时, 把变量存到 local memory (在 global memory 中, 经 L1 cache)。用 `nvcc --ptxas-options=-v` 可以看到 `lmem` 用量。spilling 会让 kernel 性能骤降。

### Q5: `__constant__` 内存什么时候用?
**答:** 所有线程读相同地址的数据, 数据量 < 64 KB。warp 内 broadcast 只需 1 cycle。典型用途: 卷积核参数、多项式系数。

### Q6: 为什么要把数据从 global 搬到 shared?
**答:** Shared memory (~5 cycle) 比 Global memory (~300 cycle) 快 ~60×, 可以放入片上反复使用。关键计算模式是: tile 数据从 global → shared, 在寄存器上计算, 结果写回 global。

---

## 7. 参考链接

- [CUDA C++ Programming Guide - Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [CUDA C++ Best Practices Guide - Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- `bank_conflict_learning.md` — Shared Memory Bank Conflict 专题
- [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

---

## 8. 学习检查清单

- [ ] 能从快到慢列出 GPU 6 种内存, 并给出典型延迟
- [ ] 知道每 SM 65536 个 32-bit 寄存器的限制
- [ ] 理解 coalesced vs non-coalesced global memory access
- [ ] 知道 `__shared__` 的 scope 是 block, lifetime 是 block
- [ ] 能用 `nvcc --ptxas-options=-v` 检查 register spilling
- [ ] 知道 constant memory 的 broadcast 机制
- [ ] 理解 data tiling: global → shared → register 的模式
- [ ] 能回答上面 6 个面试问题
- [ ] 能用 float4 优化 elementwise kernel，说清指令数减少和带宽提升原理
- [ ] 能区分 float4 适用场景（elementwise/连续访存）和不适用场景（reduce/随机访存）
- [ ] 知道 half2 / int4 / double2 等向量化数据类型
- [ ] 能写出 float4 优化版的 vector_add kernel，包含尾部处理
