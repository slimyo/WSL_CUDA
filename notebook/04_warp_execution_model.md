# 04 Warp 执行模型: SIMT, 分支发散, 调度与 Occupancy

> 对象: CUDA / GPU 零基础
> 前置: 01_gpu_hardware_architecture.md, 02_cuda_programming_model.md
> 目标: 面试能讲清楚 warp 如何执行、分支发散如何发生、occupancy 如何计算

---

## 1. Warp 是什么

**Warp 是 GPU 硬件调度的最小单位。** 一个 warp = 32 个线程, 这 32 个线程**永远同时**执行同一条指令, 只是操作的数据不同。

```
Thread Block: 256 threads = 8 warps
  warp 0: thread 0-31
  warp 1: thread 32-63
  warp 2: thread 64-95
  ...
  warp 7: thread 224-255
```

**Warp 内线程是连续编号的:** `warp_id = threadIdx.x / 32`, `lane_id = threadIdx.x % 32`。

---

## 2. SIMT — 单指令多线程

**SIMT (Single Instruction, Multiple Threads)** 是 NVIDIA 对 SIMD 的扩展:

| | SIMD (CPU, e.g. AVX) | SIMT (GPU, CUDA) |
|------|------|------|
| 宽度 | 固定 (256/512 bit) | 逻辑上 32 threads |
| 编程模型 | 显式向量寄存器 (__m256) | 标量编程 (像写单线程代码) |
| 分支 | 需要手动 mask | 硬件自动处理 (warp divergence) |
| 内存访问 | 需要显式 gather/scatter | 每个 thread 独立地址 |

**SIMT 的好处: 写 GPU 代码时你不需要想"向量", 只需要写标量 thread 代码。**

```cuda
// GPU 代码看起来像普通C代码:
__global__ void add(float *a, float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

// 但硬件执行时: warp 中 32 个线程同时执行这一条 FADD 指令
// lane 0: a[0] + b[0], lane 1: a[1] + b[1], ... lane 31: a[31] + b[31]
```

---

## 3. Warp 调度: 零开销切换

### 3.1 调度流程

```
SM Partition 内部:

Warp Scheduler (每 SM Partition 1 个)
  ├── 查看所有驻留 warps 中哪些可以发射下一条指令
  ├── 选择 1 个 warp
  ├── Dispatch Unit 把指令发到执行单元
  └── 每个时钟周期重复

就绪 warp 的条件:
  - 上一条指令的操作数已就绪 (寄存器/共享内存数据依赖解除)
  - 执行单元可用 (没有其他 warp 占用 FP32 Core 等)
  - 不在等待访存结果
```

### 3.2 零开销切换的秘密

```cuda
// warp 0 执行:
float val = global_data[idx];  // global memory 访存, ~300 cycles
// warp 0 此时 stall — 等待数据从 HBM 返回

// 硬件从不等待:
// → Warp Scheduler 立刻切换到 warp 1
// → warp 1 执行它的指令
// → warp 1 stall → 切到 warp 2
// → ...
// → 300 cycles 后, warp 0 的数据到了, warp 0 重新就绪

// 关键: 上下文切换不需要保存/恢复任何东西
// 每个 warp 的寄存器是独立的、物理隔离的
// 交换只是改变 Warp Scheduler 指向哪个 warp 的寄存器堆
```

**对比 CPU 线程切换:**
```
CPU: 保存 32+ 个寄存器 → 换页表 → TLB flush → 恢复 32+ 寄存器 → ~1μs
GPU: Warp Scheduler 换个指针 → 0 cycle
```

---

## 4. Warp Divergence (分支发散) — 面试必考

### 4.1 问题

```cuda
__global__ void divergent_kernel(float *data, int N) {
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        data[tid] *= 2.0f;  // 偶数线程
    } else {
        data[tid] /= 2.0f;  // 奇数线程
    }
}
```

**发生什么?** 一个 warp 内, lane 0,2,4,... 走 if 分支, lane 1,3,5,... 走 else 分支。

**GPU 不能同时执行两个分支 → 串行化:**

```
时间 →
  ├─ if 路径: lane 0,2,4,...,30 活跃 (lane 1,3,5,...,31 被禁用)
  ├─ else 路径: lane 1,3,5,...,31 活跃 (lane 0,2,4,...,30 被禁用)
  └─ 汇合点

总执行时间: if_path_time + else_path_time  (2× 的代价)
```

### 4.2 哪些情况会发散?

```cuda
// 1. 条件基于 threadIdx (warp 内)
if (lane_id < 16) {...} else {...}   // 发散! lane 0-15 vs 16-31

// 2. 循环次数不同
for (int i = 0; i < lane_id; i++) {...}  // 发散! 每 lane 迭代次数不同

// 3. 条件基于数据值
if (data[tid] > 0) {...}  // 可能发散 (取决于数据)

// 以下不会发散 (只要条件在 warp 内一致):
if (blockIdx.x == 0) {...}  // 不! blockIdx 在 warp 内一致
if (warp_id > 2) {...}      // 不! warp 内所有 lane 的 warp_id 相同
```

### 4.3 SIMD 风格的"if 转计算"优化

```cuda
// 发散版本:
if (lane_id < 16)
    val = a * b;
else
    val = a + b;

// 无发散版本 (避免分支):
float coeff = (lane_id < 16) ? 1.0f : 0.0f;
val = coeff * (a * b) + (1.0f - coeff) * (a + b);
```

**但也不是绝对需要避免分支:** 如果分支很短 (几条指令), 发散的开销可能小于乘加指令链。实际情况用 profiler。

### 4.4 Volta+ 的 Independent Thread Scheduling

从 Volta (SM70) 开始, GPU 支持 warp 内独立线程调度:

```
Pascal 及以前:
  所有线程在同一 PC (Program Counter) → lockstep

Volta+:
  每个线程可以有独立的 PC → warp 内可以交错执行
  但 SIMT 模型不变: 同一时刻仍执行同一条指令的线程必须有相同 PC
```

这意味着一部分线程可以在 A 分支, 一部分在 B 分支, 但硬件可以交错执行而不是先 A 后 B。不过对程序员来说, 仍是"不要依赖 warp 内同步"。

---

## 5. Occupancy — 衡量 SM 利用率

### 5.1 公式

```
Occupancy = active_warps_per_SM / max_warps_per_SM

active_warps_per_SM = min(
    驻留 blocks × (blockDim / 32),
    2048 / 32  (= 64 warps per SM max)
)
```

### 5.2 三个限制因素

**1. 寄存器限制 (最常见)**

```
active_blocks = min(
    SM_register_count / (blockDim × registers_per_thread),
    max_blocks_per_SM  (= 32)
)
```

**2. Shared Memory 限制**

```
active_blocks = min(
    SM_shared_memory / shared_memory_per_block,
    max_blocks_per_SM
)
```

**3. Block 数量限制**

```
每 SM 最多 32 blocks
每 SM 最多 2048 threads
每 block 最多 1024 threads
```

### 5.3 实际计算示例

```
环境: A100 (SM80)
  SM_registers = 65536
  SM_shared_memory = 164 KB (configurable)
  SM_max_threads = 2048
  SM_max_blocks = 32

Kernel: blockDim = 256, registers_per_thread = 72, smem_per_block = 8 KB

寄存器限制: 65536/(256×72) = 3.55 → 3 blocks
SMEM 限制: 164/8 = 20 blocks (不限制)
线程数限制: 2048/256 = 8 blocks (不限制)
块数限制: 32 (不限制)

→ active_blocks = 3
→ active_warps = 3 × 8 = 24 warps
→ occupancy = 24/64 = 37.5%
```

### 5.4 Occupancy 是不是越高越好?

**不。** 很多高性能 kernel 故意用低 occupancy:

```cuda
// 场景: 每个 thread 需要很多寄存器做 tile computation
// 高 occupancy: 256 threads × 32 regs = 8192, 8 blocks/SM = 64 warps, 100%
// 低 occupancy: 256 threads × 128 regs = 32768, 2 blocks/SM = 16 warps, 25%

// 但低 occupancy 版本可能更快, 因为:
//   - 更多寄存器 per thread → 更大 tile → 更少 global memory 访问
//   - 更多 ILP (Instruction-Level Parallelism) 在单个 warp 内
//   - global memory bandwidth 是瓶颈时, 多 warp 也无用
```

**结论:** occupancy 是手段不是目的。目标是最小化 latency × bandwidth product, 不是最大化 occupancy。

---

## 6. Latency Hiding — 延迟隐藏的数学

```
需要的 active warps = avg_latency / execution_rate

例:
  一条 global load 延迟 = 300 cycles
  一条 FMA 指令 吞吐 = 32 ops/clk (每 SM)
  
  如果每个 warp 在执行 FMA 之前需要等待 300 cycles 的 load:
  需要 300/(32/4) ≈ 38 warps per SM 来隐藏延迟

这解释了为什么 Ampere 上限是 64 warps/SM: 刚好够覆盖 ~300 cycle 的 global memory 延迟。
```

**延迟隐藏的公式:**
```
sufficient_warps = round_up(latency_in_cycles × warps_per_cycle)
≈ 300 / 4 ≈ 75  (如果 GPU 每 4 cycles 执行完一个 warp 的工作)

实际中还要考虑:
- 指令级并行 (一个 warp 内同时多条不同指令飞行)
- 内存级并行 (多个 load 请求同时在飞)
```

---

## 7. Warp-Level Primitives 简介 (预告)

在同一个 warp 内的 32 个线程之间, GPU 提供极高效的数据交换机制:

```cuda
// warp shuffle — 寄存器间直接交换, 延迟 ~5 cycles
float val = __shfl_down_sync(0xffffffff, my_val, delta);

// warp vote — 全局同步原语
int all_true = __all_sync(0xffffffff, condition);
int any_true = __any_sync(0xffffffff, condition);
int ballot   = __ballot_sync(0xffffffff, condition);
```

这是下一章 `05_warp_shuffle_primitives.md` 的主题。

---

## 8. 面试高频问题

### Q1: 什么是 warp? 为什么是 32?
**答:** Warp 是 GPU 硬件调度的最小单位, 包含 32 个线程。32 是 NVIDIA 的硬件设计选择, 刚好映射到 SM Partition 的执行单元宽度。AMD 的 wavefront 是 64。

### Q2: SIMT 和 SIMD 有什么区别?
**答:** SIMD (CPU AVX) 需要显式使用向量寄存器, 固定宽度。SIMT (GPU) 允许标量编程, 硬件自动向量化, 支持线程级分支, 每个线程独立地址空间。

### Q3: 什么是 Warp Divergence? 怎么避免?
**答:** Warp 内不同线程走不同分支 → 两条路径串行执行 → 效率减半。避免方法: 让分支条件在 warp 级别一致 (如条件基于 warp_id 而非 lane_id), 或把分支替换为条件计算。

### Q4: GPU 的零开销线程切换怎么做到的?
**答:** 每个 warp 有自己物理隔离的寄存器堆。切换 warp 只需 Warp Scheduler 改变指向的寄存器基地址, 不需要保存/恢复任何上下文。CPU 线程切换需要 ~1μs, GPU 是 0 cycle。

### Q5: Occupancy 怎么算? 受什么限制?
**答:** `active_warps/max_warps`。受三方面限制: (1) 每 SM 65536 寄存器, (2) 每 SM 最大 164KB shared memory, (3) 每 SM 最多 2048 threads / 32 blocks。

### Q6: 什么时候低 occupancy 比高 occupancy 更好?
**答:** 当瓶颈是内存带宽或每个 thread 需要大量寄存器做 compute 时。更多寄存器 per thread 意味着更大的 tile → 更少 global memory 访问。很多高性能 GEMM kernel 故意用 25-50% occupancy。

---

## 9. 参考链接

- [CUDA C++ Programming Guide - SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture)
- [CUDA C++ Programming Guide - Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-multithreading)
- [NVIDIA Volta Architecture Whitepaper - Independent Thread Scheduling](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)

---

## 10. 学习检查清单

- [ ] 理解 warp = 32 threads, lane_id = threadIdx.x % 32
- [ ] 知道 SIMT 和 SIMD 的区别 (编程模型 vs 执行模型)
- [ ] 理解 warp scheduler 如何零开销切换
- [ ] 能识别什么情况会导致 warp divergence
- [ ] 能手算 occupancy (考虑寄存器、smem、线程数三方面)
- [ ] 理解 latency hiding 的数学原理
- [ ] 知道 Volta+ independent thread scheduling 的含义
- [ ] 能回答上面 6 个面试问题
