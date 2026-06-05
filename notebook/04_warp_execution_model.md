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

## 6. Latency Hiding — 延迟隐藏详解

> **核心命题:** GPU 的 global memory 延迟是 300—800 cycles, 一条 FMA 指令延迟 4 cycles。如果 warp 每 cycle 都能发射指令, 一个 warp 最多只能连续执行 1-2 条指令就 stall。没有延迟隐藏, GPU 绝大部分时间都在空等。
> **答案:** 不是消除延迟, 而是用并行让延迟不在关键路径上出现。

### 6.1 为什么需要延迟隐藏 — 延迟全景图

GPU 内各类操作的典型延迟 (以 A100 @ 1.4 GHz 为例):

| 操作                      | 延迟 (cycles) | 延迟 (ns)  | 说明                         |
|---------------------------|---------------|------------|------------------------------|
| 同一 warp 内寄存器转发     | 0             | 0          | 当前指令直接使用上个指令结果 |
| FMA / ADD / MUL           | 4             | ~2.9       | 普通算术指令                 |
| SFU (sin, sqrt, rcp)      | 16            | ~11        | 特殊函数单元                 |
| Shared Memory (bank 无冲突)| ~30           | ~21        | 片上内存                     |
| L1 / 常量缓存             | ~30           | ~21        | 片上缓存                     |
| L2 命中                   | ~200          | ~143       | 片上缓存                     |
| Global Memory (HBM)       | ~400—800      | ~290—570   | 显存访问 (主要延迟源)        |
| Atomic (global)           | ~600—1000     | ~430—710   | 原子操作                     |
| __syncthreads             | ~40           | ~29        | 同步屏障                     |

**关键观察:** 一条 LDG (global load) 的延迟可以执行 100—200 条 FMA。如果 warp 必须等待每次访存结果才能继续, 吞吐率将极低。

**CPU vs GPU 的处理哲学:**

|                    | CPU                               | GPU                                       |
|--------------------|-----------------------------------|-------------------------------------------|
| 应对延迟的策略     | 尽量减小延迟                       | 不减小延迟, 隐藏它                         |
| 手段               | 大缓存 (L1/L2/L3), 分支预测, OoO  | 大量并行线程 + 快速上下文切换              |
| 晶体管投入         | ~50% 用于缓存 + 控制               | ~80% 用于计算单元 (ALU)                    |
| 结果               | 单线程极快, 但并行度有限           | 单线程慢, 但数千线程同时运行, 总吞吐极高   |

---

### 6.2 核心机制: Warp Scheduling 如何实现延迟隐藏

#### 6.2.1 基本调度流程

每个 SM 内有多个 Warp Scheduler (A100 每 SM Partition 1 个, 共 4 个), 每个 scheduler 管理一组驻留 warp。

```
Warp Scheduler 每 cycle 的工作:
  1. 扫描所有管理的就绪 warp (pending queue)
  2. 选择优先级最高的 warp
  3. 通过 Dispatch Unit 将下一条指令发送到执行单元
  4. 被选中的 warp 进入未就绪状态 (等待指令延迟)

就绪 warp 条件:
  - 上一条指令的操作数已就绪 (寄存器/内存数据已到)
  - 执行单元有空闲 (比如 FP32 Core 可用)
  - 指令发射队列未满
```

#### 6.2.2 零开销切换 — 时间线示例

假设 4 个 warp, 每个 warp 执行一条 LDG (延迟 300 cycles) + 一条 FMA:

```
cycle  ── warp 0 ── ── warp 1 ── ── warp 2 ── ── warp 3 ──
  0     LDG (存 t0)
  1                   LDG (存 t1)
  2                                 LDG (存 t2)
  3                                               LDG (存 t3)
  4     ld_stall     ld_stall      ld_stall      ld_stall
  5     ld_stall     ld_stall      ld_stall      ld_stall
  …
  304   FMA(t0)      ready         ready         ready
  305                 FMA(t1)       ready         ready
  306                               FMA(t2)       ready
  307                                             FMA(t3)
  308   done         done          done          done
```

**4 个 warp 完成总工作量的时间 = 308 cycles**
如果没有延迟隐藏 (只能串行执行) = (300 + 4) × 4 = 1216 cycles
**加速比 ≈ 4×** — 这就是延迟隐藏的直接效果。

> **关键是:** 每次"切换"的成本为零。warp 的寄存器是物理隔离的, Warp Scheduler 只需改变 `warp_id → register_base` 的映射, 不需要保存/恢复上下文。

---

### 6.3 三种并行手段: TLP, ILP, MLP

延迟隐藏通过三种并行度实现, 它们可以叠加使用:

#### 6.3.1 TLP — Thread-Level Parallelism (线程级并行)

**核心思路:** 用多个 warp 轮流执行, 当一个 warp 因访存/数据依赖 stall 时, 立刻切换到另一个就绪 warp。

```
假设: 每个 warp 每 4 cycles 发射 1 条指令 (instruction issue interval = 4)
      访存延迟 = 400 cycles

需要的 warp 数 ≈ 400 / 4 = 100 warps

但 GPU 每 SM 最多 64 warps → 仅用 TLP 不够!
```

**TLP 能隐藏的延迟取决于 warp 数量:**

```
hideable_latency_TLP = active_warps × issue_interval

例: 32 warps, 每 4 cycles 发 1 条指令
  → 可隐藏延迟 = 32 × 4 = 128 cycles
  → 实际 LDG 延迟 ~400 cycles → 还有 272 cycles 无法隐藏
```

**结论: TLP 单独不足以完全隐藏 GMEM 延迟。** 必须结合 ILP 和 MLP。

#### 6.3.2 ILP — Instruction-Level Parallelism (指令级并行)

**核心思路:** 单个 warp 内, 一条指令的结果被后续指令需要才形成依赖链。如果指令之间没有数据依赖, Warp Scheduler 可以连续发射这些独立指令, 不 stall。

```cuda
// 低 ILP — 每条指令都依赖前一条结果
float a = load(x[i]);          // LDG, 延迟 400
float b = a * 2.0f;            // 需要等 a → stall 400 cycles
float c = b + 1.0f;            // 需要等 b → stall 4 cycles

// 高 ILP — 两条独立的计算链
float a0 = load(x[i]);         // LDG #1
float a1 = load(x[i+1]);       // LDG #2 ← 和 LDG #1 独立
float b0 = a0 * 2.0f;          // 需要等 a0
float b1 = a1 * 2.0f;          // 需要等 a1
// b0 和 b1 之间没有依赖
// Warp Scheduler 可以在等 a0 的同时发射 LDG #2 和 a1*2 等
```

**ILP 对延迟隐藏的贡献:**

```
effective_hideable_latency = active_warps × issue_interval × ILP_factor

ILP_factor = warp 内平均并行独立指令流的数量

例: 32 warps, issue_interval = 4, ILP_factor = 4 (循环展开 4 路)
  → 可隐藏延迟 = 32 × 4 × 4 = 512 cycles → 足够覆盖 ~400 cycles LDG!
```

**获得 ILP 的方法:**

```cuda
// 方法 1: 循环展开 (Loop Unrolling)
// 4 路展开, 4 条独立的 load + 4 条独立的 FMA
#pragma unroll 4
for (int i = 0; i < N; i++) {
    sum += a[tid + i * blockDim.x];
}

// 方法 2: 手工展开 + 交错计算
float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
for (int i = 0; i < N; i += 4) {
    s0 += a[tid + (i+0) * blockDim.x];  // LDG #1 → FMA #1 (独立)
    s1 += a[tid + (i+1) * blockDim.x];  // LDG #2 → FMA #2 (独立)
    s2 += a[tid + (i+2) * blockDim.x];  // LDG #3 → FMA #3 (独立)
    s3 += a[tid + (i+3) * blockDim.x];  // LDG #4 → FMA #4 (独立)
}
// 4 条 LDG 互不依赖, 可以同时 "in-flight"
// 4 个累加器 s0-s3 也互不依赖
```

> **ILP 的代价:** 更多的寄存器。每个独立的累加器需要寄存器, 4 路展开就需要 4× 寄存器 → 减少了可驻留 warp 数 (←→ TLP 的平衡)。

#### 6.3.3 MLP — Memory-Level Parallelism (内存级并行)

**核心思路:** 一个 warp 可以发出多个**未完成**的内存请求, 然后一起等待。GPU 的 memory controller 可以同时处理多个 pending 请求。

```cuda
// 低 MLP: 一次只发 1 个 4-byte 请求
float v0 = data[tid];          // LDG #1 → stall 400 cycles → 数据回来
// 然后才能发下一个
float v1 = data[tid + N];      // LDG #2 → stall 400 cycles

// 高 MLP: 一次发 4 个 4-byte 请求
float v0 = data[tid];          // LDG #1
float v1 = data[tid + N];      // LDG #2 (不依赖 #1, 立即发射)
float v2 = data[tid + 2*N];    // LDG #3 (不依赖 #1,#2)
float v3 = data[tid + 3*N];    // LDG #4 (不依赖 #1,#2,#3)
// 4 个 LDG 在完全不同的地址上, memory controller 可以并行处理
// 总等待时间 ≈ 1 × 400 cycles (而不是 4 × 400)
```

**MLP 的限制因素:**

1. **Memory 带宽:** 如果所有请求加起来已经打满 HBM 带宽, 更多的 MLP 不会提升速度。
2. **Scoreboard 容量:** GPU 硬件跟踪每个 warp 未完成的寄存器写入。每个 warp 可同时 in-flight 的 LDG 数是有限的 (通常 16—32 个)。
3. **请求合并:** 如果多个 LDG 访问连续地址 (coalesced), 它们被合并成一次大请求, 不产生额外 MLP 收益。

**MLP 在 Intel Xe / AMD CDNA 上也类似存在, 是 GPU 架构共同特征。**

---

### 6.4 统一延迟隐藏模型 — 数学框架

将三种手段统一到一个公式:

```
warp 发出指令到结果就绪 → stall
stall 期间, Warp Scheduler 切换到其他 warp

要完全隐藏延迟需要满足:

  active_warps × issue_width × ILP_factor × MLP_factor ≥ latency / issue_interval

其中:
  active_warps  ─  当前 SM 驻留的 warp 数 (由 occupancy 决定)
  issue_width   ─  每 cycle 可发射指令数 (A100: 2 per warp scheduler)
  ILP_factor    ─  warp 内可并行执行的独立指令流数
  MLP_factor    ─  每个 warp 同时 in-flight 的独立内存请求数
  latency       ─  目标隐藏的延迟 cycles
  issue_interval ─ warp 两次成功发射之间的 cycles (通常 4)

→ 不考虑 ILP/MLP 时, 纯 TLP 所需 warp 数:
  required_warps = latency / (issue_width × issue_interval)
                  = 400 / (2 × 4) = 50 warps  ← 接近 A100 极限 64

→ 考虑 4 路 ILP 后:
  required_warps = 400 / (2 × 4 × 4) = 12.5 ≈ 13 warps
          即 20% occupancy 就够!

→ 考虑 4 路 ILP + 4 路 MLP 后:
  required_warps = 400 / (2 × 4 × 4 × 4) = 3.1 ≈ 4 warps
          即 6% occupancy 就够!
```

**实际计算验证:**

| 配置                       | ILP | MLP | 所需 warps | 等效 occupancy | 适用场景           |
|----------------------------|-----|-----|-----------|----------------|--------------------|
| 简单 vector add            | 1   | 1   | 50        | ~78%           | 带宽敏感           |
| 4 路展开 vector add        | 4   | 1   | 13        | ~20%           | 带宽敏感+计算混合  |
| 4×4 GEMM 微内核            | 4   | 4   | 4         | ~6%            | 计算密集型 (GEMM)  |
| 8 路展开 + 软件 prefetch   | 8   | 4   | 2         | ~3%            | 计算极密集          |

---

### 6.5 TLP vs ILP — 寄存器预算与 Occupancy 的权衡

**核心矛盾:** 寄存器总量是固定的 (A100 SM: 65536 个 32-bit 寄存器)。寄存器分配给 warp 越多, 能驻留的 warp 越少:

```
SM 寄存器预算:

TLP 策略 (256 threads/block, 16 regs/thread):
  每 block: 256 × 16 = 4096 寄存器
  可驻留 block: 65536 / 4096 = 16 blocks
  总 warp: 16 × 8 = 128 → 上限 64 → occupancy 100%
  每个 warp 寄存器: 16 × 32 = 512

ILP 策略 (256 threads/block, 64 regs/thread):
  每 block: 256 × 64 = 16384 寄存器
  可驻留 block: 65536 / 16384 = 4 blocks
  总 warp: 4 × 8 = 32 warps → occupancy 50%
  每个 warp 寄存器: 64 × 32 = 2048
```

**选择指南:**

| 场景                                 | 推荐策略 | 原因                                                         |
|--------------------------------------|----------|--------------------------------------------------------------|
| 内存带宽敏感 (copy, add, scale)      | TLP      | 瓶颈在带宽, 需要多 warp 来饱和 HBM                           |
| 计算受限 (matmul large tile)         | ILP      | 计算本身已掩盖访存, 更多寄存器提升 tile 大小                  |
| 混合 (stencil, reduction)            | 平衡     | 需要一定 TLP 保证带宽利用, 也需要 ILP 减少访存次数            |
| 延迟敏感 (pointer chasing, graph)    | 高 TLP   | 无法预取, 只能靠最大 warp 数来隐藏延迟                       |

> **重要认识:** Occupancy (TLP) 和 ILP 都是延迟隐藏的手段, 不是目的。最终指标是 wall-clock time。
> 很多高性能 kernel (cuBLAS, CUTLASS GEMM) 的 occupancy 只有 25-50%, 但比 100% occupancy 的 naive 实现快 10×+。
> 原因: ILP 提升带来了更少的 global memory 访问 (更大 tile), 抵消了 occupancy 的损失。

---

### 6.6 延迟隐藏的局限性 — 什么时候隐藏不了?

#### 6.6.1 所有 warp 同时 stall

**最坏场景:** 所有 warp 都在等待同一种资源, 没有就绪的 warp 可以切换。

```cuda
// 场景 1: 所有 warp 都做 global load, 打满 HBM 带宽
// → 所有 warp 都在等数据回来 → Warp Scheduler 无 warp 可切
// → 实际延迟 = 访存延迟, 隐藏失败

// 场景 2: 严重的 shared memory bank conflict
// → 所有 warp 都在 bank conflict 中串行化

// 场景 3: 所有 warp 都在等 __syncthreads
// → 阻塞在 barrier, 无 warp 可调度
```

**如何诊断:** 使用 NVIDIA Nsight Compute (ncu) 查看 stall reason:

```
Metric: sm__pipeline_util_count_warps / sm__warp_active.avg.pct_of_peak_sustained_elapsed

主要 stall reasons:
  long_scoreboard  — 等待 global/local memory (高延迟访存)
  short_scoreboard — 等待共享内存/常量/纹理
  not_selected     — warp 就绪但 scheduler 没选它 (说明已经够快!)
  wait             — 等待 barrier (__syncthreads)
  no_instruction   — 等待指令 fetch
```

#### 6.6.2 带宽墙 — 无法突破的物理限制

延迟隐藏可以隐藏延迟, 但**不能隐藏带宽限制**:

```
隐藏延迟 → 提高吞吐 → 打满 HBM 带宽 → 到达物理极限 → 无法更快

假设 A100 HBM 带宽 = 2039 GB/s, kernel 需要 2500 GB/s → 无论多少 warp, 一定跑不满
```

**结论:** 如果你的 kernel 已经打满内存带宽, 再多的 warp 也不会提速。此时优化方向是减少访存量 (更好的算法/更大的 tile/更高效的 cache 利用) 而不是增加 occupancy。

#### 6.6.3 不适用延迟隐藏的场景

某些 GPU 工作负载不适合用 TLP/ILP 隐藏延迟:

| 场景                          | 为什么不适用                                        | 替代方案               |
|-------------------------------|-----------------------------------------------------|------------------------|
| Pointer chasing / 链表遍历    | 每个 load 的地址依赖上一个 load 的结果, 不能预取     | 尽可能用加速器/合并访问 |
| 图遍历 / BFS                  | 访存模式随机, 无法 coalesce, ILP/MLP 收益有限        | 用 TLP 尽量填满带宽    |
| 强同步依赖算法               | 频繁 __syncthreads 破坏 warp 交错                    | 减少同步点, 重新设计   |

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
