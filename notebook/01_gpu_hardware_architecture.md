# 01 GPU 硬件架构基础：从零理解 GPU 怎么工作

> 对象: CUDA / GPU 零基础
> 目标: 面试能讲清楚 GPU 的物理结构和执行原理

---

## 1. 为什么要有 GPU?

先忘掉 AI。GPU 最初的工作是**图形渲染**——屏幕上几百万个像素，每个像素的颜色计算几乎是独立的。这种"海量独立小任务"的模式和 CPU 完全不一样:

| | CPU | GPU |
|------|------|------|
| 核心数量 | 几个到几十个 | 几千到上万个 |
| 单核能力 | 极强 (乱序执行、分支预测、大 cache) | 弱 (顺序执行、简单控制) |
| 擅长的任务 | 复杂逻辑、串行任务 | 数据并行、大量简单计算 |
| 延迟 vs 吞吐 | 优化延迟 (单个任务快速完成) | 优化吞吐 (海量任务整体完成) |
| 典型场景 | 操作系统、数据库、浏览器 | 渲染、矩阵乘、深度学习 |

**设计哲学**: CPU 把晶体管花在如何快: 大容量L1/L2/L3 cache、乱序执行、分支预测器、超标量流水线。GPU 把晶体管花在干得多: 海量简单的计算单元，用海量线程掩盖访存延迟。

---

## 2. GPU 的整体芯片结构

以一块典型的 NVIDIA GPU (比如 A100/H100) 为例，从外到内看:

```
                    ┌──────────────────────────────┐
                    │          GPU 芯片              │
                    │                              │
                    │  ┌─────────────┐  ┌────────┐ │
                    │  │   GPC 0     │  │ GPC n  │ │
  ┌──────┐          │  │ ┌─────────┐ │  │        │ │
  │ HBM  │←────────→│  │ │  TPC... │ │  │  ...   │ │
  └──────┘  内存总线  │  │ └─────────┘ │  │        │ │
  ┌──────┐          │  │ ┌─────────┐ │  │        │ │
  │ HBM  │          │  │ │   SM    │ │  │        │ │
  └──────┘          │  │ │   SM    │ │  │        │ │
                    │  │ └─────────┘ │  │        │ │
                    │  └─────────────┘  └────────┘ │
                    │                              │
                    │  ┌─────────────────────────┐  │
                    │  │     L2 Cache (共享)      │  │
                    │  └─────────────────────────┘  │
                    └──────────────────────────────┘
```

**层级解释:**

### 2.1 GPC (Graphics Processing Cluster)

GPU 最高层级的处理单元。一个 GPC 包含多个 TPC。从开发者视角可以直接忽略这一层，CUDA 编程中不会直接和 GPC 交互。

### 2.2 TPC (Texture Processing Cluster)

每个 GPC 含多个 TPC，每个 TPC 含 2 个 SM。这一层也在 CUDA 编程中透明。

### 2.3 SM (Streaming Multiprocessor) — 核心! 

**SM 是 GPU 的"车间"，是调度和执行的基本单元。** 面试要说清楚的就是 SM。

一个 SM 包含:
- **CUDA Cores**: 执行整数/浮点运算的算术单元
- **Tensor Cores**: 专用于矩阵乘加 (D = A×B + C) 的硬件单元
- **Warp Schedulers**: 负责选择 warp 发射指令
- **Register File**: 数千个 32-bit 寄存器 (每 SM 65536 个 on H100)
- **Shared Memory / L1 Cache**: 片上 SRAM
- **LD/ST 单元**: 负责 load/store 操作
- **SFU (Special Function Unit)**: 负责 sin/cos/sqrt/exp 等特殊函数

### 2.4 SM Partition (SM 子分区)

从 Volta 架构开始，每个 SM 被物理划分为 **4 个相同的 SM Partition**，每个 Partition 拥有:
- 自己的 Warp Scheduler
- 自己的 Register File (1/4 的 SM 寄存器)
- 自己的 CUDA Cores / Tensor Core

**但 Shared Memory 和 L1 Cache 是整个 SM 4 个 Partition 共享的。**

### 2.5 不同架构的参数对比 (面试常问)

| 架构 | 代号 | 年份 | 代表 GPU | 每 SM CUDA Core | 每 SM Tensor Core | SM 数 | HBM 带宽 |
|------|------|------|---------|:---:|:---:|:---:|------|
| SM70 | Volta | 2017 | V100 | 64 FP32 + 32 FP64 | 8 | 80 | 900 GB/s |
| SM75 | Turing | 2018 | T4, RTX 2080 | 64 FP32 + 64 INT32 | 8 | - | - |
| SM80 | Ampere | 2020 | A100 | 64 FP32 + 64 INT32 | 4 | 108 | 2.0 TB/s |
| SM90 | Hopper | 2022 | H100 | 128 FP32 + 64 INT32 | 4 | 132 | 3.35 TB/s |

**关键观察:** Hopper 每 SM 的 FP32 Core 翻倍到 128 = 每个 Partition 32 FP32 Core，所以一条 FP32 指令在一个 Partition 上只需要一个周期就能执行完一个 warp 的 32 个线程。

---

## 3. SM 内部解剖: 车间是怎么干活的

```
┌─────────────────────────────────────────────────────────────┐
│                         SM (流多处理器)                       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  │ Partition 0  │  │ Partition 1  │  │ Partition 2  │  │ Partition 3  │
│  │              │  │              │  │              │  │              │
│  │ Warp Sched 0 │  │ Warp Sched 1 │  │ Warp Sched 2 │  │ Warp Sched 3 │
│  │ Dispatch U0  │  │ Dispatch U1  │  │ Dispatch U2  │  │ Dispatch U3  │
│  │              │  │              │  │              │  │              │
│  │ 16 FP32 Core│  │ 16 FP32 Core │  │ 16 FP32 Core │  │ 16 FP32 Core │
│  │ 16 INT32 Cor│  │ 16 INT32 Cor │  │ 16 INT32 Cor │  │ 16 INT32 Cor │
│  │  8 FP64 Core│  │  8 FP64 Core │  │  8 FP64 Core │  │  8 FP64 Core │
│  │  4 SFU      │  │  4 SFU       │  │  4 SFU       │  │  4 SFU       │
│  │  1 TensorCor│  │  1 TensorCor │  │  1 TensorCor │  │  1 TensorCor │
│  │  RegFile 16K│  │  RegFile 16K │  │  RegFile 16K │  │  RegFile 16K │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Shared Memory / L1 Cache (共享)              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**每个时钟周期的执行流程:**
1. **Warp Scheduler** 从就绪 warp 中选择一个
2. **Dispatch Unit** 把指令发到执行单元
3. 16 个 FP32 Core 各执行 2 个线程 (32/16=2 cycles，Hopper 只需 1 cycle)
4. 如果当前 warp 因为访存 stall，scheduler 立刻切换到另一个就绪 warp

**这就是 GPU 隐藏延迟的核心手段: 零开销的 warp 切换。**
- 更换 warp id 就能更换运行的线程，是因为每个 warp id 都关联了一套静态、完整的硬件状态（每个 warp 都有一个独立的程序计数器（PC），并且这个 PC 和它的寄存器一样，也是静态固定在硬件里的。），其中最关键的两个是：寄存器基址（用于数据）和程序计数器（用于指令地址）。调度器本质上就是一个高速的多路选择器，它选中哪个 id，就把对应的一套 PC 和寄存器基址输出到执行流水线。所以，更换 id 的一瞬间，下一条指令的地址就已经确定了——它就是那个 warp 的 PC 当前值。

---

## 4. SM 中的执行单元详解

### 4.1 CUDA Core

CUDA Core 执行整数和单精度浮点运算。一条 FADD/FMUL/FFMA 指令在 CUDA Core 上执行。
- “图中哪个物理单元是 CUDA Core”，那就是 16 FP32 Core（以及可能还包括 16 INT32 Core）。
- “CUDA Core 在整个 SM 中扮演什么角色”，它就是执行每个线程标量运算（如加法、乘法、乘加）的具体电路，是整个 SM 里数量最多的运算单元，负责处理绝大多数基本算术指令。

**注意:** CUDA Core 不等于 thread! 一个 CUDA Core 在一个时钟周期内执行一个操作，但一个 warp (32 threads) 会由 16 个 FP32 Core 在 2 个周期内完成 (Hopper 架构下 32 Core 只需 1 周期)。

### 4.2 Tensor Core

Tensor Core 是专门加速矩阵乘加 (D = A×B + C) 的硬件。比 CUDA Core 快 8-16 倍。

| 架构 | 每 SM Tensor Core | 一次 MMA tile | FP16 TFLOPS |
|------|:---:|------|:---:|
| Volta (V100) | 8 | m8n8k4 | 125 |
| Turing (T4) | 8 | m8n8k4 (+INT8/INT4) | 65 |
| Ampere (A100) | 4 | m16n8k16 (+TF32/BF16) | 312 |
| Hopper (H100) | 4 | m16n8k16 (+FP8) | 990 |

每个 Tensor Core 每个时钟周期完成一个 m×n×k 的矩阵乘加 (FMA)。
- Warp 不直接“运行”在任何特定的 Core 上，而是通过调度器将不同类型的指令派发给不同的专用硬件单元（CUDA Core、Tensor Core、LSU 等）。CUDA Core 负责标量运算（每线程独立），Tensor Core 负责矩阵运算（整个 Warp 协作）。这就是 GPU 异构计算的精髓。
- Tensor Core 以 Warp 为单位提供服务。当 Warp 执行一条矩阵指令时，Tensor Core 作为一个整体硬件单元，一次性处理该 Warp 所请求的整个矩阵运算（比如 D = A×B + C，其中 A、B、C 都是分布在 Warp 内各线程的寄存器中的小矩阵块）。这个过程中，Warp 内 32 个线程的寄存器作为输入输出缓冲区，Tensor Core 在后台完成计算，不需要像 CUDA Core 那样把一个 Warp 拆成多批。

### 4.3 SFU (Special Function Unit)

执行 sin/cos/sqrt/exp/log 等超越函数。数量少 (per Partition 4 个)，延迟高，避免频繁使用。

### 4.4 LD/ST Unit

负责 global/local/shared memory 的 load 和 store 操作。

---

## 5. 从硬件到软件: CUDA 编程模型映射

**面试必考: 软件层级和硬件的对应关系。**

| 软件概念 | 硬件实体 | 说明 |
|------|------|------|
| **Thread** | CUDA Core 上执行的一个数据流 | 软件层面的最小执行单元 |
| **Warp (32 threads)** | SM Partition 上统一调度执行 | 硬件调度的基本单位 |
| **Thread Block** | 一个 SM | block 内所有 thread 在同一个 SM 上 |
| **Grid** | 整个 GPU (所有 SM) | 一次 kernel launch 的所有 block |
| **Shared Memory** | SM 内的 SRAM | block 内所有 thread 共享 |
| **Register** | SM 内的 Register File | 每个 thread 私有 |
| **Global Memory** | HBM/VRAM | 整个 grid 共享，CPU 也可访问 |

### 5.1 Block → SM 的绑定规则

1. 一个 block 只能在一个 SM 上执行，不能跨 SM
2. 一个 SM 可以同时驻留多个 block (只要资源够)
3. block 被分配到哪个 SM 由硬件调度器决定，开发者不可控
4. 一旦分配，block 在 SM 上运行到结束

---

## 6. 关键硬件限制 (面试数字题)

**以 A100 (SM80) 为例:**

| 限制项 | 数值 | 影响 |
|------|:---:|------|
| 每 SM 最大线程数 | 2048 | occupancy 上限 |
| 每 SM 最大 block 数 | 32 | block 数量上限 |
| 每 block 最大线程数 | 1024 | 编程时的 blockDim 上限 |
| 每 SM 寄存器数 | 65536 (32-bit) | 每个 thread 可用寄存器 = 65536/线程数 |
| 每 SM shared memory | 最大 164 KB | block 间瓜分 |
| 每 thread 最大寄存器 | 255 | kernel 编译限制 |
| Warp size | **32** | 锁死，不能改 |

**Occupancy 计算示例:**

```
blockDim = 256 → 每个 block 有 256/32 = 8 warps
每 thread 用 64 个寄存器 → block 用 256×64 = 16384 寄存器
每 SM 65536 寄存器 → 最多驻留 65536/16384 = 4 blocks
4 blocks × 256 threads = 1024 threads → 1024/2048 = 50% occupancy
```

---

## 7. GPU vs CPU 架构差异深度对比

| 维度 | CPU | GPU |
|------|------|------|
| 核心数 | 数~数十 | 数千~上万 |
| 每个核心 | 超标量、乱序、大 cache | 顺序、小 cache |
| 线程切换开销 | 大 (保存/恢复上下文) | **零** (上下文就在寄存器里) |
| 内存延迟处理 | 大 cache 减少 miss | 海量线程掩盖延迟 |
| 分支预测 | 复杂预测器 | 无 (warp divergence) |
| FP64:FP32 比 | 1:1 | 1:32 或更低 |
| SIMD 宽度 | 256/512 bit (AVX) | 32 threads (warp) |

**核心哲学差异:** CPU 是"让一个任务快"，GPU 是"让一百万个任务一起完成"。
- 通过硬件支持的、零开销的、极高频的线程/Warp切换，使得计算单元在等待内存的时间片里，永远有其他就绪的计算任务可以执行。
- “上下文就在寄存器里” = 每个线程拥有专属、永驻的寄存器组，无需在切换时保存/恢复任何状态。
- 零开销切换 = 硬件调度器只改变一个指向当前寄存器的索引，一个时钟周期内就能换一组线程执行。
- 这也解释了 GPU 的局限：若一个线程需要太多寄存器（如复杂的递归、大量局部数组），能同时驻留的线程数就会下降，进而无法填满内存延迟的隐藏槽，性能会骤降。这正是 GPU 编程中“寄存器压力”需要优化的原因。
---

## 8. 各代架构关键演进

### Fermi (2010, SM20)
- 第一个完整的 GPU 计算架构
- 引入 L1/L2 cache
- 每 SM 32 CUDA Core

### Kepler (2012, SM30/35)
- **引入 Warp Shuffle** (这就是我们后面要学的基础!)
- 每 SM 192 CUDA Core
- 引入动态并行 (device launch device)

### Maxwell (2014, SM50)
- 能效比大幅提升
- 每 SM 128 CUDA Core
- 引入 shared memory 大小可配置

### Pascal (2016, SM60/61)
- HBM2, NVLink
- 统一内存 (Unified Memory)
- FP16 支持 (用于推理)

### Volta (2017, SM70)
- **引入 Tensor Core** (AI 计算的转折点)
- Independent Thread Scheduling (独立线程调度，允许 warp 内分支)
- 每 SM 64 FP32 + 32 FP64 Core

### Turing (2018, SM75)
- **RT Core** (光线追踪)
- INT8/INT4 Tensor Core
- **SM Partition 机制成熟**

### Ampere (2020, SM80)
- **TF32** (训练加速，19-bit)
- **BF16 支持**
- 每 SM 4 Tensor Core (但每个更强)
- 异步拷贝: `cp.async`
- L2 cache 扩大到 40 MB

### Hopper (2022, SM90)
- **FP8** 支持 (H100)
- **TMA (Tensor Memory Accelerator)**: 异步地址计算+数据搬运
- FP32 Core 翻倍到 128/SM
- L2 cache 扩大到 50 MB

### Blackwell (2024, SM100)
- FP4 支持
- L2 cache 扩大到 126 MB
- TMEM (Tensor Memory) 专门为 Tensor Core 提供数据

---

## 9. 一个简单的思维模型

把 GPU 想像成一个工厂:

```
GPU = 富士康大工厂
SM = 独立车间 (A100 有 108 个车间)
SM Partition = 车间里的 4 条流水线
Warp Scheduler = 流水线拉长 (每条流水线一个)
CUDA Core = 流水线上的工人
Warp = 一批 32 个待加工的零件
Thread Block = 分配给一个车间的完整生产任务

拉长的工作: 看到某批零件还在等原材料 (访存)，立刻喊下一批零件上线加工。
这就是 GPU 隐藏内存延迟的奥秘——"生产不停，流水不断"。
```

---

## 10. 面试高频问题

### Q1: 什么是 SM? 它包含哪些部分?
**答:** SM (Streaming Multiprocessor) 是 GPU 的调度和执行基本单元。包含 Warp Scheduler、CUDA Cores、Tensor Cores、Register File、Shared Memory/L1 Cache、SFU、LD/ST 单元。一个 SM 被分为 4 个 SM Partition。

### Q2: 一个 SM 能同时执行多少个 warp?
**答:** 每个 SM Partition 的 Warp Scheduler 每个时钟周期只能发射 1 条指令 (来自 1 个 warp)。4 个 Partition 同时可发射 4 条指令，对应 4 个 warp。但 SM 可以驻留多达 64 个 warp (2048 threads)，通过快速切换来掩盖延迟。
- 一个时钟周期内：SM 最多执行 4 个 Warp（每个 Warp 执行 32 个线程的同一条指令）。
    - SIMT（单指令多线程），在一个 Warp 的 32 个线程中，所有线程在当前时钟周期执行的指令地址（PC）是完全相同的。也就是说，这 32 个线程共享同一个程序计数器（PC）。虽然指令相同，但每个线程操作的数据可以完全不同。这是通过两个机制实现的：
        1. 不同的寄存器：每个线程拥有自己专用的寄存器集合。加法指令中的“源寄存器B”和“源寄存器C”是相对于每个线程自己的寄存器编号。比如：
        2. 不同的内存地址：如果指令是加载（LOAD）或存储（STORE），每个线程可以计算不同的地址。例如，访问数组的第 i 个元素：data\[thread_id\]，那么每个线程会去不同地址拿数据。
    - 按照“同一条指令”的要求，32 个线程无法同时执行不同的路径。解决方案是：
        - 串行化 + 掩码：Warp 调度器先执行偶数线程的路径，此时奇数线程被掩码（mask） 禁用（它们不写结果，也不产生副作用）。执行完偶数路径后，再执行奇数路径，此时掩码反过来。
        如：
        ```cpp
        if (thread_id % 2 == 0) {
            // 偶数线程做 A
        } else {
            // 奇数线程做 B
        }
        ```
        - 这就是 发散（divergence） 的性能代价：本来一个周期能完成，现在需要两个或更多周期。
- 同时驻留：SM 上最多可以存放 48 或 64 个 Warp（取决于架构），它们不都在执行，而是被调度器以极快的速度轮换，从而隐藏延迟。
所以，同时执行的 Warp 数（4）远小于最大驻留 Warp 数（64）。后者是为了让调度器总有足够多的就绪 Warp 来填充每周期 4 个发射槽位，避免执行单元空闲。

### Q3: 为什么 warp size 是 32?
**答:** 硬件设计选择。32 个线程一组刚好映射到一个 SM Partition 的执行单元，是 SIMT 模型的天然粒度。AMD 的 wavefront 是 64。
- 它是硬件设计的具体工程妥协结果：让一个 Warp 的线程数正好是每个时钟周期可执行单元数量的整数倍（通常是 2 倍），既保证足够的并行度以隐藏内存延迟，又控制分支发散的开销在可接受范围内，同时简化掩码和调度逻辑。这个数字不是算法决定的，而是由晶体管预算、面积、功耗和仿真权衡出的经验值。

### Q4: CUDA Core 和 Tensor Core 有什么区别?
**答:** CUDA Core 执行通用标量运算 (FADD/FMUL/FFMA)。Tensor Core 是专门的矩阵乘加单元，一个指令完成 m×n×k 的 tile 矩阵乘法，吞吐是 CUDA Core 的 8-16 倍。只适用于矩阵乘模式。

### Q5: GPU 凭什么比 CPU 快?
**答:** 
1. 海量核心: 数千核心并行计算
2. 零开销线程切换: 上下文就在寄存器里
3. 高带宽 HBM: A100 2TB/s vs CPU DDR5 ~100GB/s
4. Tensor Core: 专用矩阵乘硬件

### Q6: Occupancy 是什么意思?
**答:** 驻留在 SM 上的 active warp 数除以理论最大 warp 数的比率。高 occupancy 意味着更多的 warp 可用于隐藏延迟。但不是越高越好——有时候用更少的 warp 但更多的寄存器 per thread 反而更快。

---

## 11. 参考链接

- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CUDA C++ Programming Guide - Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)
- [LeetCUDA README](/home/hz/code/cpp/third_party/LeetCUDA/README.md)
- [CUDA 如何调度 kernel 到指定的 SM - 知乎](https://www.zhihu.com/question/652642080/answer/1985070382152184624)

---

## 12. 学习检查清单

- [ ] 能画出 SM 的内部结构框图
- [ ] 理解 CPU 和 GPU 的设计哲学差异 (latency vs throughput)
- [ ] 能说清楚 thread → warp → block → grid 和硬件的对应关系
- [ ] 知道 SM Partition 的概念和 SM 的 4 个子分区结构
- [ ] 能计算给定 blockDim 和寄存器用量的 occupancy
- [ ] 知道各代架构关键演进 (Volta→Tensor Core, Ampere→TF32/BF16, Hopper→FP8/TMA)
- [ ] 理解 warp 切换为什么是零开销的
- [ ] 能回答上面 6 个面试问题
