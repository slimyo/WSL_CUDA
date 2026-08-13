
# Tensor Cores 精度体系与 WMMA 入门：深度解析版
> **Learning Path**: 阶段 3 — AI 推理核心
> **Prerequisites**: `reduce_warp_learning.md`, `bank_conflict_learning.md`
> **Code Ref**: `third_party/LeetCUDA/kernels/hgemm/`
---
## 1. 精度体系：从 FP32 到 FP4 的取舍
**核心权衡**：Range（动态范围）vs Precision（精度）。
训练主要怕 overflow（溢出），推理主要怕 underflow/precision loss（精度丢失）。
| 格式 | Bits | Exp | Mantissa | Range (Approx) | 关键特性与陷阱 |
|------|:---:|:---:|:---:|------|------|
| **FP32** | 32 | 8 | 23 | ~3.4e38 | 黄金标准，但也浪费带宽。 |
| **TF32** | 19 | 8 | 10 | 同 FP32 | **A100 特有**。保持 FP32 Range，精度降至 FP16 水平。软件层面无缝替代 FP32。 |
| **BF16** | 16 | 8 | 7 | 同 FP32 | **LLM 训练首选**。Exp 与 FP32 相同，无需 Loss Scaling。缺点是尾数位太少，累加时需注意精度。 |
| **FP16** | 16 | 5 | 10 | ~65504 | **推理主流**。陷阱：Range 极窄（65504）。Attention Score 或 Adam 计算中极易溢出，必须配合 Loss Scaling。 |
| **FP8 E4M3** | 8 | 4 | 3 | ~448 | **H100 推理**。精度更高，Range 极小，常用于 Activations。 |
| **FP8 E5M2** | 8 | 5 | 2 | ~57344 | **H100 训练**。Range 大，精度差，常用于 Gradients。 |
### 实战细节：累加器精度
在 Kernel 编写中，FP16 的累加通常在 Tensor Core 内部以 **FP32** 精度执行，最后写回时再转回 FP16，以防止精度崩塌。
---
## 2. Tensor Core 原理：硬件视角
**公式**：$D = A \times B + C$
**本质**：Tensor Core 是一个 **SIMT 之上的矩阵乘加单元**。
### 计算吞吐 vs. 数据搬运
- **算力**：Tensor Core 比 CUDA Core 快 8-16x。例如 A100 FP16 算力 312 TFLOPS。
- **瓶颈**：通常是 **Memory Bandwidth** 和 **Register/Shared Memory Pressure**。
- **关键指标**：Arithmetic Intensity（计算密度）。只要数据能喂饱 Tensor Core，性能就能起飞。
---
## 3. WMMA API：初学者的把手
WMMA (Warp Matrix Multiply Accumulate) 是 NVIDIA 提供的高层 C++ API。它隐藏了底层的寄存器映射细节。
```cpp
#include <mma.h>
using namespace nvcuda::wmma;
// 1. 定义 Fragment (寄存器分配)
// 一个 fragment 占用整个 Warp 的寄存器资源
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag; // 累加器建议用 float
// 2. 加载
wmma::load_matrix_sync(a_frag, (const half*)sram_ptr_a, 16); 
wmma::load_matrix_sync(b_frag, (const half*)sram_ptr_b, 16);
// 3. 计算
wmma::fill_fragment(c_frag, 0.0f);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
// 4. 存储
wmma::store_matrix_sync((float*)dst_ptr, c_frag, 16, wmma::mem_row_major);
```
---
## 3.5 深度对比：Warp Reduce (SIMT) vs. WMMA (Tensor Core)
以常见的 **Softmax**（使用 Warp Reduce）和 **HGEMM**（使用 WMMA）为例，这两者代表了 GPU 计算的两种完全不同的哲学。理解它们的区别是优化 Flash Attention 等复杂 Kernel 的基石。
### 1. 硬件执行单元与指令集
| 特性 | Warp Reduce (Softmax) | WMMA (HGEMM) |
|------|------|------|
| **执行单元** | **CUDA Cores** (FP32/FP64 Unit) | **Tensor Cores** (矩阵乘法专用单元) |
| **指令粒度** | **Scalar/Vector**：一条指令处理 1 个数据 (如 `FADD`, `FMUL`) | **Matrix**：一条指令处理一个矩阵块 (如 `HMMA.m16n8k16`) |
| **计算密度** | 低 (1 Op / cycle / thread) | 极高 (如 $16 \times 8 \times 16 = 2048$ 次乘加 / cycle) |
**关键洞察**：
- **Softmax** 包含 `exp`, `div` 等非线性操作，Tensor Core 无法处理，必须用 CUDA Core。
- **HGEMM** 全是乘加运算，是 Tensor Core 的主场。
### 2. 线程角色与数据视角
这是新手最容易困惑的地方：**"我的数据在哪个线程里？"**
#### Warp Reduce (Softmax) 模式
- **线程角色**：**独立战士**。每个线程拥有独立的数据所有权。
- **数据视角**：
  - Thread $i$ 负责 $x_i$。
  - 计算 Max/Sum 时，线程之间通过 Shuffle (`__shfl_xor_sync`) 交换数据，完成 Warp 级别的归约。
- **代码特征**：
  ```cuda
  // 每个 thread 独立执行，逻辑清晰
  float val = input[tid];
  // 跨 warp 交换数据做归约
  for (int offset = 16; offset > 0; offset /= 2) {
      val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  ```
#### WMMA (Tensor Core) 模式
- **线程角色**：**协作齿轮**。单个线程**没有**完整的矩阵数据，它只持有一个"碎片" (Fragment)。
- **数据视角**：
  - 一个 Warp (32 threads) 共同"拥有"一个 $16 \times 16$ 的矩阵块。
  - Thread $i$ 持有矩阵 A 和 B 的若干散乱元素。这些元素在寄存器中的布局是**不透明**的，由 Hardware-defined Layout 决定。
- **代码特征**：
  ```cuda
  // 看起来像变量，实际是寄存器集合
  // 整个 Warp 必须同步加载，单个 thread 无法独立操作 a_frag
  wmma::fragment<wmma::matrix_a, ...> a_frag;
  wmma::load_matrix_sync(a_frag, smem_ptr, 16); 
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  ```
### 3. 性能瓶颈分析
| 瓶颈类型 | Warp Reduce (Softmax) | WMMA (HGEMM) |
|------|------|------|
| **Memory Bandwidth** | **主要瓶颈**。Softmax 是典型的 Bandwidth-bound。每个数据做几次运算就写回，利用率低。 | **次要瓶颈**（如果流水线做得好）。Tensor Core 计算太快，往往需要 Double Buffering 来喂饱它。 |
| **Instruction Latency** | 敏感。循环内的 `max` 或 `sum` 操作有依赖链，难以指令级并行 (ILP)。 | 不敏感。一条 `mma_sync` 指令内部高度并行。 |
| **Register Pressure** | 低。只需要少量寄存器保存当前元素。 | **极高**。一个 Fragment 可能占用几十个寄存器，Occupancy 容易因寄存器不足而下降。 |
### 4. 为什么 Softmax 不用 Tensor Core？
1.  **算子不匹配**：Tensor Core 本质是 $D = A \times B + C$。Softmax 包含 `Exp` 和 `Reduce`。
    -   `Reduce` 是沿着行/列压缩维度，不是矩阵乘法。
    -   `Exp` 是逐元素非线性运算，Tensor Core 无法处理。
2.  **数据流差异**：Softmax 需要两遍扫描，且中间结果交互复杂，不适合 Tensor Core 的"大块数据吞吐"模式。
### 5. 实战启示：Flash Attention 的融合
Flash Attention 是理解这两种模式结合的经典案例：
```
Flash Attention Kernel 分解:
1. Q x K^T (Matmul) -> 使用 WMMA / Tensor Core (核心算力瓶颈)
2. Softmax (Reduce + Exp) -> 使用 CUDA Core / Warp Shuffle (内存瓶颈，数据局部性关键)
3. P x V (Matmul) -> 使用 WMMA / Tensor Core
```
**优化逻辑**：
- 如果纯用 WMMA，中间结果 $S = QK^T$ 会写回 HBM，Softmax 再读回来，带宽爆炸。
- Flash Attention 将 Softmax 这一步的 **Warp Reduce 逻辑** 嵌入到 WMMA Kernel 的 **Epilogue** 阶段。
- **数据流**：Tensor Core 算完 $QK^T$ -> 结果留在 Shared Memory -> 直接用 CUDA Core 跑 Softmax -> 结果喂给下一轮 Tensor Core。
---
## 4. WMMA → MMA PTX：性能进阶之路
WMMA 简单，但黑盒限制了极致性能。MMA PTX (Inline Assembly) 允许精细控制寄存器布局。
### MMA PTX 优势
1.  **布局控制**：WMMA 隐藏了寄存器布局，导致难以优化 Epilogue。MMA PTX 让你知道数据在哪个寄存器。
2.  **指令融合**：可以将 MMA 结果直接用于后续指令，减少寄存器读写。
3.  **更小的 Tile**：WMMA 最小 16x16x16，MMA PTX 支持 `m16n8k16`，减少了寄存器压力。
### 代码示例：m16n8k16
```cpp
// A: 16x16, B: 16x8, C: 16x8
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
    "{%0, %1, %2, %3}, "  // D: 4 regs
    "{%4, %5, %6, %7}, "  // A: 4 regs
    "{%8, %9}, "          // B: 2 regs
    "{%10, %11, %12, %13};" // C: 4 regs
    : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]), "r"(b[1]),
      "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
);
```
### 进阶难点：ldmatrix 指令
MMA PTX 最难的地方在于**数据喂给 Tensor Core 前，必须重排到特定的寄存器布局**。
`ldmatrix` 指令是 SM75+ 的神器，它能直接把 Shared Memory 数据加载成 MMA 需要的寄存器布局，避免了繁琐的手动 Shuffle。
---
## 5. Hopper SM90：wgmma 与异步计算
**痛点**：SM80 中，Warp 发射 MMA 指令后，必须等待 Tensor Core 返回结果，导致 Warp Stall。
**wgmma (Warpgroup MMA)**：
- **Warp Group**：4个 Warp (128 threads) 协同工作。
- **异步执行**：Warp 发射 `wgmma` 指令后，**立即释放**，可以去计算下一个 Tile。计算和存储访问完全流水线化。
```cpp
// Hopper 伪代码流程
// 1. TMA 异步加载 Global -> Shared
// 2. wgmma 异步计算
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
    "{%0, ...}, %1, %2, %3, 1, 1, 0, 0;"
    : ...
    : "r"(addr_a), "r"(addr_b), "r"(addr_c));
```
---
## 6. Blackwell SM100：Tensor Memory (TMEM)
**SM90 的瓶颈**：wgmma 累加器占用大量寄存器（RF），限制了 Occupancy。
**SM100 的解法：Tensor Memory (TMEM)**
- **定义**：一种全新的、仅 Tensor Core 可访问的 SRAM。
- **位置**：位于 SM 内部，但在 Register File 之外。
- **优势**：累加器直接放在 TMEM，不再占用宝贵的 RF，寄存器压力骤降。
---
## 7. 总结与检查清单
| 特性 | Warp Reduce | WMMA | MMA PTX | wgmma |
|------|------|------|------|------|
| **典型算子** | Softmax, LayerNorm | HGEMM (入门) | HGEMM (极致优化) | FlashAttn v2/v3 |
| **线程角色** | 独立战士 | 协作齿轮 | 协作齿轮 | 协作齿轮组 |
| **瓶颈** | 带宽 | 计算密度 | 寄存器压力 | 流水线延迟 |
**检查清单：**
- [ ] **精度理解**：为什么 BF16 训练不需要 Loss Scaling？
- [ ] **线程视角**：WMMA 中，Thread $i$ 能独立读取 Fragment 中的第 $j$ 个元素吗？（答案：不能，布局不透明）
- [ ] **混合编程**：能讲清 Flash Attention 是如何在一个 Kernel 里调度 Tensor Core (Matmul) 和 CUDA Core (Softmax) 的吗？
- [ ] **架构理解**：Hopper 的 wgmma 为何能解决 Warp Stall 问题？(异步执行 + Producer/Consumer 模式)
