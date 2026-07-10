# Transformer Decoder 架构：LLaMA 拆解
> **对象**: LLM 推理入门 / 算子岗
> **目标**: 面试能手画 Llama decoder layer 数据流图，说清每个组件的张量形状、数学原理、访存模式及算子优化策略。
> **核心差异**: 重点关注 **KV Cache**、**算子融合** 以及 **Prefill vs Decode** 的性能差异。
---
## 1. Llama Decoder Layer 数据流全景
> **图例说明**:
> `[B, S, H]`: Batch, Sequence, Hidden Dim
> `GEMM`: 矩阵乘法 (General Matrix Multiply)
> `ElemW`: 逐元素操作
> `KV Cache`: 推理时缓存的 Key/Value 矢量
```text
Input x [B, S, H] (来自上一层或 Embedding)
   │
   ├──────────────────────┐ (用于 Residual Add)
   │                      │
   ▼                      │
┌──────────────┐          │
│  RMSNorm     │  ElemW   │  (Pre-Norm: 保障训练稳定, 推理可融合入后续 GEMM)
└──────┬───────┘          │
       │ x_norm           │
       ▼                  │
┌───────────────────────────────────────────┐
│  1. QKV Projection (GEMM)                 │
│     Q = x @ Wq, K = x @ Wk, V = x @ Wv   │
│     Output: Q[B,S,H], K[B,S,H_k], V[B,S,H_v]
└───────┬───────────────┬───────────────────┘
        │               │
        ▼               ▼
┌──────────────┐  ┌──────────────┐
│  RoPE(Q, K)  │  │ KV Cache     │◄─────┐ (推理核心: PagedAttention)
│  (ElemW)     │  │ Read/Write   │      │
└──────┬───────┘  └──────┬───────┘      │
       │                 │              │
       ▼                 ▼              │
┌───────────────────────────────────────────┐
│  2. Attention Core                        │
│     Attn = Softmax(Q @ K.T) @ V           │
│     Output: Context [B, S, H]             │
└───────────────────┬───────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ Out Proj (GEMM)│  Wo @ Attn_Out
            └───────┬───────┘
                    │ out
    ┌───────────────┴───────────────┐
    │          Residual Add         │  x + out
    └───────────────┬───────────────┘
                    │ x_attn
    ┌───────────────┴───────────────┐
    │          RMSNorm              │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────┐
    │  3. FFN (SwiGLU)                       │
    │     Gate = x @ W_gate   (GEMM)         │
    │     Up   = x @ W_up     (GEMM)         │
    │     Hidden = SiLU(Gate) * Up (ElemW)   │
    │     Out   = Hidden @ W_down (GEMM)     │
    └───────────────┬───────────────────────┘
                    │
    ┌───────────────┴───────────────┐
    │          Residual Add         │
    └───────────────┬───────────────┘
                    │
                    ▼
            Output [B, S, H] (去往下一层)
```
---
## 2. 核心组件深度拆解
### 2.1 RMSNorm (Root Mean Square Layer Normalization)
**数学对比：**
```cpp
// RMSNorm (LLaMA 使用): 移除了均值统计，无需中心化
rms = sqrt(mean(x²) + ε)
y = (x / rms) * γ
// LayerNorm: 包含均值统计和偏置
mu = mean(x)
sigma = sqrt(mean((x - mu)²) + ε)
y = γ * ((x - mu) / sigma) + β
```
*   **优势**: 省去了一次 Reduce (求 `mu`) 和一次向量加法 (`x - mu`)。在 CUDA 实现中，减少了一次全局同步和显存读写。
**CUDA/算子实现要点：**
*   **Fusion (融合)**: 推理中通常**不单独写** RMSNorm Kernel，而是将其作为 **Epilogue** 融合进前面的 GEMM 或 Attention Kernel 中，避免写回 HBM 再读出。
*   **Kernel 结构** (如果独立实现):
    *   **Grid/Block**: `Grid = (Batch * Seq)`, `Block = (Hidden Dim / VecSize)`。
    *   **Reduce**: 使用 Warp Shuffle (`__shfl_down_sync`) 或 Shared Memory 进行 `x²` 的求和。
    *   **Numerics**: 注意 `rsqrtf` 的精度，有时需要使用 `rsqrt` intrinsic + Newton 迭代保证精度。
### 2.2 RoPE (Rotary Position Embedding)
**数学原理：**
将绝对位置编码注入到 Query 和 Key 的向量空间中，通过**旋转矩阵**实现。
*   **性质**: $f(q, m)^T f(k, n) = g(q, k)^T g(k, n-m)$。即点积仅依赖于相对位置 $(m-n)$。
*   **推理优势**: 位置信息**可加**。新 token 的 K/V 计算完后直接旋转写入 Cache，无需重算历史 token 的位置编码。
**CUDA/算子实现要点：**
*   **计算密集度**: 极低（主要为 `sincos` 指令），属于 **Memory Bound**。
*   **优化手段**:
    *   **Vectorization**: 使用 `float4` (128bit) 加载 `x`，一次计算 4 个 float 或 2 个 half2 的旋转。
    *   **SFU (Special Function Unit)**: GPU 的 SFU 计算 `sincos` 非常快，但要避免线程分歧。
    *   **Fusion**: 通常与 QKV Projection 之后的结果直接融合，或者融合进 Attention 的 Kernel 开始阶段。
### 2.3 SwiGLU (FFN 激活函数)
**结构分解：**
```cpp
// 三个 GEMM 权重矩阵
Gate = X @ W_gate   // [B, S, H] @ [H, 4H/3] -> [B, S, 4H/3]
Up   = X @ W_up     // [B, S, H] @ [H, 4H/3] -> [B, S, 4H/3]
Hidden = Swish(Gate) ⊙ Up  // Element-wise 乘法
Out  = Hidden @ W_down     // [B, S, 4H/3] @ [4H/3, H] -> [B, S, H]
```
*   **参数量**: 相比标准 FFN (2个矩阵)，SwiGLU 有 3 个矩阵，参数量增加了 1.5 倍，但效果显著提升。
**CUDA/算子实现要点：**
*   **Kernel 分解**:
    1.  **Gate/Up GEMM**: 两个小 GEMM。在推理中，若 Batch 够大可合并为一个 Kernel；若 Decode 阶段 (M=1)，通常是两个 GEMV (或 FP8 GEMM)。
    2.  **Activation**: 逐元素操作，使用 CUDA Core 或 Tensor Core (如果使用 Tensor Cores 做 Epilogue)。
    3.  **Down GEMM**: 最大的计算量所在。
*   **Fusion**: `GEMM(Gate) -> Swish` -> `GEMM(Up) -> Mul` -> `GEMM(Down)` 是一个典型的**多层融合**机会，中间结果全部留在寄存器/Shared Memory，不写回 HBM。
### 2.4 GQA (Grouped-Query Attention) - 推理加速核心
**原理：**
*   **MHA**: $n_q$ 个 Q head 对应 $n_k$ 个 K/V head。KV Cache 显存占用巨大。
*   **MQA**: 所有 Q head 共享 1 组 K/V head。显存最小，但精度受损。
*   **GQA**: $n_q$ 个 Q head 分组共享 $n_k$ 个 K/V head ($n_k = n_q / G$)。平衡了显存与精度。
**推理影响 (面试必问)：**
*   **显存占用**: KV Cache 大小减少 $G$ 倍。这是大模型（如 Llama-3 70B）能长文本推理的关键。
*   **Kernel 实现**: Attention Kernel 中，需要处理 Q head 重复广播 K/V 的逻辑。这影响了 **Shared Memory** 的 Tiling 策略（K/V 块加载后要广播给多个 Q block）。
---
## 3. 性能分析：Prefill vs Decode (AI Infra 核心)
| 组件 | 算子类型 | Prefill 阶段 (S 大) | Decode 阶段 (S=1) | 关键优化点 |
| :--- | :--- | :--- | :--- | :--- |
| **RMSNorm** | Reduce + EW | **Memory Bound** (高并发) | **Latency Bound** (单次操作极小) | 融合进 GEMM Epilogue |
| **RoPE** | EW (Math) | **Memory Bound** | **Latency Bound** | 向量化加载, FP16/BF16 处理 |
| **QKV Proj** | GEMM | **Compute Bound** ( Tensor Core ) | **Memory Bound** (Weight Loading) | 1. FlashAttention / Fused MHA<br>2. **INT8/FP4 量化** (Decode时权重访存是瓶颈) |
| **Attention** | GEMM + Softmax | **Compute Bound** (QK^T 大矩阵乘) | **Memory Bound** (读 KV Cache 是瓶颈) | 1. **FlashAttention-2** (Tiling)<br>2. **PagedAttention** (解决 KV 内存碎片)<br>3. **KV Cache 量化** |
| **FFN (SwiGLU)** | 3x GEMM | **Compute Bound** | **Memory Bound** | 同 QKV Proj，Decode 阶段极度依赖权重量化 |
| **Output** | GEMM | **Compute Bound** | **Memory Bound** | 1. Weight Streaming<br>2. Speculative Decoding (采样层优化) |
> **面试金句**：
> *   **Prefill 是算力游戏**：序列长，矩阵大，拼的是 Tensor Core 利用率和 FLOPs。
> *   **Decode 是带宽游戏**：序列为 1，矩阵乘法退化为向量乘法，计算量很小，绝大部分时间花在从 HBM 读取权重 (几十 GB) 和 KV Cache 上。
> *   **算子优化目标**：Prefill 追求高 Throughput (吞吐)，Decode 追求低 Latency (延迟) 和高 Cache Hit Rate。
---
## 4. LeetCUDA 组件源码索引
| Llama 组件 | LeetCUDA 路径 | 关键 Kernel / 实现技巧 |
| :--- | :--- | :--- |
| **RMSNorm** | `kernels/rms-norm/` | `rms_norm.cu`: `warpReduceSum` 实现 reduce，避免 shared memory 冲突 |
| **RoPE** | `kernels/rope/` | `rope.cu`: `sincos` 计算，`float4` 向量化加载 (`load_float4`) |
| **SwiGLU** | `kernels/swish/`, `kernels/sgemm/` | `swish.cu`: `x * sigmoid(x)`; SGEMM 用于 Gate/Up/Down 投影 |
| **Attention** | `kernels/flash-attn/` | `flash_attn_mma.py`: 使用 WGMMA (Tensor Core) 指令在线 Softmax |
| **GEMV (Decode)**| `kernels/hgemv/` | 针对单 Token 生成的半精度向量乘法优化，针对 Weight Bound 场景 |
| **Elementwise**| `kernels/elementwise/` | 展示通用的逐元素算子向量化模板 |
---
## 5. 学习检查清单 (针对面试)
- [ ] **手绘图**: 能在白板上画出第 1 节的完整流程图，并在旁边标注每个矩阵的形状 $[B, S, H]$。
- [ ] **KV Cache**: 解释 PagedAttention 是如何解决 KV Cache 内存碎片问题的（类比操作系统虚拟内存）。
- [ ] **量化**: 解释为什么 Decode 阶段权重量化 (W4A16/W8A16) 带来的收益远大于 Prefill 阶段。
- [ ] **RoPE**: 画出 2D 平面上的旋转示意图，解释相对位置不变性。
- [ ] **RMSNorm**: 手写一段 CUDA 伪代码，展示如何计算 `mean(x²)` (使用 Warp Shuffle)。
- [ ] **SwiGLU**: 解释为什么它比标准 ReLU FFN 效果好，且计算量更大的矛盾是如何在推理中被接受的（通过硬件加速）。
---
## 6. 推荐阅读
1.  **Llama 2/3 Paper**: Open Foundation and Fine-Tuned Chat Models.
2.  **FlashAttention 2**: Faster Attention with Better Parallelism and Work Partitioning.
3.  **vLLM Paper**: Efficient Memory Management for LLM Serving with PagedAttention.
4.  **LeetCUDA**: `/third_party/LeetCUDA/kernels/` (重点阅读 `sgemm`, `rms-norm`, `rope` 目录下的 `README` 和注释).
