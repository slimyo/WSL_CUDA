# FlashAttention 家族：v1→v2→v3→v4 与 FlashDecoding

> 对象: 算子岗位（必达 L3）
> 前置: softmax_learning.md, flash_attention_learning.md, tensor_cores_intro.md
> 目标: 面试能在白板推 online softmax，讲清 v1→v2→v3→v4 每代改动
> 参考 LeetCUDA: `flash-attn/`

---

## 1. Online Softmax 推导（面试手写核心）

### 1.1 问题定义

```
标准 softmax:
  softmax(x_i) = exp(x_i) / Σ exp(x_j)

问题：需要知道全局 max 和 sum → 需要两次 pass
  pass 1: m = max(x)
  pass 2: sum = Σ exp(x_j - m)
  pass 3: y_i = exp(x_i - m) / sum

Online softmax: 一次 pass 同时维护 (m, d)
```

### 1.2 递推公式

```
初始化: m_0 = -inf, d_0 = 0

对每个新元素 x_i:
  若 x_i > m_{i-1}:
    新的 max: m_i = x_i
    新 denominator: d_i = d_{i-1} × exp(m_{i-1} - x_i) + 1
    解释: 旧 denominator 需要 rescale（因为 exp(m_{i-1} - x_i) < 1）
  否则:
    m_i = m_{i-1}
    d_i = d_{i-1} + exp(x_i - m_{i-1})
```

### 1.3 分块合并公式（FlashAttention 关键）

```
两个部分结果 (m1, d1) 和 (m2, d2)，设 m_new = max(m1, m2):

d_new = d1 × exp(m1 - m_new) + d2 × exp(m2 - m_new)

O_new = (O1 × d1 × exp(m1 - m_new) + O2 × d2 × exp(m2 - m_new)) / d_new

这使 Q 分块后每次迭代可以合并 KV 块的结果。
```

> **完整推导见 `softmax_learning.md` §2。**

---

## 2. FlashAttention v1 (Tri Dao, 2022)

### 2.1 核心思想

**不写 N×N 注意力矩阵（S, P）到 HBM——全部在 on-chip 完成。**

```
对每个 Q_block (Br × d):
  初始化 m = -inf, d = 0, O_acc = 0     # O_acc 是"未归一化"的累加器
  循环每个 KV_block (Bc × d):
    S = Q_block × K_block^T              (Br × Bc, on-chip)
    m_new = max(m, rowmax(S))
    P = exp(S - m_new)                   (Br × Bc)
    d = d × exp(m - m_new) + rowsum(P)   # 旧 sum rescale + 新块贡献
    O_acc = O_acc × exp(m - m_new) + P × V_block   # 旧输出 rescale + 新块贡献
    m = m_new
  最终: O_block = O_acc / d              # 归一化只在最后做一次

复杂度: FLOPs 仍是 O(N²·d)（flash 不省计算），HBM 访存从 O(N²) 降到 O(N·d)
关键：归一化（除 d）推迟到最后，循环中间只维护未归一化的 O_acc——
     这是手写/Triton 实现最容易写错的地方。
```

### 2.2 Tile 大小选择

```
A100 164 KB shared memory:
  smem_Q: Br × d × 2B (FP16)
  smem_K: Bc × d × 2B
  smem_V: Bc × d × 2B
  smem_S: Br × Bc × 2B

d=128:
  (Br + 2Bc) × 128 × 2 + Br × Bc × 2 ≤ 164KB
  取 Br=128, Bc=128: 128×128×2×4 ≈ 131KB ✓
```

---

## 3. FlashAttention v2 (2023)

### 3.1 相比 v1 的改动

| 改进 | v1 | v2 |
|------|:---:|:---:|
| Q/K/V 分块策略 | Split KV（K/V 循环在外层） | **Split Q（Q 循环在外层）** |
| 非 matmul FLOPs 比例 | 较高 | **减少非 matmul FLOPs** |
| warp 间 work partitioning | 简单分区 | **改进分区，减少 SMEM 通信** |
| causal mask 效率 | 从头到尾都检查 | 跳过超出 causal 范围的块 |
| head_dim 支持 | ≤128 | **≤256（H100 支持）** |

**Split Q 的优势（两个层面）：**
```
层面 1 —— thread block 间（SM 级并行度）：
  v1: 并行维度 = batch × heads，每个 block 串行扫完整个序列
      → 长序列 + 小 batch 时 SM 喂不满
  v2: Q 维也切给不同 thread block（grid = batch × heads × num_Q_blocks）
      → seq 维并行，长序列下 occupancy 大幅提升
      （Q 块之间天然独立——每个 Q 块的输出只依赖它自己看到的 KV）

层面 2 —— block 内 warp 之间：
  v1: 4 个 warp 沿 K 维切 → 每个 warp 算出部分和，
      必须经 SMEM 交换 + 同步才能合出一行 O（"split-K"开销）
  v2: 4 个 warp 沿 Q 维切 → 每个 warp 独占自己的输出行，
      整个内层循环不需要 warp 间通信
```

### 3.2 非 matmul FLOPs 减少

```
v1 在每步内循环中做两件"非 matmul"的事：
  ① 每步都用 exp(m_old - m_new) rescale 旧的 O_acc
  ② 输出更新里掺着除法

v2 的改法：
  - rescale 仍然要做（max 变了不得不修），但归一化除法只在最外层做一次
  - rowmax/rowsum 的统计量维护精简，能合进 matmul 的尽量合进
为什么收益大：Tensor Core matmul 吞吐是 CUDA Core elementwise 的几十倍，
GPU 的非 matmul FLOP "贵" 16× 以上——同样的 FLOPs 数，落在哪个单元上天差地别。
```

---

## 4. FlashAttention v3 (Hopper, 2024)

### 4.1 Hopper 新特性利用

```
v3 充分利用 H100 (SM90) 硬件：
  - TMA (Tensor Memory Accelerator)：异步 global→shared 拷贝，不占用计算单元
  - wgmma：Warpgroup-level MMA，异步矩阵乘
  - Warp Specialization：生产者 warp（负责 TMA 加载）vs 消费者 warp（负责 wgmma）
  - FP8 支持
```

### 4.2 Warp Specialization 架构

```
┌───────────────────────────────────────────────────┐
│                   Block                            │
│  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ Producer Warps   │  │ Consumer Warps          │  │
│  │ ×2-3 warps       │  │ ×2-3 warps              │  │
│  │                  │  │                         │  │
│  │ TMA load:        │  │ wgmma（异步 matmul）    │  │
│  │ K_block → SMEM   │  │ online softmax          │  │
│  │ V_block → SMEM   │  │ O 更新                  │  │
│  │ 异步 + overlap   │  │ 与 producer 的加载 overlap │
│  └─────────────────┘  └─────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

---

## 5. FlashAttention v4 (Blackwell, 2026)

### 5.1 为什么 Blackwell 要重写

```
Blackwell (SM100) 的硬件不对称性：
  - Tensor Core 吞吐暴增（FP8 TC ~数 PFLOPS）
  - Tensor Memory（专用 SRAM）新增，与 SMEM 并存
  - tcgen05 指令：direct-to-Tensor Memory MMA
  - CUDA Core 相对更慢 → elementwise 成为新瓶颈

核心矛盾：v3 的 online softmax 中仍有非 matmul elementwise
这部分在 Blackwell 上变成瓶颈。
```

### 5.2 v4 的三个关键优化

**① 延迟 rescale（H₀ 优化，跳出 90% 的 rescale）**
```
标准 online softmax 每次新 KV block 都 rescale 旧 O。
v4 观察：如果新 block 的 rowmax 相比旧 block 增加不大，跳过 rescale。
仅在 rowmax 显著增大时做一次完全 rescale。
→ 对长序列，约 90% 的 KV block 不需要 rescale → O 更新变纯 matmul。
```

**② FMA 多项式近似 exp（绕开 SFU 瓶颈）**
```
硬件 exp（MUFU.EX2）走 SFU，每 SM 的 SFU 吞吐远低于 FP32 FMA 管线，
attention 每个 score 都要一次 exp → SFU 成为吞吐瓶颈。

v4 的做法：软件实现 exp2——先做 range reduction（拆出整数部分当指数），
小数部分用低阶多项式（FMA 即可算）近似，再拼回浮点数。
→ exp 从"少量 SFU 单元"搬到"大量 FP32 CUDA Core 的 FMA 管线"上，
  吞吐大幅提升（注意：FMA 跑在 CUDA Core 上，不是 Tensor Core）。
```

**③ tcgen05 / Tensor Memory 异步 MMA**
```
tcgen05 是 Blackwell 的新指令：
  - 直接从 Tensor Memory 做 MMA，不走 SMEM→Register
  - 更大的 tile 粒度（32×64 等）
  - 异步 warpgroup-level 操作
```

### 5.3 v4 架构示意

```
┌───────────────────────────────────────────┐
│  Block (Blackwell SM100)                  │
│                                           │
│  Tensor Memory: K/V 分块预加载             │
│  tcgen05: Q 和 K 的 MMA → S               │
│  (在 Tensor Memory 上)                     │
│                                           │
│  Register File: (m, d, O) 维护             │
│  FMA poly exp 做在线 softmax               │
│  (绕过 SFU，避免 pipeline stall)           │
│                                           │
│  H₀ 决策: 判断是否跳过 rescale             │
│  不 rescale → O 更新 = 纯 mma              │
│  rescale    → 做一次完全 rescale            │
└───────────────────────────────────────────┘
```

---

## 6. FlashDecoding / FlashDecoding++

### 6.1 decode 阶段的根本问题

```
decode：query_len = 1, 每个 token 单独处理

prefill attention:
  Q: [seq_len, d] → 可以沿 seq_len 分块 Q → 多个 SM 并行
  Br × Bc 切分 → 容易喂满 SM

decode attention:
  Q: [1, d] → 只有 1 个 query，没有 Q 维可切
  并行度只剩 batch × heads:
    batch=1, heads=32 → 只能开 32 个 thread block
    → A100 有 108 个 SM，2/3 的 SM 直接闲置
    → 且每个 block 要串行扫完整条 KV（seq 越长越慢）
```

### 6.2 Split-KV 解法

```
标准 decode attention（1 query × KV）：
  1 thread block 处理 1 head
  遍历所有 KV → 输出 1 个 O

Split-KV (FlashDecoding):
  把 KV 序列切成多块，每个 block 处理一部分 KV
  每个 block: 输出部分结果 (partial_O, partial_m, partial_d)
  最终：基于 online softmax 合并公式合并所有 partial 结果

优点：
  - KV 越长，可分的块越多 → 更多 SM 并行
  - 长上下文时效率很高

缺点：
  - 最终一步需要全局合并（kernel launch 或者 warp reduce）
  - 短 KV 时不划算（block_size 太小时 overhead 占比大）
```

### 6.3 FlashDecoding++ (2023)

```
FlashDecoding++ (2023) 的核心改进：
  1. 异步 softmax（unified max）：split-KV 各 partial 之间本来要互相
     rescale（依赖彼此的 max）。FD++ 观察到 LLM 的 attention score 分布
     稳定，用一个预先统计好的"统一 max"代替真实 running max
     → 各 partial 完全独立、无同步，溢出时才 fallback 重算
  2. flat GEMM 优化：decode 的小 M GEMM（M<8）用双缓冲等手段
     提高 tensor core 利用率，而不是 padding 到 M=64
  3. 启发式 dataflow：按 batch/seq 形状在不同 kernel 实现间动态选择

关键 insight：decode attention 的并行策略和 prefill 完全相反
  - prefill: split Q → 沿 query 维并行
  - decode: split KV → 沿 key/value 维并行
  - 面试区分度：很多人只懂 prefill 的 flash，不懂 decode 为什么换并行策略

工程现状：vLLM/SGLang/FlashInfer 的 decode kernel 都内置 split-KV
（FlashInfer 里参数化为 split_kv / num_splits，可启发式自动选）。
```

---

## 7. LeetCUDA FlashAttention 源码

| 文件 | 内容 | 对应 |
|------|------|------|
| `kernels/flash-attn/mma/` | MMA PTX flash attention（多个 stage/swizzle 版本） | FA v1/v2 风格 prefill kernel |
| `kernels/flash-attn/cutlass/` | CUTLASS-based attention | FA v2/v3 style |
| `kernels/flash-attn/flash_attn_mma.py` | Python binding + benchmark | Torch 接口 |
| `kernels/openai-triton/fused-attention/` | Triton 版 flash attention | 现场手写题的最佳范本 |
| `kernels/openai-triton/merge-attn-states/` | **partial attention 状态合并 kernel** | **正是 FlashDecoding split-KV 的 merge 步骤（§6.2）** |
| `ffpa-attn/`（仓库顶层） | FFPA：large head_dim (>256) 的 attention 优化 | MLA 时代大 head_dim 的工程参考 |

**学习顺序：**
1. 先读 `softmax_learning.md` 理解 online softmax 数学
2. 再读 `flash_attention_learning.md` 理解 FA v1 流程
3. 再读此文件（v2/v3/v4/FlashDecoding）
4. 最后看 `flash-attn/mma/` 里 MMA 的 CUDA 实现

---

## 8. 学习检查清单

- [ ] 能在白板推导 online softmax 的递推公式（单元素和分块合并）
- [ ] 能说清 v1→v2→v3→v4 每代到底改了什么、为什么
- [ ] 能解释为什么 decode attention 要换 split-KV 策略
- [ ] 能说清 v4 的延迟 rescale 和 FMA poly exp 优化
- [ ] 理解 wgmma (Hopper) 和 tcgen05 (Blackwell) 的作用

---

## 9. 自测 / 面试题

1. 推导：分块计算 softmax 时，新块到来如何修正之前块的 (m,d,O)？
2. v2 相对 v1 具体改了哪几点？为什么"减少非 matmul FLOPs"在 Tensor Core 上收益大？
3. decode 时 batch=1、heads=32、head_dim=128，为什么直接用 prefill flash kernel 会浪费 GPU？怎么改？
4. v4 的 H₀ 优化在什么条件下效果最好？
5. 写一个 Triton FlashAttention forward kernel 的伪代码（causal mask）。

---

## 10. 推荐阅读

| 资料 | 来源 |
|------|------|
| FlashAttention: Fast and Memory-Efficient Exact Attention | Tri Dao et al. (NeurIPS'22) |
| FlashAttention-2: Faster Attention with Better Parallelism | Tri Dao (2023) |
| FlashAttention-3: Fast and Accurate Attention with H100 | Shah et al. |
| FlashAttention-4: Blackwell | 2026 Paper |
| FlashDecoding: Efficient Decoding | 2023 |
| Triton FlashAttention Tutorial | OpenAI Triton official |
| LeetCUDA flash-attn 源码 | `/third_party/LeetCUDA/kernels/flash-attn/` |
| Split-Q vs Split-KV 图解 | FlashAttention-2 Blog (Tri Dao) |
