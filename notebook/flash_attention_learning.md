# FlashAttention 学习笔记

> TODO.md: 阶段5 — LLM 推理最核心 kernel
> 前置: `softmax_learning.md`, `tensor_cores_intro.md`
> 参考: `third_party/LeetCUDA/kernels/flash-attn/`

## 1. 问题: 标准 Attention 的 O(N²) 内存墙

```
标准流程:
  S = Q × K^T     (N×N, 写入 HBM)
  P = softmax(S)  (N×N, R/W HBM)
  O = P × V       (N×d, 结果)

瓶颈: S 和 P 都是 O(N²), LLM 长序列时 HBM 带宽成为瓶颈。
N=8K 时 S 约 256MB (FP16), 远大于 L2 cache。
```

## 2. FlashAttention 的核心思想

**不写 S 和 P 到 HBM**, 用分块 + online softmax 在 on-chip memory 中完成计算:

```
将 Q 分成 blocks (沿 seqlen):
for each Q_block:
    (m, d, O_block) = (-inf, 0, 0)
    for each KV_block:
        // 在 shared memory / register 中计算
        S_block = Q_block × KV_block^T   (on-chip)
        // online softmax 更新 (m, d)
        m_new = max(m, rowmax(S_block))
        d_new = d * exp(m - m_new) + rowsum(exp(S_block - m_new))
        // 重新缩放旧 O 并累加新结果 (rescale 只乘一次!)
        O_block = O_block * exp(m - m_new)
                + exp(S_block - m_new) × V_block
        m = m_new, d = d_new
    O[Q_block] = O_block / d   (最终 normalize)
```

核心: 用 online softmax 的 (m,d) 维护机制, 分块计算后重新缩放。

## 3. 两种分块策略

### Split KV (FA-1 风格)
沿 seqlen 分块 KV: `for each KV_block`
Q 和 O 留 on-chip, K/V 流式加载。简单但 shared memory 压力大。

### Split Q (FA-2 风格)
仍沿 **seqlen** 切，但换"谁拥有并行维"：每个 thread block 负责一个 Q_block
（grid 多了 seq 维 → 长序列也能喂满 SM），block 内各 warp 分到不同的 Q 行，
独占自己的输出行 → 内层 KV 循环中 warp 间零通信，SMEM 压力小、occupancy 高。
（注意：不是沿 head_dim 切！head_dim 整个留在寄存器/SMEM 里。）
LeetCUDA 实现的是 FA-2 风格。详细对比见 10_flashattention_deep_dive.md §3。

## 4. 学习路线

```
Step 1: FP32 single-head, 无 causal mask
  理解 QKV tiling 循环结构

Step 2: 加入 causal mask
  下三角 mask, 跳过 K/V 中超出当前 Q 位置的 token

Step 3: FP16 + Tensor Cores
  用 WMMA/MMA 加速 S = Q×K^T 和 P×V 的矩阵乘

Step 4: Multi-head
  多个 head 并行, 共享 QKV 加载
```

## 5. Shared Memory 布局

```
smem_Q[Br][d]     // Q tile, Br 行
smem_K[Bc][d]     // K tile, Bc 行
smem_V[Bc][d]     // V tile
smem_S[Br][Bc]    // 临时 S (不写 HBM)
```

Br, Bc 选择满足 shared memory 限制 (A100: 164KB)。典型值: Br=128, Bc=128。

## 6. 推荐阅读

1. FlashAttention 原始论文 (Dao et al. 2022)
2. FlashAttention-2 论文 (Dao 2023) — Split Q 设计
3. `kernels/flash-attn/` MMA 实现
4. `softmax_learning.md` — online softmax 的数学基础

