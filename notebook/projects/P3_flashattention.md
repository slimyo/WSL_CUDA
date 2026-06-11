# P3 · FlashAttention Forward：CUDA 手写 + Triton 双实现（2-2.5 周）

> 前置阅读: softmax_learning.md, flash_attention_learning.md, 10_flashattention_deep_dive.md §1-3
> 前置项目: P1（验收工具）、P2（WMMA tile 直接复用）
> 产出: ①naive attention 对照组 ②CUDA flash forward（fp32→WMMA fp16）③Triton 版（causal）
>       ④"HBM 流量实测"报告——用 ncu 证明 flash 的 IO 优势
> 这是算子岗现场手写概率最高的题，目标是【闭着眼能写出 Triton 版】。

---

## 1. 项目定义

固定问题：单 batch 多 head 的 scaled-dot-product attention forward
`O = softmax(QKᵀ/√d)·V`，shape `[n_heads=8, seq_len=512~4096, head_dim=64]`，fp16。
（6GB 卡上 seq=4096 的 naive 版 S 矩阵 8×4096²×2B=256MB，刚好能跑、刚好能"痛"——
完美的对照组。）

## 2. 工程框架

```
src/projects/p3_flash_attn/
├── attn_v0_naive.cu        # 三个 kernel: GEMM→softmax→GEMM, S/P 落 HBM
├── attn_v1_flash_f32.cu    # 单 kernel flash, CUDA core, fp32, 不求快只求对
├── attn_v2_flash_wmma.cu   # S=QKᵀ 和 PV 换成 WMMA (复用 P2)
├── attn_v3_causal.cu       # +causal mask + 跳过整块
├── triton/
│   ├── flash_fwd.py        # Triton 版（最终要会默写的就是它）
│   └── bench.py            # vs torch.nn.functional.sdpa / xformers
└── verify.py               # 用 torch 算参考输出, 误差 < 1e-2
```

## 3. 分步任务

### Step 1（2 天）v0 naive 对照组 + 建立"痛感"

三 kernel 实现，然后用 P1 的 SOP 量化它的病情：
- nsys：三个 kernel 的时间占比（softmax 占比远超其 FLOPs 占比 → 为什么？）
- ncu：实测总 HBM 读写字节，手算验证 `≈ 2·N²·heads·2B 的 S/P 流量主导`
**这份数据就是你报告第一节的"动机"——flash 省的就是这部分。**

### Step 2（4 天）v1 flash fp32：先把数学写对

按 10 章 §2.1 修正后的伪代码实现（**未归一化累加器 + 最后除 d**）：
```
每个 block 负责一个 (head, Q_block[Br=64])：
  smem: Q_tile[64][64], K_tile[Bc=64][64], V_tile[64][64]
  寄存器: O_acc[每线程若干], m[row], d[row]
  for kv_block in range(N/Bc):
      S = Q_tile @ K_tileᵀ / sqrt(dk)          # smem 内
      m_new = max(m, rowmax(S))                 # warp shuffle reduce
      P = exp(S - m_new)
      d = d*exp(m-m_new) + rowsum(P)
      O_acc = O_acc*exp(m-m_new) + P @ V_tile
      m = m_new
  O = O_acc / d  → 写回 HBM 一次
```
**验收（按顺序，全过才算过）：**
1. 与 torch 参考误差 <1e-3（fp32）
2. ncu 实测 HBM 总流量降为 naive 的 ~1/10 以下（理论 O(N²)→O(N·d)，
   贴出两个数字——这是整个项目最有说服力的一张图）
3. 经典错误自查：rescale 乘两遍？最后忘了除 d？m 初始化 0 而不是 -inf？

### Step 3（3 天）v2 接上 WMMA + v3 causal

- S=QKᵀ、P@V 换成 P2 的 16×16×16 WMMA tile（P 要先从 fp32 转 half——
  体会"非 matmul 操作（exp/rescale）和 matmul 交错"的代价，这正是 FA 系列
  逐代优化的对象，10 章 §3.2/§5 的体感版）
- causal：block 级判断——`kv_block_end <= q_block_start` 的块全跳过
  （耗时应接近减半），对角块内做 element mask
**验收：seq=2048 时 v2 比 v1 提速 >2×；causal 比非 causal 快 ~1.8×。**

### Step 4（4 天）Triton 版——面试默写件

参考官方 tutorial 06-fused-attention，但**必须自己从空文件写**（抄一遍≠会写）：
```python
@triton.jit
def flash_fwd(Q, K, V, O, L, stride..., N, D: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m  = tl.program_id(0)          # Q block
    pid_bh = tl.program_id(1)          # batch*head
    q = tl.load(...)                                    # [BLOCK_M, D]
    m_i = tl.full([BLOCK_M], -float('inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for start_n in range(0, hi, BLOCK_N):               # hi 由 CAUSAL 决定
        k = tl.load(...); v = tl.load(...)
        s = tl.dot(q, tl.trans(k)) * scale
        if CAUSAL: s = tl.where(mask, s, -float('inf'))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    tl.store(O..., (acc / l_i[:, None]).to(tl.float16))
```
加 `@triton.autotune`（BLOCK_M/N ∈ {32,64,128} × num_warps ∈ {2,4,8}），
观察 Turing 上的最优配置和教程（A100 调的）不同——**autotune 存在的理由**。
**验收：①正确 ②性能在 torch SDPA(flash 后端) 的 70% 以上 ③能脱稿重写（限时 30 分钟）。**

### Step 5（1 天）报告

四版本对比表 + HBM 流量图 + "Triton 版和 CUDA 版我各花了几小时、性能差多少"
——这就是 14 章工具链选型题的亲身答案。

## 4. 关键能力

1. **online softmax 三件套**（递推、分块合并、未归一化累加）写对且讲清
2. **IO 复杂度分析**：手算 naive O(N²) vs flash O(N·d) 流量并用 ncu 实证
3. **mixed 精度纪律**：统计量(m,l)和累加器永远 fp32，进 tensor core 才转 fp16
4. **Triton 思维**：tile 级编程 + autotune，和 CUDA 版的开发效率/性能对比有一手数据

## 5. 常见坑

- smem 超限：Br=Bc=64, d=64 时 Q+K+V tile = 3×64×64×2B=24KB ✓；想加 S tile 注意预算
- `exp` 用 `__expf`/`tl.exp` 快速版，精度够用（FA4 还嫌它慢，见 10 章 §5.2）
- Triton 的 `tl.dot` 要求块维 ≥16；head_dim 不是 2 幂时用 mask 补齐
- 对比 torch 时确保它走的后端一致（`torch.backends.cuda.sdp_kernel` 强制 flash）

## 6. 扩展方向（按性价比排序）

- **GQA 支持**：KV head 数 < Q head 数，kernel 里做 head 映射（11 章 §2.3 实现版）
- backward（难度大增，理解重计算即可，面试少考手写）
- 变长 batch（cu_seqlens 风格,varlen）——衔接 serving 真实输入
- 对照读 LeetCUDA `kernels/flash-attn/mma/basic/` 的 MMA PTX 版，写"它比我的
  WMMA 版多做了什么"笔记（不要求复现）

## 7. 参考

- Triton 官方 tutorial: 06-fused-attention（结构参考，别照抄）
- LeetCUDA: `kernels/flash-attn/mma/basic/`、`kernels/openai-triton/fused-attention/`
- FlashAttention v1/v2 论文（算法 1 的伪代码就是 Step 2 的依据）
- notebook: 10 章（v1/v2 机制）、softmax_learning.md §2
