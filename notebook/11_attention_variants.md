# Attention 变体与 KV Cache 结构：MHA / MQA / GQA / MLA

> 对象: LLM 推理入门 / 算子岗
> 前置: flash_attention_learning.md, 08_transformer_architecture.md
> 目标: 面试能算 KV cache 大小、对比各变体差异、讲 DeepSeek MLA
> 参考 LeetCUDA: `flash-attn/`

---

## 1. KV Cache 大小公式（面试必算）

```
KV cache size = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch × dtype_bytes
              ↑                             ↑                     ↑      ↑
          K 和 V 各 1 份         每层 KV cache 大小        当前序列长度  batch_size
```

**Llama-7B (MHA, 32 KV heads):**
```
每层: 2 × 32 × 128 × seq_len × 2B = 16,384 × seq_len Bytes
seq_len=4096: 64 MB/layer
32 层: 2.1 GB
```

**Llama-2-70B (GQA, 8 KV heads):**
```
每层: 2 × 8 × 128 × seq_len × 2B = 4,096 × seq_len Bytes
seq_len=4096: 16 MB/layer
80 层: 1.3 GB
```

**DeepSeek-V3 (MLA, kv_lora_rank=512 + rope_dim=64):**
```
注意：MLA 不能套上面的 2× 公式——K 和 V 共享同一个 latent，只缓存一份！
每 token 每层缓存 = (d_c + d_rope) × dtype = (512 + 64) × 2B = 1,152 Bytes
seq_len=4096: 4.7 MB/layer
61 层: ≈ 0.3 GB

对比等规模 MHA（128 heads × 128 dim）: 2 × 128 × 128 × 2B = 64 KB/token/layer
→ MLA 压缩 ≈ 57×（DeepSeek-V2 论文称减少 93.3% KV cache）
```

**这个公式决定了：最大 batch 大小 / 可支持上下文长度。**

---

## 2. MHA → MQA → GQA 演进

### 2.1 核心矛盾

```
KV cache 太大 → 显存瓶颈

MHA: 每个 Q head 配独立 K/V head → KV 最大
MQA: 所有 Q head 共享 1 组 K/V → KV 缩 num_heads 倍
GQA: 分组共享 → trade-off（Llama-2/3 用）
```

### 2.2 对比

```
配置: hidden=4096, n_heads=32, head_dim=128, n_kv_heads=?

MHA (n_kv_heads=32):
  KV size = 2 × 32 × 128 × 2B = 16 KB/token/layer
  质量: 最高
  KV 开销: 最大

MQA (n_kv_heads=1):
  KV size = 2 × 1 × 128 × 2B = 0.5 KB/token/layer
  质量: 略降（所有 Q head 共享同一个 K/V）
  KV 开销: 极小

GQA (n_kv_heads=8):
  KV size = 2 × 8 × 128 × 2B = 4 KB/token/layer
  质量: 接近 MHA（每组 4 个 Q head 共享 1 组 K/V）
  KV 开销: 约为 MHA 的 1/4
```

### 2.3 对 Kernel 的影响

**MHA kernel：**
```cuda
// 所有 head 独立，无需 broadcast
for (int h = 0; h < num_heads; h++) {
    Q_h = Q[h * head_dim : (h+1) * head_dim]
    K_h = K[h * head_dim : (h+1) * head_dim]
    O_h = softmax(Q_h × K_h^T) × V_h
}
```

**GQA kernel：**
```cuda
// KV head 需要 broadcast 到多个 Q head
for (int g = 0; g < num_groups; g++) {
    K_g = K[g * head_dim : (g+1) * head_dim]  // 1 组 KV
    for (int h = 0; h < group_size; h++) {
        q_idx = g * group_size + h
        Q_h = Q[q_idx * head_dim : (q_idx+1) * head_dim]
        O_h = softmax(Q_h × K_g^T) × V_g
    }
}
```

GQA kernel 需要对 K/V 做 broadcast 加载：同样一份 K cache 被多个 Q head 复用。对 memory-bound 的 decode 阶段，这比 MHA 的独立加载更友好（但 Cache 层面需要保证数据不 miss）。

---

## 3. MLA (Multi-head Latent Attention) — DeepSeek-V2/V3

### 3.1 核心思想

**把完整的 K/V 压缩到一个低维 latent 向量中缓存，用时上投影还原。**

```
标准 (MHA):
  K = x × W_k (d × d_kv)
  V = x × W_v (d × d_kv)
  → 缓存 K, V: 2 × d_kv 每 token

MLA:
  c = x × W_down (d × d_c)    ← 这个 c 很小
  K = c × W_up_k    (d_c × d_kv)  ← 从 latent 上投影
  V = c × W_up_v    (d_c × d_kv)
  → 只在 cache 中存 c: d_c << 2 × d_kv
```

### 3.2 省了多少

```
DeepSeek-V3 实际配置:
  hidden=7168, n_heads=128, head_dim=128（attention 内部）
  kv_lora_rank (d_c) = 512, qk_rope_head_dim = 64

每 token 每层缓存的就两样东西：
  压缩 latent c_kv:  512 个元素（K 和 V 共用！）
  解耦 RoPE 的 k_rope: 64 个元素（所有 head 共享一份）
  合计 576 元素/token/layer

对比若用 MHA（128 heads）: 2 × 128 × 128 = 32,768 元素/token/layer
  → 32768 / 576 ≈ 57× 压缩
对比 GQA-8 等效配置:      2 × 8 × 128 = 2,048 元素 → MLA 仍省 ~3.6×
且 DeepSeek 论文显示 MLA 的模型质量优于同规模 GQA（不是用质量换显存）
```

### 3.2b 推理时的关键技巧：Matrix Absorption（矩阵吸收）

```
朴素做法：decode 时把缓存的 c 经 W_up_k/W_up_v 还原成完整 K/V 再做 attention
  → 还原出来的 K/V 又是 MHA 的大小，白省了！

矩阵吸收：利用结合律，把上投影矩阵"吸收"进 Q 和 O 的投影里
  score = (q^T W_up_k) · c = ((W_up_k^T q))^T · c
  → 把 W_up_k 合并进 Q 的投影（W_up_v 同理合并进 output proj）
  → attention 直接在 512 维 latent 空间里做，K/V 从不被物化

效果：decode 时 MLA 等价于一个 head_dim=512+64 的 MQA
  → kernel 视角：query head 128 个、KV "head" 只有 1 个 latent
  → 这就是为什么 MLA decode kernel（如 FlashMLA）长得像大 head_dim 的 MQA
注意：prefill 时算力充足，通常反而走"还原成 MHA"的路径（两种等价形态）。
```

### 3.3 RoPE 与低秩压缩的不兼容（面试难点）

```
问题：
  RoPE 需要在每个 token 的 K/Q 上旋转，但 MLA 把 K 压缩成了低秩 c
  c 的低秩空间与 RoPE 的旋转操作不兼容（旋转打破低秩结构）

解法：解耦 RoPE（Decoupled RoPE）
  - K 分为两部分：
    1. 主 K（从 latent c 经 W_up_k 上投影）：无位置编码
    2. 辅助 K（从 x 经一个小型 W_k_rope 投影）：带 RoPE
  - Q 也类似：主 Q + 辅助 Q(带 RoPE)
  - 计算 attention score：
    score = (Q_main × K_main^T + Q_rope × K_rope^T) / sqrt(d)

好处：
  - 主 K/V 保持低秩压缩 → 省 KV cache
  - 位置信息通过小维度辅助 K/Q 嵌入 → 不破坏压缩
  - 辅助投影维度通常很小（~64）→ 额外开销可忽略
```

---

## 4. 各变体总结

| 变体 | KV 头数 | KV cache 大小（等量对比） | 质量 | 使用模型 |
|------|:---:|:---:|:---:|------|
| MHA | = n_heads | 基准 (100%) | 最高（独立 Q/K per head） | 原始 Transformer, GPT, LLaMA-1 |
| MQA | 1 | ~3% | 略降 | PaLM, Falcon |
| GQA | k (8 常见) | ~25% (k=8) | 接近 MHA | **LLaMA-2/3, Qwen, Mistral** |
| MLA | 1 latent | ~6-20% | 接近 MHA | **DeepSeek-V2/V3** |

**主流趋势（2025）:**
- 大模型普遍用 GQA（性价比最高）
- DeepSeek 系用 MLA（极致 KV 压缩 → 支持大 batch/长上下文）
- MQA 较少（质量损失在超大模型上更明显）

---

## 5. LeetCUDA 相关源码

| LeetCUDA 目录 | 对应 | 说明 |
|------|------|------|
| `kernels/flash-attn/mma/` | MMA-based attention | 多 head 并行结构、GQA 改造的起点 |
| `ffpa-attn/`（仓库顶层） | 大 head_dim attention | MLA 吸收后 head_dim=576 的场景，正是 FFPA 优化的 large head_dim 问题 |

> 注：LeetCUDA 的 `kernels/transformer/` 目前是空目录，没有可参考代码。
> MLA kernel 的工业实现请直接读 **DeepSeek FlashMLA**（github.com/deepseek-ai/FlashMLA，
> Hopper 上的 MLA decode kernel）和 vLLM/SGLang 的 MLA backend。

---

## 6. 学习检查清单

- [ ] 能背 KV cache 大小公式，给定模型配置（hidden, n_layers, n_kv_heads, seq_len, batch, dtype）能算显存需求
- [ ] 能说清 MHA→MQA→GQA 的演化动机和 trade-off
- [ ] 能解释 GQA 对 kernel 的影响（KV broadcast）
- [ ] 能讲清 MLA 怎么省 KV cache（低秩压缩 + latent cache）
- [ ] 能解释 decoupled RoPE 解决什么问题、怎么工作

---

## 7. 自测 / 面试题

1. 同样 70B 模型，MHA 改成 GQA(8 组)，KV cache 缩多少倍？对最大 batch 的影响？
2. MLA 的 latent 压缩为什么不能直接套 RoPE？decoupled RoPE 怎么解决？
3. GQA 对 attention kernel 要改什么？为什么 decode 阶段 GQA 可能比 MHA 更友好？
4. DeepSeek-V3 如果用 MHA 替代 MLA，同样显存下 batch 最多缩多少倍？

---

## 8. 推荐阅读

| 资料 | 来源 |
|------|------|
| MQA: Fast Transformer Decoding | Shazeer (2019) |
| GQA: Training Generalized Multi-Query Transformer | Ainslie et al. (2023) |
| DeepSeek-V2 / V3 Technical Report | DeepSeek (2024-2025) |
| Fast Transformer Decoding: KV Cache | NVIDIA Developer Blog |
