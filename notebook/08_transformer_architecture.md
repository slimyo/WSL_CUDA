# Transformer Decoder 架构：LLaMA 拆解

> 对象: LLM 推理入门 / 算子岗
> 前置: 03_gpu_memory_hierarchy.md, 06_roofline_and_flops.md, 07_numerical_formats.md
> 目标: 面试能手画 Llama decoder layer，说清每个组件的数学、访存、计算特性
> 参考 LeetCUDA: `rope/`, `rms-norm/`, `gelu/`, `swish/`, `transformer/`, `layer-norm/`

---

## 1. Llama Decoder 结构全景

```
                     ┌──────────────────────┐
  input              │    Embedding Layer    │  token→vec lookup
                     └──────────┬───────────┘
                                │
                ┌───────────────┴───────────────┐
                │         RMSNorm               │  前置 norm
                │     + RoPE (Q/K only)         │  位置编码
                └───────────────┬───────────────┘
                                │
                ┌───────────────┴───────────────┐
                │      Attention (GQA)          │
                │   Q=Wq × x, K=Wk × x, V=Wv   │  proj
                │   Attn(Q,K,V)                 │  核心计算
                │   Out = Wo × attn_out         │  output proj
                └───────────────┬───────────────┘
                                │  residual +
                                │
                ┌───────────────┴───────────────┐
                │         RMSNorm               │  前置 norm (FFN)
                └───────────────┬───────────────┘
                                │
                ┌───────────────┴───────────────┐
                │     FFN (SwiGLU)              │
                │   gate = W_gate × x           │
                │   up   = W_up × x             │
                │   out  = silu(gate) * up      │
                │   down = W_down × out         │
                └───────────────┬───────────────┘
                                │  residual +
                                │
                ┌───────────────┴───────────────┐
                │          ×32 layers           │
                └───────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │        RMSNorm (final)        │
                └───────────────┬───────────────┘
                                │
                ┌───────────────┴───────────────┐
                │        LM Head (linear)       │  hidden → vocab
                └───────────────┬───────────────┘
                                │
                                ▼
                            logits → softmax → token
```

---

## 2. 核心组件逐一拆解

### 2.1 RMSNorm (Root Mean Square Normalization)

**数学：**
```
RMSNorm:  y = gamma × x / sqrt(mean(x²) + epsilon)

LayerNorm: mu = mean(x), sigma = sqrt(mean((x-mu)²) + epsilon)
           y = gamma × (x-mu)/sigma + beta

RMSNorm 省掉了减均值和加 bias → 1 次 reduce vs LN 的 2 次 reduce
```

**CUDA 特点：**
```
fused kernel 架构（每行一个 block）：
  1. load x → block reduce x² → smem 存 rms
  2. sync → 用 rms normalize 当前 x → store y

LeetCUDA: third_party/LeetCUDA/kernels/rms-norm/rms_norm.cu
  rms_norm_f32_kernel / rms_norm_f16_kernel
  与 layer-norm 的区别见 layernorm_rmsnorm_learning.md 的对比
```

### 2.2 RoPE (Rotary Position Embedding)

**数学：**
```
对 Q/K 的每对 (x_{2i}, x_{2i+1}) 做旋转：
  [out_{2i}, out_{2i+1}] = [x_{2i} × cos(position_θ_i) - x_{2i+1} × sin(θ)
                             x_{2i} × sin(position_θ_i) + x_{2i+1} × cos(θ)]

  θ_i = 1 / base ^ (2i/d)  (base=10000.0)

关键性质：
  - 相对位置编码：两个 token 的 QK 点积只依赖于它们的位置差
  - 与 KV cache 兼容：每个 token 的 K 只需在自己的 position 上旋转一次，
    缓存旋转后的 K 即可，后续 decode 不需要重算历史 K（绝对可加性）
```

**CUDA 特点：**
```
elementwise 操作 + sincos 计算（SFU 指令）：
  每个 thread 处理 2 个元素（一对 (x_{2i}+x_{2i+1})）

LeetCUDA: third_party/LeetCUDA/kernels/rope/rope.cu
  rope_f32_kernel: grid(N/2), block(256), 每 thread 1 pair
  rope_f32_v2_kernel: grid(seq_len), block(hidden_dim/2), token 为单位
  rope_f32x4_pack_kernel: float4 打包，grid(N/4/2)
```

### 2.3 SwiGLU（FFN 激活函数）

**数学：**
```
SwiGLU(x) = silu(gate(x)) * up(x)  其中 silu(x) = x × sigmoid(x)

LLaMA FFN:
  gate = x @ W_gate  (project to intermediate_dim)
  up   = x @ W_up    (project to intermediate_dim)
  hidden = silu(gate) * up
  out  = hidden @ W_down  (project back to hidden_dim)

Llama-7B 配置: hidden_dim=4096, intermediate_dim=11008
```

**CUDA 特点：**
```
门控和上投影是 GEMM → elementwise（silu + multiply）→ GEMM
fused kernel 可以把 silu + multiply 融进 epilogue

LeetCUDA: third_party/LeetCUDA/kernels/swish/swish.cu
  swish(x) = x × sigmoid(x) 的 CUDA 实现
  gelu/swish/sigmoid 都是 elementwise，可参考向量化模式
```

### 2.4 残差连接 (Residual / Skip Connection)

```
每层的每个子层（attention/FFN）输出都加上输入：
  x_attn = x + attention(rmsnorm(x))
  x_ffn  = x_attn + ffn(rmsnorm(x_attn))

面试关键：这是 Pre-LN（前置 norm）结构，LLaMA 用此。
Post-LN（原始 Transformer）是先 attn 后 norm → LLaMA 不用。
Pre-LN: 训练更稳定（梯度不易爆炸），不用 warmup。
```

### 2.5 GQA (Grouped-Query Attention)

**参考 M3 笔记（11_attention_variants.md），先了解概念：**
```
MHA: 每个 Q head 配独立 K/V head → KV 大
MQA: 所有 Q head 共享 1 组 K/V → KV 小但质量略降
GQA: 分组共享 → tradeoff（Llama-2/3 用）
```

---

## 3. 各组件计算与访存特性

| 组件 | 计算类型 | FLOPs/layer | 访存模式 | 计算/访存比 | 典型瓶颈 |
|------|---------|:---:|------|:---:|:---:|
| RMSNorm (fused) | elementwise+reduce | O(hidden) | HBM ×3 R/W | 低（streaming） | **memory** |
| RoPE | elementwise+sin/cos | O(hidden) | HBM ×2 R/W | 极低 | **memory** |
| QKV Proj | GEMM (H×3H) | 6×HID² | weight+output | 高 | **compute** |
| Attention | softmax+matmul | O(seq²×d) | KV cache I/O | 分场景 | prefill: compute / decode: memory |
| Output Proj | GEMM (H×H) | 2×HID² | weight+output | 高 | **compute** |
| FFN gate/up | GEMM (H×int) | 4×HID×int | weight+output | 高 | **compute** |
| FFN down | GEMM (int×H) | 2×int×HID | weight+output | 高 | **compute** |
| SwiGLU mul | elementwise | O(intermediate) | R/W ×1 | 极低 | **memory** |
| Residual add | elementwise | O(hidden) | R/W ×2 | 极低 | **memory** (融合后可以忽略) |
| LM Head | GEMM (H×vocab) | 2×H×V | weight | 高 | **compute** |

> **注意：上表的"典型瓶颈"按 prefill（seq_len 大）标注。decode 时所有 GEMM 退化为
> GEMV（M=batch，很小），算术强度 ≈ batch，全部变成 memory-bound——这正是 06/09
> 章 prefill/decode 二分的微观体现。LeetCUDA 的 `sgemv/`、`hgemv/` 就是这类 kernel。**

---

## 4. LeetCUDA 组件源码索引

| Llama 组件 | LeetCUDA 源码 | 文件 |
|------|------|------|
| RMSNorm | `kernels/rms-norm/` | `rms_norm.cu`（f32/f16 版本） |
| LayerNorm | `kernels/layer-norm/` | `layer_norm.cu`（f32/f16x8 等） |
| RoPE | `kernels/rope/` | `rope.cu`（f32/f32x4 版本） |
| GeLU | `kernels/gelu/` | `gelu.cu`（f32/f16/f16x2/f16x8） |
| Swish/SiLU | `kernels/swish/` | `swish.cu` |
| Sigmoid | `kernels/sigmoid/` | `sigmoid.cu` |
| ReLU | `kernels/relu/` | `relu.cu` |
| ELU | `kernels/elu/` | `elu.cu` |
| HardSwish | `kernels/hardswish/` | `hardswish.cu` |
| GEMM (FFN/Proj) | `kernels/sgemm/`,`hgemm/` | mmult stages |
| Embedding | `kernels/embedding/` | `embedding.cu` |
| Attention | `kernels/flash-attn/` | `flash_attn_mma.py` |
| Elementwise fuse | `kernels/elementwise/` | `elementwise.cu` |

---

## 5. 学习检查清单

- [ ] 能手画 Llama decoder layer 结构图，标注每个子层的输入输出形状
- [ ] 能说清 RMSNorm vs LayerNorm 的区别（数学+CUDA 实现差异）
- [ ] 能解释 RoPE 的旋转原理、为什么支持相对位置编码
- [ ] 能推导 SwiGLU 的计算流程（gate/up/down 三条 path）
- [ ] 能说清 Residual + Pre-LN 的结构优势
- [ ] 能给每个组件标计算类型和典型瓶颈
- [ ] 能在 LeetCUDA 中找到每个组件的 CUDA 实现

---

## 6. 自测 / 面试题

1. LLaMA 为什么用 RMSNorm 而不是 LayerNorm？省了多少计算？
2. RoPE 的旋转矩阵为什么能使点积只依赖于相对位置？
3. SwiGLU 相比 ReLU FFN 多了什么计算？为什么效果更好？
4. Pre-LN 相比 Post-LN 有什么训练优势？LLaMA 用哪个？
5. 说出 LLaMA-7B 每层 FFN 的各投影矩阵形状。

---

## 7. 推荐阅读

| 资料 | 来源 |
|------|------|
| LLaMA: Open and Efficient Foundation Language Models | Meta (2023) |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | Zhuiyi Tech |
| RMSNorm: Root Mean Square Layer Normalization | arXiv |
| LeetCUDA rope/rms-norm/swish 源码 | `/third_party/LeetCUDA/kernels/` |
