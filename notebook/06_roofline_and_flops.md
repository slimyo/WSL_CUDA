# Roofline 模型与 FLOPs 计算

> 对象: CUDA / LLM 推理入门
> 前置: 01_gpu_hardware_architecture.md, 03_gpu_memory_hierarchy.md, 04_warp_execution_model.md
> 目标: 面试能徒手算 arithmetic intensity，在 roofline 图上判定 compute-bound vs memory-bound
> 参考 LeetCUDA: `sgemm/`, `hgemm/`, `flash-attn/`

---

## 1. 为什么需要 Roofline

**GPU 优化的第一问不是"怎么加速"，而是"瓶颈在哪"。**

```
编出来的 kernel 要么在等数据（memory-bound），要么在等计算（compute-bound）：

  memory-bound  → 优化方向：减少访存（fusion, quantization, better cache reuse）
  compute-bound → 优化方向：提高算力利用率（pipeline, tensor core, more parallelism）

Roofline 模型用一个二维图+定量公式，把这个问题变成精确计算。
```

---

## 2. Roofline 模型定义

### 2.1 核心概念

| 术语 | 符号 | 定义 | 单位 |
|------|:---:|------|:---:|
| 运算量 | FLOPs | 浮点运算次数 | FLOPs |
| 访存量 | Bytes | 从 HBM 读写的字节数 | Bytes |
| **算术强度** | **I** | FLOPs / Bytes (ratio) | FLOPs/Byte |
| GPU 峰值算力 | P<sub>peak</sub> | GPU 最大 FLOPs/s | TFLOPS |
| GPU 峰值带宽 | B<sub>peak</sub> | GPU 最大 Byte/s | TB/s |
| **Ridge Point** | **I<sub>0</sub>** | P<sub>peak</sub> / B<sub>peak</sub> | FLOPs/Byte |

**A100 (80GB) 关键参数：**

| 配置 | P<sub>peak</sub> | B<sub>peak</sub> | Ridge Point |
|------|:---:|:---:|:---:|
| FP32 CUDA Core | 19.5 TFLOPS | 2.0 TB/s | 9.75 FLOPs/Byte |
| TF32 Tensor Core | 156 TFLOPS | 2.0 TB/s | 78 FLOPs/Byte |
| FP16/BF16 Tensor Core | 312 TFLOPS | 2.0 TB/s | 156 FLOPs/Byte |
| INT8 Tensor Core | 624 TOPS | 2.0 TB/s | 312 OPs/Byte |

> 注意：**A100 没有 FP8 Tensor Core**（FP8 从 Hopper/SM90 开始），A100 上 8-bit 走 INT8。

**H100 (SXM) 关键参数（dense，不带 sparsity）：**

| 配置 | P<sub>peak</sub> | B<sub>peak</sub> | Ridge Point |
|------|:---:|:---:|:---:|
| FP64 Tensor Core | 67 TFLOPS | 3.35 TB/s | 20 FLOPs/Byte |
| FP32 CUDA Core | 67 TFLOPS | 3.35 TB/s | 20 FLOPs/Byte |
| FP16/BF16 Tensor Core | 989 TFLOPS | 3.35 TB/s | 295 FLOPs/Byte |
| FP8 Tensor Core | 1,979 TFLOPS | 3.35 TB/s | 591 FLOPs/Byte |

> **面试坑：NVIDIA 官方宣传页常写 2× 的数字（FP16 1979 / FP8 3958 TFLOPS），那是 2:4 结构化稀疏（sparsity）算力，LLM 推理一般用不上。报数字时务必说明 dense 还是 sparse。**

> **ridge point = P<sub>peak</sub> / B<sub>peak</sub>**
> I > I<sub>0</sub> 时 kernel 是 compute-bound，I < I<sub>0</sub> 时是 memory-bound。

---

## 3. 算术强度实战：以 Llama-7B decode 为例

这是整条路线最重要的计算。每个推理岗面试都有概率让你在白板上推一遍。

### 3.1 模型参数

```
Llama-7B:
  hidden_dim = 4096
  num_layers = 32
  num_heads = 32
  head_dim = 128
  num_kv_heads = 32 (MHA)
  FFN: gate+up+down = 3 × (4096 × 11008)
  vocab_size = 32000
```

### 3.2 Decode 单步 FLOPs

**decoder_layer = attention + FFN**

Attention FLOPs:
```
QKV projection: 3 × 2 × hidden_dim × hidden_dim                ≈ 100 MFLOPs
Q × K^T:        2 × head_dim × seq_len                          ≈ 1 MFLOPs
P × V:          2 × seq_len × head_dim                          ≈ 1 MFLOPs
Output proj:    2 × hidden_dim × hidden_dim                     ≈ 33 MFLOPs

每层 attention: ≈ 135 MFLOPs
```

FFN FLOPs (SwiGLU):
```
gate = x × W_gate:    2 × hidden_dim × intermediate_size        ≈ 90 MFLOPs
up   = x × W_up:      2 × hidden_dim × intermediate_size        ≈ 90 MFLOPs
v    = silu(gate) * up                                          ≈ 0 (elementwise)
down = v × W_down:    2 × intermediate_size × hidden_dim        ≈ 90 MFLOPs

每层 FFN: ≈ 270 MFLOPs
```

**Llama-7B decode 每步 ≈ 32 × (135M + 270M) ≈ 13 GFLOPs**

### 3.3 Decode 单步 HBM 访存

**权重体积（memory-bound 的主因）：**
```
Attention 权重（QKV + O）:  4 × 4096 × 4096 × 2B = 134 MB
FFN 权重（gate+up+down）:  3 × 4096 × 11008 × 2B = 270 MB
每层合计:                     ≈ 404 MB
32 层合计:                    ≈ 12.9 GB
```

**KV cache 读（decode 每步读全部 KV）：**
```
KV cache size per layer:
  2 × num_kv_heads × head_dim × seq_len × 2B
  2 × 32 × 128 × 4096 × 2B = 67 MB/layer
32 层: ≈ 2.1 GB
```

**单步总访存 ≈ 12.9 GB + 2.1 GB = 15 GB**

> 这是 decode 为什么 memory-bound 的根本原因：只算约 13 GFLOPs 但要搬 15 GB 数据。

### 3.4 算术强度计算

```
I = FLOPs / Bytes = 13e9 / 15e9 ≈ 0.87 FLOPs/Byte

A100 ridge point (BF16 Tensor Core): 156 FLOPs/Byte
0.87 << 156 → **deeply memory-bound**

实际 GPU 利用率 ≈ I / I_0 = 0.87/156 × 312 TFLOPS ≈ 1.7 TFLOPS (0.5%)
```

### 3.5 batch 增大对 decode 的影响

**batch = B 时：**

```
FLOPs: B × 13 GFLOPs (每个 token 独立算)
权重搬运: 12.9 GB (不变！权重被 batch 内 B 个请求共享)
KV cache 搬运: B × 2.1 GB (每个请求有自己的 KV)

总访存: 12.9 + B × 2.1 GB
总 FLOPs: B × 13 GFLOPs

I(B) = 13B / (12.9 + 2.1B) FLOPs/Byte

B=1:   I = 0.87
B=16:  I = 208 / 46.5 = 4.47       (↑ 5×)
B=64:  I = 832 / 147.3 = 5.65
B=128: I = 1664 / 281.7 = 5.91
```

**关键 Insights:**
1. batch 增大 → 权重被摊薄 → 算术强度上升（但受 KV cache 增速限制）
2. 这就是 continuous batching 的商业价值：近似"免费"提吞吐
3. 量化（W4A16）可以把权重搬运砍 4× → 算术强度提升约 3×

**进阶（面试加分）：整体 I(B) 有上限，要分组件看。**
```
I(B) = 13B / (12.9 + 2.1B)，当 B→∞ 时渐近线 = 13/2.1 ≈ 6.2 FLOPs/Byte
→ 把权重+KV 加在一起算，无论 batch 多大都"到不了" ridge point(156)？

矛盾的解法：拆开看两类组件——
  线性层（QKV/O proj、FFN）：FLOPs = 2B×权重参数量，访存 ≈ 权重字节（被 B 摊薄）
    → I ≈ B（batch 多大算术强度就约多大）→ B 到几百时这部分变 compute-bound
  attention（读 KV）：每个请求的 KV 独立，batch 摊不薄
    → 永远 memory-bound，且 batch/seq 越大占比越高

结论：大 batch 下 decode 的瓶颈从"搬权重"迁移到"搬 KV cache"。
这正是 KV 量化、GQA/MLA、FlashDecoding 这些技术存在的根本原因。
```

### 3.6 Prefill 算术强度（对比）

```
prefill seq_len=4096：

FLOPs per layer:
  attention: QKV/O proj (0.55T) + Q×K^T (0.14T) + P×V (0.14T) ≈ 0.82 TFLOPs
  FFN: 3 × 2 × 4096 × 4096 × 11008 ≈ 1.1 TFLOPs

每层 ≈ 1.9 TFLOPs，32 层 ≈ 62 TFLOPs（详细分项见 09 §1.3）

访存 ≈ 模型权重 (12.9 GB) + KV write (2.1 GB) ≈ 15 GB
I(prefill) ≈ 62e12/15e9 ≈ 4100 FLOPs/Byte → **compute-bound**
```

**这解释了 prefill/decode 的根本二分：**
| | Prefill | Decode |
|---|---|---|
| 瓶颈 | compute-bound (吃算力) | memory-bound (吃带宽) |
| 优化方向 | tensor core, pipeline | weight sharing, batching |
| 关键指标 | TTFT | TPOT/ITL |

---

## 4. GEMM 算术强度分析

### 4.1 M×N×K 矩阵乘

```
C[M][N] = A[M][K] @ B[K][N]

FLOPs = 2 × M × N × K  (乘+加)

访存 = (M×K + K×N + M×N) × dtype_bytes

I = (2 × M × N × K) / ((M×K + K×N + M×N) × dtype_bytes)
```

**典型场景（FP16，dtype_bytes=2）：**

| 场景 | M×N×K | FLOPs | 访存 | I (FLOPs/Byte) | 判定 |
|------|-------|------|------|:---:|:---:|
| Decode Q×K | 1×4096×128 | 1 M | 1.1 MB | ~0.9 | memory-bound |
| Prefill attention | 4096×4096×128 | 4.3 G | 8.6 MB | ~500 | compute-bound |
| FFN gate proj | 4096×4096×11008 | 370 G | 164 MB | ~2250 | compute-bound |

### 4.2 LeetCUDA 中的 GEMM 优化层次

| LeetCUDA 文件 | 优化步骤 | 算术强度范围 | 适用场景 |
|------|---------|:---:|------|
| `sgemm.cu` | Naive → Tiled → Vectorized | I < 30 | small mat/memory-bound |
| `sgemm_wmma_tf32_stage.cu` | WMMA + multi-stage pipeline | I > 100 | large mat/compute-bound |
| `hgemm/` | WMMA → MMA → MultiStage Pipeline | I > 100 | HGEMM peak perf |
| `ws-hgemm/` | Warp Specialization | I > 200 | Hopper HGEMM |

**优化原则：先量算术强度，再选优化策略。**

---

## 5. LeetCUDA 源码映射索引

| LeetCUDA 目录 | 对应学习内容 | Roofline 联系 |
|------|------|------|
| `kernels/sgemm/` | SGEMM naive→tiled→WMMA | `sgemm.cu` 文件可分析算术强度梯度 |
| `kernels/hgemm/` | HGEMM multi-stage pipeline | `mmult_stage.cu` pipeline overlap |
| `kernels/flash-attn/` | FlashAttention tiling | `flash_attn_mma.py` memory 分析 |
| `kernels/ws-hgemm/` | Warp Specialization | `naive_ws_hgemm_sm8x.cu` |
| `kernels/elementwise/` | Elementwise (带宽受限) | `elementwise.cu` 纯 bandwidth-bound |
| `kernels/nvidia-nsight/` | Nsight profiling 指标 | nsight 源码与 roofline metrics |

---

## 6. 学习检查清单

- [ ] 能说清 arithmetic intensity 定义，能手算 ridge point
- [ ] 能推导 Llama-7B prefill/decode 的 FLOPs + 访存 + 算术强度
- [ ] 能解释 batch 增大为什么提升 decode 算术强度
- [ ] 能在 roofline 图上定位一个 kernel 的瓶颈
- [ ] 能说清 kernel 优化方向由瓶颈决定（非"玄学优化"）
- [ ] 能理解 Tensor Core 的 ridge point 远高于 CUDA Core 意味着什么

---

## 7. 自测 / 面试题

1. A100 BF16 算力 312 TFLOPS、带宽 2 TB/s，ridge point 是多少？算术强度=2 的 kernel 瓶颈在哪？
2. 为什么 decode 单步算术强度极低？用 (FLOPs)/(权重字节+KV字节) 推一遍。
3. 用 roofline 论证：batch 从 1 加到 16，decode throughput 为何近线性涨而单请求 TPOT 几乎不变？
4. Llama-7B decode 一个 token，在 A100 上理论最小延迟大概多少？（提示：memory wall）
5. 某 kernel 在 roofline 上落在 memory-bound 区域，你给三类不同方向的优化方案？

---

## 8. 推荐阅读

| 资料 | 来源 |
|------|------|
| GPU Roofline Model 介绍 | NVIDIA Developer Blog |
| Roofline: An Insightful Visual Performance Model | Williams et al. (2009) |
| NVIDIA A100/H100 Whitepaper | nvidia.com |
| LeetCUDA sgemm / hgemm 源码 | `/third_party/LeetCUDA/kernels/sgemm/` |
| Nsight Compute Roofline 分析 | Nsight Compute 内置工具 |
