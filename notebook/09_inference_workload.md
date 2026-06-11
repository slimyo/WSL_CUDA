# LLM 推理工作负载特性与指标体系

> 对象: LLM 推理入门 / 算子岗
> 前置: 06_roofline_and_flops.md, 08_transformer_architecture.md
> 目标: 面试能用 roofline 论证 prefill/decode 二分、说清 TTFT/TPOT/throughput/goodput
> 参考 LeetCUDA: `flash-attn/`, `sgemm/`

---

## 1. Prefill vs Decode 二分

**这是全路线最重要的一句话。**

### 1.1 定义

```
Prefill（预填充）:
  - 一次处理整个输入 prompt（如 4096 tokens）
  - 大矩阵 GEMM（seq_len × hidden）
  - 算术强度高 → compute-bound
  - 关键指标：TTFT（Time To First Token）

Decode（解码）:
  - 逐 token 自回归生成（一次 1 个 token）
  - 每步把整模型权重 + 全部 KV cache 从 HBM 搬一遍
  - 算术强度极低 → memory-bound
  - 关键指标：TPOT/ITL（Time Per Output Token / Inter-Token Latency）
```

### 1.2 核心差异

| | Prefill | Decode |
|---|---|---|
| 每次处理 | 整个 prompt | 1 个 token |
| 计算类型 | 大矩阵 GEMM | 小矩阵 + 大量访存 |
| 瓶颈 | compute-bound (算力) | memory-bound (带宽) |
| 算术强度 | ~4100 FLOPs/Byte | ~0.87 FLOPs/Byte |
| 优化方向 | Tensor Core, pipeline | 量化, batching, weight sharing |
| GPU 利用率 | 70-90% | < 1% (batch=1) |
| 关键指标 | TTFT | TPOT / ITL |

### 1.3 Prefill 详细分析

```
Llama-7B prefill seq_len=4096:

Attention per layer:
  QKV Proj:     3 × 2 × seq_len × H × H           ≈ 412 GFLOPs
  Q × K^T:      2 × seq_len × seq_len × head_dim × n_heads = 137 GFLOPs
  P × V:        137 GFLOPs
  Output Proj:  2 × seq_len × H × H                 = 137 GFLOPs
  Total Attn:   ≈ 823 GFLOPs

FFN per layer:
  gate/up/down: 3 × 2 × 4096 × 4096 × 11008 = 1.1 TFLOPs

每层 ≈ 1.9 TFLOPs, 32 层 ≈ 61 TFLOPs

访存 ≈ 12.9 GB (权重) + 32 × 67 MB (KV write) ≈ 15 GB

算术强度 ≈ 61e12/15e9 ≈ 4100 FLOPs/Byte >> A100 ridge point(156)
→ compute-bound, A100 预期 240 TFLOPS+, TTFT ≈ 61T/240T ≈ 254ms
```

### 1.4 Decode 详细分析

```
Llama-7B decode batch=1:

每层:
  QKV proj:                   3 × 2 × 4096 × 4096            ≈ 100 MFLOPs
  Q×K^T (每 head, 32 head):   2 × 4096(seq) × 128 × 32        ≈ 1 MFLOPs(合计)
  P×V:                        同上                            ≈ 1 MFLOPs(合计)
  Output proj:                2 × 4096 × 4096                 ≈ 33 MFLOPs
  FFN:                        3 × 2 × 4096 × 11008            ≈ 270 MFLOPs

每层 ≈ 405 MFLOPs, 32 层 ≈ 13 GFLOPs

访存 ≈ 12.9 GB (权重) + 2.1 GB (KV cache) ≈ 15 GB

算术强度 ≈ 13e9 / 15e9 ≈ 0.87 FLOPs/Byte << A100 ridge point(156)
→ deeply memory-bound, 实际可达 ≈ I × 带宽 ≈ 1.7 TFLOPS（峰值的 0.5%）
→ 不量化的 A100 decode 约 15GB / 2TB/s ≈ 7.5ms/token (忽略 overhead)
```

---

## 2. 指标体系

### 2.1 核心指标

| 指标 | 全称 | 含义 | 典型值 | 受谁影响 |
|------|------|------|:---:|------|
| **TTFT** | Time To First Token | 从请求发出到收到第 1 个 token | 0.3-2s | prefill 速度 |
| **TPOT** | Time Per Output Token | 每个输出 token 的生成时间 | 6-50ms (batch=1) | decode 单步 |
| **ITL** | Inter-Token Latency | token 间延迟 = TPOT | 同 TPOT | 同 |
| **Throughput** | — | 每秒生成的 token 数 | tokens/s | 并发/量化 |
| **Goodput** | — | 满足 SLO 约束下的有效吞吐 | tokens/s | 服务质量 |
| QPS | Queries Per Second | 每秒能处理的请求数 | — | 系统吞吐 |
| SLO | Service Level Objective | 服务质量目标（如 P99 TPOT < 30ms） | — | 服务标准 |

### 2.2 Goodput 的重要性（2024+ 高频考点）

```
Throughput != Goodput

背景：
  两种请求：短 prompt（"Hello" → 512 tokens）和长 prompt（8K prompt → 128 tokens）
  全部堆高吞吐 → 长 prefill 进来时，所有 decode 用户的 TPOT 剧烈抖动
  Throughput 可能很高，但 P99 TPOT 超出 SLO

Goodput = 在满足 SLO 约束下的 throughput
  目的：不让高吞吐掩盖服务质量的下降
  常用 SLO：P50 TTFT < 500ms, P99 TPOT < 30ms
```

### 2.3 延迟 vs 吞吐的根本张力

```
大 batch 提吞吐：
  batch=16 decode: throughput ↑ ~5×, latency 只 ↑ ~3×（权重搬运被摊薄，见 §3.1）
  但 batch 内若混入 prefill: 显存吃紧、compute 被抢，decode 延迟抖动

决定性的矛盾：
  大 batch → 高吞吐 → 单请求延迟增加
  小 batch → 低延迟 → 吞吐不足

后面的调度策略（M5）都是在解这个矛盾：
  Continuous Batching: 动态加减请求，最大化 batch 的同时尽量维持延迟
  Chunked Prefill: 把长 prefill 切块，与 decode 捎带，减少 TPOT 抖动
  P/D 分离: prefill 和 decode 在不同卡上做，各自 batch
```

---

## 3. Batching 对 decode 的"免费"提吞吐效应

### 3.1 数学推导

```
decode batch = B 时：

FLOPs: B × 13 GFLOPs
权重访存: 12.9 GB (B 个请求共享！)
KV cache 访存: B × 2.1 GB
总访存: 12.9 + B × 2.1 GB

time_min = (12.9 + 2.1B) GB / 2 TB/s  （假设 memory-bound）
tokens/s = B / time_min

batch=1:  t = 7.5ms,  吞吐 = 133 tokens/s
batch=16: t = 23.3ms, 吞吐 = 688 tokens/s  (↑ 5.2×)
batch=64: t = 73.6ms, 吞吐 = 870 tokens/s  (↑ 6.5×)
```

**核心洞察：batch 增大 ≠ 延迟按比例增大**
- 权重搬运 12.9 GB 是固定开销 → 被 B 个请求摊薄
- TPOT 从 batch=1 的 7.5ms 涨到 batch=16 的 23.3ms (只涨 3×)
- 吞吐涨了 5.2× → "近乎免费"

### 3.2 什么时候 batch 不再免费

```
整体算术强度 I(B) = 13B / (12.9 + 2.1B)，B→∞ 时渐近线 = 13/2.1 ≈ 6.2
→ 把权重和 KV 加在一起看，整体永远"够不到" ridge point。

正确的分析要拆组件（这是面试区分度所在）：
  1. 线性层（QKV/O proj + FFN）：权重被 B 摊薄，I ≈ B
     → B 涨到 ~ridge point（A100 BF16 约 150-300）时，这部分转 compute-bound
  2. attention（读 KV cache）：每个请求 KV 独立，B 摊不薄
     → 永远 memory-bound，B 和 seq_len 越大，KV 访存占比越高

实际限制顺序：
  显存容量（KV cache 塞满）通常先到 → 然后是 KV 带宽 → 最后才是算力
结论：大 batch decode 的瓶颈从"搬权重"迁移到"装下并搬 KV cache"，
这是 PagedAttention（装下）、GQA/MLA/KV量化（搬得少）的根本动机。
```

---

## 4. 指标之间的权衡

```
吞吐优先 → 大 batch → 低成本
延迟优先 → 小 batch → 满足实时性
goodput → 在两者之间找到平衡点

常见 trade-off:
| 策略 | TTFT | TPOT | Throughput | Goodput |
|------|:---:|:---:|:---:|:---:|
| greedy (大 batch) | ↑ 高 | ↑ 高 | ↑ 高 | ↓ 可能低 |
| conservative (小 batch) | ↓ 低 | ↓ 低 | ↓ 低 | ↑ 可能高 |
| 动态调整 | 可控 | 可控 | 中 | ↑ 最高 |
```

---

## 5. 学习检查清单

- [ ] 能说清 prefill 和 decode 的根本区别（计算 vs 访存）
- [ ] 能手算 Llama-7B prefill/decode 的 FLOPs、访存、算术强度
- [ ] 能解释为什么 batch 增大 decode 吞吐近线性涨而延迟亚线性涨
- [ ] 能说清 TTFT / TPOT / Throughput / Goodput 的定义和区别点
- [ ] 能举一个 throughput 高但 goodput 低的场景
- [ ] 能讲清 prefill 和 decode 混在一起的相互干扰

---

## 6. 自测 / 面试题

1. 为什么 throughput 高不等于 goodput 高？举一个吞吐高但违反 SLO 的场景。
2. prefill 和 decode 放在同一块卡上，会怎么互相干扰？
3. batch 从 1 加到 8，decode 吞吐涨多少？延迟涨多少？（用数字说话）
4. decode 算术强度那么低，为什么不同时跑 100 个请求？（提示：显存）
5. 一个请求 prefill 8K tokens 需要多长时间（A100）？

---

## 7. 推荐阅读

| 资料 | 来源 |
|------|------|
| LLM Inference Performance: The Definitive Guide | Modal Blog |
| Efficient LLM Inference (prefill/decode) | NVIDIA Developer |
| vLLM: Easy, Fast, and Cheap LLM Serving | vLLM Blog |
| Continuous Batching | Orca (OSDI'22) |
| Roofline Model for LLM Inference | — |
