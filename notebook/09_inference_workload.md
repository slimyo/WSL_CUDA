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
你的笔记整体框架很好，但第二部分确实缺少具体场景和因果链条。我重写了第二部分，补上你需要的细节，并保持与第1部分的衔接。

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

---

### 2.2 为什么需要 Goodput？（含完整场景推演）

#### 2.2.1 问题起源：两类请求的天然冲突

真实系统中同时存在两类请求：

| 请求类型 | 例子 | prefill 特点 | decode 特点 |
|---------|------|-------------|------------|
| **短 prompt 长生成** | "写一篇关于AI的论文" | prompt 很短（10 tokens），prefill 极快（~1ms） | 要生成 2000 tokens，decode 2000 步 |
| **长 prompt 短生成** | 给一篇 8K 文章，问"总结一下" | prompt 8K tokens，prefill 需要 ~500ms | 只生成 100 tokens 的摘要 |

**它们在物理资源上的冲突**：
- 短 prompt 请求在 decode 时，batch 里突然混入一个长 prompt 的 prefill
- 根据 §1.3，prefill 是 compute-bound，会占满 GPU 算力约 500ms
- 这段时间内，同 batch 的 decode 请求全部被阻塞

#### 2.2.2 具体场景：为什么 Throughput 高但用户体验差？

```
假设一个在线聊天服务，SLO 要求：
  P99 TPOT < 30ms  ← 用户感知的"流畅度"，超过 30ms 会感觉卡顿

服务策略：为了最大化吞吐，batch_size 设为 16

系统中有 15 个短请求正在 decode：
  - 它们都进入 batch，正常情况下 TPOT ≈ 23ms（见 §3.1），满足 SLO
  - 此时 throughput = 16/0.023 ≈ 696 tokens/s，看起来很漂亮

突然，一个长 prompt 请求进来（8K tokens，prefill 需要 500ms）：
  - 系统把它加入当前 batch 做 prefill
  - 这 500ms 内，那 15 个 decode 请求的下一步全部被阻塞
  - 对那 15 个用户来说，这一步的 TPOT 从 23ms 飙升到 500ms
  - P99 TPOT 直接爆炸，违反 SLO
```

**结果**：
```
整个统计周期内：
  Throughput = 700 tokens/s（仍然很高）
  
  但是：
  - 15 个用户中有 1 个经历了 500ms 的卡顿
  - P99 TPOT = 500ms >> 30ms SLO
  - Goodput = 0 ← 因为违反了 SLO，这些吞吐不算"有效"
```

**关键洞察**：Throughput 看的是总量，Goodput 看的是"在规定时间内完成的总量"。高吞吐可以靠堆积请求实现，但如果延迟不可控，用户就会流失。

---

### 2.3 延迟 vs 吞吐的根本张力（含数学论证）

#### 2.3.1 大 batch 为什么能提吞吐？

回顾 §1.4 的 decode 访存分析：

```
单请求 decode 访存 = 权重 12.9 GB + KV cache 2.1 GB = 15 GB
batch=1 算术强度 = 13 GFLOPs / 15 GB ≈ 0.87 FLOPs/Byte
```

当 batch=B 时（假设所有请求的序列长度相同，KV cache 均为 2.1 GB）：

```
总 FLOPs = B × 13 GFLOPs
总访存  = 12.9 GB + B × 2.1 GB  ← 权重被 B 个请求共享，只读一次
算术强度 = 13B / (12.9 + 2.1B) FLOPs/Byte

batch=1:  15 GB / 2 TB/s = 7.5ms,  吞吐 = 1/0.0075 = 133 tokens/s
batch=16: 46.5 GB / 2 TB/s = 23.3ms, 吞吐 = 16/0.0233 = 687 tokens/s  ↑ 5.2×
batch=64: 147.3 GB / 2 TB/s = 73.7ms, 吞吐 = 64/0.0737 = 868 tokens/s ↑ 6.5×
```

**关键结论**：
- 权重 12.9 GB 是固定开销，被 B 均摊
- batch 扩大 16 倍，延迟仅扩大 3 倍（23.3ms / 7.5ms ≈ 3.1×）
- 吞吐提升 5.2 倍——这就是"近乎免费"的来源

#### 2.3.2 大 batch 的真正代价

1. **绝对延迟增大**：单请求 TPOT 从 7.5ms 到 23.3ms，对实时对话来说可能仍在 SLO 内，但如果继续扩大 batch 就会超出
2. **显存压力**：每个请求的 KV cache 需要 2.1 GB，batch=16 时仅 KV cache 就占 33.6 GB，加上权重 12.9 GB，已占 A100 80GB 的 58%
3. **与 prefill 混合时的延迟抖动**（如 §2.2 所述）——这是最致命的

#### 2.3.3 调度策略如何缓解（对应矛盾的不同侧面）

这三种策略在 §5 会详细展开，这里只给出它们在"解决矛盾的哪个侧面"：

```
矛盾核心：
  吞吐 ←→ 延迟 ←→ 混合负载下的稳定性

策略对应：
  Continuous Batching：最大化 batch 利用率 → 提升吞吐，同时减少空闲等待
  Chunked Prefill：    打破 prefill 对算力的长时间独占 → 减少 decode 的延迟抖动
  P/D 分离：           物理隔离两类负载 → 各自独立优化，彻底消除相互干扰
```

**Continuous Batching 的核心思路（一句话版）**：
```
传统方式：等一个 batch 中所有请求都完成，再组下一个 batch
Continuous Batching：每完成一步 decode，立即移出已结束的请求，加入新请求
→ 让 batch 始终保持满员，不浪费"座位"
```

**为什么 Chunked Prefill 能减少抖动**：
```
不用 Chunked Prefill：
  prefill 一次处理 8K tokens，耗时 500ms
  这段时间内所有 decode 被阻塞

用 Chunked Prefill：
  把 8K tokens 的 prefill 切成 8 块，每块 1K tokens ≈ 62ms
  在每块 prefill 之间插入 decode 步骤
  → 每个 decode 步骤最多被阻塞 62ms，而不是 500ms
  → 延迟抖动从 500ms 降到 62ms
```

---

### 2.4 指标之间的权衡（面试速查表）

| 策略方向 | TTFT | TPOT | Throughput | Goodput | 适用场景 |
|---------|:---:|:---:|:---:|:---:|---------|
| 大 batch + 静态调度 | ↑ | ↑ | ↑ | ↓（抖动大） | 离线批处理 |
| 小 batch（batch=1） | ↓ | ↓ | ↓ | ↓（吞吐不够） | 单用户实时 |
| Continuous Batching | ↑ | → | ↑ | ↑ | 在线服务（中等负载） |
| P/D 分离 + 大 batch | → | ↓ | ↑ | ↑ | 大规模在线服务 |
| Chunked Prefill | → | ↓（减少抖动） | → | ↑ | 混合长度 prompt |

---

## 5. 学习检查清单

- [ ] 能说清 prefill 和 decode 的根本区别（计算 vs 访存）
- [ ] 能手算 Llama-7B prefill/decode 的 FLOPs、访存、算术强度
- [ ] 能解释为什么 batch 增大 decode 吞吐近线性涨而延迟亚线性涨（把权重/KV 拆开算）
- [ ] 能说清 TTFT / TPOT / Throughput / Goodput 的定义和区别
- [ ] **能用 §2.2 的场景讲清楚：为什么 throughput 很高，但 goodput 可能是 0**
- [ ] 能讲清 prefill 和 decode 混在一起时的物理冲突（compute-bound vs memory-bound 的互斥）

---

## 6. 自测 / 面试题

1. **为什么 throughput 高不等于 goodput 高？**  
   举一个具体场景：15 个 decode 请求 batch=16，突然混入一个 8K prefill，TPOT 从 23ms 飙升到 500ms，违反 P99 < 30ms 的 SLO，goodput = 0。

2. **prefill 和 decode 放在同一块卡上，会怎么互相干扰？**  
   Prefill 是 compute-bound，占满算力；decode 是 memory-bound，需要持续搬数据。prefill 运行时，decode 的下一步无法执行，导致 TPOT 剧烈抖动。

3. **batch 从 1 加到 8，decode 吞吐涨多少？延迟涨多少？**  
   吞吐：133 → 533 tokens/s（↑ 4×）。延迟：7.5ms → 14.4ms（↑ 1.9×）。用 §2.3.1 的公式计算。

4. **decode 算术强度那么低，为什么不同时跑 100 个请求？**  
   显存放不下。batch=100 时 KV cache = 210 GB，远超 A100 的 80 GB。即使量化后，KV cache 也是主要限制。

5. **一个请求 prefill 8K tokens 需要多长时间（A100）？**  
   §1.3 的公式：计算量 ≈ 61 × (8192/4096)² ≈ 244 TFLOPs，TTFT ≈ 244T / 240T ≈ 1.0 秒。