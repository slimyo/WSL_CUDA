# 推理调度粒度：从 Continuous Batching 到 P/D 分离

> 对象: LLM 推理工程
> 前置: 09_inference_workload.md, 12_kv_cache_management.md
> 目标: 面试能画 P/D 分离架构图、讲清 chunked prefill 和 speculative decoding

---

## 1. 调度的核心问题

**目标：在有限的 GPU 上，最大化满足 SLO 约束下的有效吞吐。**

```
三个核心矛盾：
  1. Prefill vs Decode 混跑互相干扰（两个阶段算力/带宽需求不同）
  2. 大 batch 提吞吐 vs 单请求延迟升高
  3. 不同请求的 prompt 长度、target 长度差异大 → 调度不公平
```

---

## 2. 调度演进历史

### 2.1 Request-Level Scheduling（原始方式）

```
请求 A 来了 → 处理 A 的 prefill → 逐步 decode A → A 完成
→ 请求 B 来了 → 处理 B 的 prefill → ...

问题：
  - GPU 利用率低（每步只处理 1 个请求）
  - 无法利用 batching 的"免费提吞吐"效应
```

### 2.2 Iteration-Level Scheduling（Orca, OSDI'22）

```
每个 decode step 动态决定当前 batch：
  已完成请求 → 踢出
  新请求 → 加入
  每个请求各自在不同进度

Continuous Batching（连续批处理）：
  ┌─────────────────────────────────────┐
  │ Step 1: 请求 A(dec), B(dec), C(dec)   │
  │ Step 2: A 完成, C 完成, D(pre) 加入   │
  │ Step 3: B(dec), D(pre→dec), E(dec)  │
  │ Step 4: ...                           │
  └─────────────────────────────────────┘

优点：batch 持续利用，无等待窗口
缺点：长 prefill 进来时，decode 用户 TPOT 剧烈抖动
```

### 2.3 Chunked Prefill (Sarathi-Serve)

**解决长 prefill 造成的 TPOT 抖动问题。**

```
朴素 continuous batching：长 prefill 一来，
  prefill 占满算力 → decode 被阻塞 → TPOT 从 8ms 跳到 200ms

Chunked Prefill：
  把长 prefill 切成小块（如每块 512 tok）
  每个 decode step 捎带（piggyback）1 块 prefill
  ┌─────────────────────────────────────────┐
  │ Step 1: A(dec), B(dec), C(pre-chunk1)   │  ← prefill 切小块
  │ Step 2: A(dec), B(dec), C(pre-chunk2)   │  ← C 的 prefill 继续，A/B 的 decode 不被阻塞
  │ Step 3: A(dec), B(dec), C(pre-chunk3)   │  ← prefill 完成
  │ Step 4: A(dec), B(dec), C(dec)          │  ← C 开始 decode
  └─────────────────────────────────────────┘

效果：prefill 被"摊平"，TPOT 抖动显著降低
代价：prefill 变慢（TTFT 轻微上升），因为失去了 GEMM 的 compute-bound 优势
```

---

## 3. Prefill / Decode 分离开（当前最大热点）

### 3.1 动机

直接来自 M1 的 prefill/decode 二分：

```
Prefill: compute-bound + care TTFT
Decode:  memory-bound + care TPOT

混在一起的问题：
  - Prefill 抢走 Tensor Core → decode 更慢
  - Decode 的断续执行 → prefill 被碎片化
  - 无法各自独立扩缩（prefill 需要大算力，decode 需要大带宽）
  - 不能用不同硬件（如 prefill 可以配 H100、decode 配 L40S 等便宜卡）
```

### 3.2 架构

```
          ┌─────────────┐
          │  Load       │
          │  Balancer   │  ← P/D-aware routing
          └──────┬──────┘
                 │
       ┌─────────┴─────────┐
       │                    │
  ┌────▼────┐         ┌────▼────┐
  │ Prefill  │         │ Prefill  │       P 池
  │ Pool 1   │         │ Pool N   │
  └────┬────┘         └────┬────┘
       │                    │
       └──────┬─────────────┘
              │ KV transfer (NVLink/RDMA/NCCL)
       ┌──────┴─────────────┐
       │                    │
  ┌────▼────┐         ┌────▼────┐
  │ Decode  │         │ Decode  │       D 池
  │ Pool 1  │         │ Pool N  │
  └─────────┘         └─────────┘

P/D 分离 = 拆到不同 GPU 池，各自独立扩缩
KV transfer = prefill 完成时 KV cache 路由到 decode worker
```

### 3.3 代表系统

| 系统 | 特点 |
|------|------|
| DistServe | 最早的 P/D 分离系统之一 |
| Splitwise | 分离后各池用不同硬件 |
| Mooncake (KVCache-Centric) | 以 KV cache 为中心的分离架构，全局 KV pool |
| vLLM | 2024 版正式支持 P/D 分离 |
| NVIDIA Dynamo | P/D 分离编排层（路由 + 自动扩缩 + KV 传输协调） |

### 3.4 关键工程难点：KV Transfer

```
prefill 完成时 → 把全部 KV cache 传输到 decode worker

问题：
  - 显存带宽：Llama-7B seq=4K, 32 层 → 2.1 GB KV cache
  - 传输延迟：PCIe 4.0 x16 ≈ 32 GB/s → 2.1 GB / 32 = 66ms
  - 这 66ms 直接加到 TTFT 的 P99 上

解法：
  - NVLink (900 GB/s H100): 2.1 GB / 900 = 2.3ms ✓
  - RDMA (200 GB/s InfiniBand): 2.1 GB / 200 = 10.5ms
  - Pipelined KV transfer:
    prefill 每完成几层就发送 → decode 不等全部到达就开始
```

### 3.5 什么时候不值得 P/D 分离

```
P/D 分离新增成本：
  - 额外的硬件（两组 GPU 池）
  - KV 传输带宽/延迟
  - 编排复杂度

不值得的场景：
  - 负载很低（GPU 空闲时不需要分离）
  - prefill 很短（prompt 短的场景增值不大）
  - 单卡够用
```

---

## 4. Speculative Decoding（投机解码）

### 4.1 动机

```
decode 是 memory-bound → GPU 利用率低
能不能一次生成多个 token，让 decode 步骤变"少"？

本质：把 memory-bound 的 decode 压缩成更少的步骤
```

### 4.2 原理

```
小 draft model（如 100M 参数）快速猜测下 k 个 token：
  Draft: ["the", "capital", "of", "France"]

大目标模型（如 7B）并行验证这 k 个 token：
  → 并行 forward（batch = k, 不是 k 步串行！）
  → 检查哪些 token 通过验证
  → 接受通过的部分，丢弃未通过的部分

好处：
  - Draft model 步骤快（参数少，memory-bound 稍好）
  - 验证步骤是 batched decode（算术强度更高）
  - 整体 token/s 提升
```

### 4.3 为什么不掉精度

```
大模型的验证步骤保证：
  - 如果 draft 猜对了全部 k 个 → 全部接受（和原始输出一样）
  - 如果某步猜错了 → 回退到第一个错误的 token 之前
  - 验证使用原始大模型的 logits → 分布不变
```

### 4.4 常见变体

| 变体 | 核心思路 | 优点 | 缺点 |
|------|------|------|------|
| 标准投机解码 | 独立小模型做 draft | 实现简单、稳定 | 需额外加载小模型 |
| Medusa | 在 LLM head 加多头并行预测 | 无额外模型 | 需要微调 head |
| EAGLE | 在特征层做自回归 draft | 精度高 | 实现复杂 |
| Lookahead Decoding | 基于 n-gram 重复模式预测 | 无需额外模型 | 仅 style repetition 场景有效 |
| Self-Speculative | 同模型跳过中间层做 draft | 无需额外权重 | 速度提升有限 |

### 4.5 什么时候 Spec Decoding 反而变慢

```
关键细节：接受是"前缀式"的——第 i 个 token 被拒，i 之后的全部作废。
所以期望接受数不是 k×α，而是等比数列：

  E[接受数] = α + α² + ... + α^k = α(1-α^k)/(1-α)   (α = 单 token 接受率)

  k=5, α=0.9: E ≈ 3.7   → 一轮验证顶 4.7 个 token（含验证步自产的 1 个），划算
  k=5, α=0.5: E ≈ 0.97  → 一轮才多出 ~1 个 token，
              而这一轮多花了 5 次 draft forward + 1 次 batch=6 的大模型 forward
  α 再低 → 净变慢

何时变慢的判定：draft 开销 + 验证开销 > 接受 token 数 × 单步 decode 时间
关键：draft 质量要高（α 高）、target 要够大（单步贵，省下的步才值钱）。
注意 trade-off：大 batch 时 GPU 本来就被喂饱（decode 不再深度 memory-bound），
投机解码"用闲置算力换步数"的空间反而缩小——spec decode 在小 batch/低延迟
场景收益最大，高吞吐场景要重新测。
```

---

## 5. 学习检查清单

- [ ] 能说清 iteration-level scheduling 解决了什么问题
- [ ] 能画 chunked prefill 的时序图，解释"捎带"机制
- [ ] 能手画 P/D 分离架构图（P 池 → KV transfer → D 池）
- [ ] 能讲清 P/D 分离比合并部署好了什么、新增了什么成本
- [ ] 能解释 speculative decoding 的验证流程和为什么不掉精度
- [ ] 能说清什么情况下 spec decode 反而变慢

---

## 6. 自测 / 面试题

1. 不用 chunked prefill，长 prompt 进来时 decode 用户 TPOT 为什么抖动？
2. P/D 分离相比合并部署，省了什么又新增了什么成本（KV transfer）？
3. 投机解码为什么不损失精度？draft 接受率低时为什么变慢？
4. 假设你有一个模型，prefill compute-bound 和 decode memory-bound，你怎么选调度策略？
5. Continuous batching 中，一个新的长 prefill 请求来了，你怎么调度不会污染 decode？

---

## 7. 推荐阅读

| 资料 | 来源 |
|------|------|
| Orca: Iteration-Level Scheduling (OSDI'22) | Seoul National University |
| Sarathi-Serve: Chunked Prefill (OSDI'24) | Microsoft Research India |
| DistServe: Prefill/Decode Disaggregation (OSDI'24) | 北大 (Yinmin Zhong et al.) + UCSD |
| Splitwise (ISCA'24) | Microsoft Azure Research |
| Mooncake: KVCache-Centric Disaggregation (FAST'25 最佳论文) | **月之暗面 Moonshot AI (Kimi)** + 清华 |
| NVIDIA Dynamo: Disaggregated Serving (2025) | NVIDIA |
| Fast Inference via Speculative Decoding (2022) | Leviathan et al. (Google) |
| Medusa: Multi-Head Speculative Decoding | Tianle Cai et al. (Princeton 等) |
| EAGLE / EAGLE-2 / EAGLE-3 | 北大 Yuhui Li et al. |
| vLLM Scheduler 源码 (`vllm/core/scheduler.py`) | vLLM GitHub |
