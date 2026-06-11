# 分布式推理并行：TP / PP / EP / SP 与 NCCL

> 对象: LLM 推理工程
> 前置: 09_inference_workload.md, 08_transformer_architecture.md, 13_scheduling.md
> 目标: 面试能手画 Megatron TP 的 all-reduce 位置、说清 EP/SP/CP 思路

---

## 1. 为什么要分布式？

```
大模型放不进单卡（显存不足）或单卡 inference 太慢。

常见单卡限制：
  Llama-7B: ~13 GB (FP16) + KV cache → A100 80GB 还 OK
  Llama-70B: ~130 GB (FP16) → 需要至少 2 张 A100 (80GB × 2)
  Llama-405B: ~810 GB → 多机多卡
```

---

## 2. Tensor Parallelism (TP, 张量并行)

### 2.1 Megatron-LM 范式

**思路：把单层的 GEMM 切成多份，分到不同 GPU 上并行算。**

### 2.2 Attention 层的 TP

```
每个 head 独立 → 按 head 维度切分

MHA (num_heads=32, TP=2):
  GPU 0: head 0-15
  GPU 1: head 16-31

QKV proj: 不需要通信（每个 GPU 有独立权重）
attention: 每 GPU 算自己的 16 个 head
Output proj: 需要 all-reduce（所有 GPU 的结果需要合并）

通信：output proj 后做一次 all-reduce
```

### 2.3 FFN 层的 TP

```
FFN = gate(x) + up(x) → silu → down

列切分（gate/up）：
  GPU 0: W_gate[:, :11008/2], W_up[:, :11008/2]
  GPU 1: W_gate[:, 11008/2:], W_up[:, 11008/2:]
  每 GPU 独立算 gate/up

行切分（down）：
  GPU 0: W_down[:11008/2, :], GPU 1: W_down[11008/2:, :]
  down 后需要 all-reduce 合并

通信：每层 2 次 all-reduce（output proj 和 FFN down）
```

### 2.4 TP 通信量估算

```
关键认知：all-reduce 传的是【激活】，不是权重！
被 reduce 的张量形状 = [batch × seq, hidden]（output proj / FFN down 的输出）

每次 all-reduce 的数据量（每卡视角，ring all-reduce 收+发 ≈ 2×(TP-1)/TP × 消息）:
  消息大小 = batch × seq × hidden × 2B

Llama-7B (hidden=4096), FP16:
  Prefill (seq=4096, batch=1):
    每次 all-reduce 消息 = 4096 × 4096 × 2B = 33.5 MB
    每层 2 次 → 67 MB/layer → 32 层 ≈ 2.1 GB
    NVLink 900 GB/s → ~2.4ms；PCIe Gen5 64 GB/s → ~34ms（直接吃掉 TTFT）
  Decode (1 token, batch=1):
    每次消息只有 4096 × 2B = 8 KB → 带宽根本不是问题，
    瓶颈变成每层 2 次集合通信的【启动延迟】(~若干 μs)
    → 32 层 × 2 次 × ~5μs ≈ 0.3ms+，对 TPOT 是不可忽略的固定税

所以：TP 的 prefill 怕带宽不够，decode 怕延迟太多——都指向"必须 NVLink 域内"
```

### 2.5 TP 面试要点

```
"TP = 切权重、all-reduce 激活。每层 2 次 all-reduce（attention out proj 后、
 FFN down 后）。通信量正比于 batch × seq × hidden，与权重大小无关。
 prefill 看带宽、decode 看延迟，所以 TP 一般不出 NVLink 域（单机 8 卡）。"
```

---

## 3. Pipeline Parallelism (PP, 流水并行)

### 3.1 基本思路

```
把模型按层切成多段，每 GPU 负责连续几层。

GPU 0: layers 0-7
GPU 1: layers 8-15
GPU 2: layers 16-23
GPU 3: layers 24-31

推理：GPU 0 → GPU 1 → GPU 2 → GPU 3（串行）
```

### 3.2 Bubble（气泡）

```
训练的 bubble：micro-batch 流水填充/排空期间部分 stage 空闲。
推理的"bubble"形态不同：
  单请求时只有 1 个 stage 在干活，其余 stage 全闲 → 利用率 = 1/num_stages
  → 必须靠并发请求/micro-batch 把流水线灌满，吞吐才能 ×num_stages
  单请求延迟不会变好（甚至因 stage 间传输略变差）

对比：
  - TP：层内并行 → 降低单步延迟，但每层 2 次 all-reduce（高频通信）
  - PP：层间并行 → 只在 stage 边界传一次激活（低频通信，可跨节点），
        提吞吐不提延迟
典型组合：节点内 TP=8（NVLink），跨节点 PP/EP
```

---

## 4. Expert Parallelism (EP, 专家并行 - MoE)

### 4.1 背景：MoE 模型

```
Mixtral 8×7B: 8 个 expert，每 token 路由 top-2
DeepSeek-V3（细粒度 MoE，2025 主流形态）:
  每层 256 个小 routed expert + 1 个 shared expert，每 token 选 top-8
  总参数 671B、每 token 激活仅 37B → 推理算力省，但通信/调度复杂度暴涨
```

### 4.2 EP 做法

```
每个 expert 放在不同 GPU 上：
  GPU 0: expert 0, 1
  GPU 1: expert 2, 3
  GPU 2: expert 4, 5
  GPU 3: expert 6, 7

每步需要：
  - all-to-all 通信（把每个 token 路由到它所需的 expert）
  - expert 计算
  - all-to-all 回来

通信：all-to-all = 所有 GPU 和所有 GPU 交换数据
  → 通信量：每个 token 的 hidden 向量 × top_k × 2（dispatch+combine）

工程要点（2025）：
  - expert 负载不均 → 热 expert 所在卡成为长尾（EPLB 负载均衡器做 expert 复制/重排）
  - DeepSeek 开源 DeepEP：专为 MoE all-to-all 写的通信库
    （NVLink+RDMA 混合转发、FP8 dispatch、低延迟 decode 模式 + hook 式通信-计算 overlap）
  - 大规模部署趋势"大 EP"：与 P/D 分离结合，decode 池用几十~上百卡的 EP
    把每卡 expert 数压到极少 → 权重访存被摊薄（DeepSeek V3 推理系统即此架构）
```

---

## 5. Sequence / Context Parallelism (SP/CP)

```
超长上下文（128K+）时：
  - FlashAttention 的 O(N²) 复杂度开始显现
  - 单卡放不下全部 KV

SP/CP 沿 seq_len 维切：
  - 每个 GPU 处理一部分 seq_len
  - attention 计算需要跨 GPU 聚合（类似 TP 但沿 seq 维）
  - 通信模式：all-gather / reduce-scatter
```

---

## 6. 通信原语与硬件

### 6.1 常用集合通信

| 原语 | 操作 | 用途 |
|------|------|------|
| all-reduce | 所有 GPU 求和，结果各一份 | TP 的输出合并 |
| all-gather | 每 GPU 收集所有 GPU 的数据 | SP 的注意力聚合 |
| reduce-scatter | 求和后分散到各 GPU | 梯度累积 |
| all-to-all | 每 GPU 向所有 GPU 发送 | MoE 路由 |
| broadcast | 从 root GPU 发给所有人 | 参数分发 |

### 6.2 硬件互联

```
NVLink（节点内 GPU 之间，经 NVSwitch 全互联）:
  H100: 900 GB/s 双向 (NVLink 4)；GB200 NVL72 用 NVLink 5 (1.8 TB/s)
  把 72 张 GPU 连成一个 NVLink 域 → "机柜级单机"
  延迟: ~1μs 级

PCIe（GPU↔CPU、无 NVLink 时的 GPU↔GPU）:
  Gen5 x16: ~64 GB/s 单向
  比 NVLink 慢 10×+

RDMA / InfiniBand 或 RoCE（节点间）:
  NDR 400: 400 Gbps ≈ 50 GB/s / 卡
  比 NVLink 慢 ~20×

NCCL 是跑在这些链路之上的【集合通信库】（不是硬件）：
  自动探测拓扑，节点内走 NVLink/PCIe、节点间走 RDMA，
  实现 ring / tree 等 all-reduce 算法
```

---

## 7. P/D 分离 + 分布式

```
P/D 分离中的跨节点 KV cache 传输：
  prefill 池 → decode 池（RDMA / NVLink）

Mooncake 的 KV cache pool 架构：
  - 全局 KV cache pool（跨节点）
  - prefill 完成 → KV 入 pool
  - decode worker 从 pool 取对应序列的 KV
  - RDMA 零拷贝传输
```

---

## 8. 学习检查清单

- [ ] 能画 Megatron TP 对 attention + FFN 的切法，标出 all-reduce 位置
- [ ] 能估算每层 TP 的通信量
- [ ] 能解释 TP vs PP vs EP 的适用场景区别
- [ ] 能理解 all-to-all 在 MoE 中的作用
- [ ] 能说出 NVLink / PCIe / RDMA 的速度量级
- [ ] 能理解 P/D 分离中的 KV transfer 链路

---

## 9. 自测 / 面试题

1. Megatron TP 对 attention + FFN 怎么切？all-reduce 插在哪两处？
2. 为什么 TP 适合卡内 NVLink 而不适合跨节点？
3. MoE 的 all-to-all 通信量怎么估算？
4. P/D 分离中 KV cache 跨节点传输走什么链路？瓶颈在哪？
5. sequence parallel 在什么场景下需要？

---

## 10. 推荐阅读

| 资料 | 来源 |
|------|------|
| Megatron-LM: Training Multi-Billion Parameter Language Models | Shoeybi et al. |
| MoE / expert parallel 相关论文 | Google / DeepMind |
| NCCL 文档 | NVIDIA Developer |
| DeepSeek-V2/V3 MoE 并行方案 | DeepSeek |
| Mooncake: KV Cache-Centric Disaggregation | arXiv |
