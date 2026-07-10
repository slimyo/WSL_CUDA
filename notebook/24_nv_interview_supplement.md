# 面试题补充：硬件系统设计 / 分布式并行 / GPU 底层知识

> 对象: 冲刺 Tier-1 推理/算子岗
> 前置: 完成 06-23 章，此文件补充前述 notebook 未覆盖的面试题
> 目标: 覆盖 NV 系厂（NVIDIA、微软、字节 AML 等）面试中出现的"计算-通信联合设计"与"GPU 硬件行为"题

---

## 1. Matmul 最优分块与 MoE TopK 的 Roofline 计算

### 1.1 Matmul 最佳输入大小：Roofline 求交点

```
给定 GPU 的峰值算力 P (TFLOPS) 和 HBM 带宽 B (GB/s)，
Matmul C(M,N) = A(M,K) x B(K,N) 的计算量 = 2*M*K*N FLOPs，
访存量 = M*K + K*N + M*N (约 M*K + K*N, 输出可 cache 时不算)。

计算密度 I = 2*M*K*N / (M*K + K*N)

为 reach 峰值算力，需 I >= P/B (Roofline 交点)。

以 H100 SXM (1979 TFLOPS, 3.35 TB/s) 为例：
  P/B = 1979e12 / 3.35e12 ~ 591 FLOPs/byte

若 M=N=K (方阵)：
  I = 2*N^3 / (2*N^2) = N
  需 N >= 591 才能 reach 峰值算力。
  若 N=128, I=128 -> 实际吞吐仅 ~128/591 * 1979 ~ 429 TFLOPS（memory-bound）。

若 batch=1, d_model=4096 (M=1, K=N=4096)：
  I = 2*1*4096*4096 / (1*4096 + 4096*4096) = 2*4096^2 / (4096 + 4096^2) ~ 2*4096^2 / 4096^2 ~ 2
  计算密度仅 2 FLOPs/byte -> 深度 memory-bound（decode 阶段）

结论：
  - 大矩阵（N > 5000+）：compute-bound，可达峰值
  - 小矩阵（N < 500）：memory-bound，拼带宽/减少访存
  - Matmul 形状 (M,N,K) 决定瓶颈，block tile 大小匹配 SM 资源和 L1/SMEM
```

### 1.2 Optimal Tile Size 的 SM 级计算

```
一个 SM 上执行 GEMM tile，tile 尺寸受制约于：

(1) 寄存器数限制
  H100: 65536 regs/SM, 2048 threads/SM (最大)
  若每个 thread 负责计算 tile 的一小块，每个 thread 的 reg 占用量 = 各类型累加器
  例：每个 thread 算 16x8 output tile (128 个 fp32 累加器 = 128 regs)
  -> 每个 SM 最多 65536/128 = 512 threads -> 16 warps（受 warp scheduler 16 限制）

(2) Shared Memory 限制
  H100: 228 KB shared memory/SM (可配置)
  Tile (BK, BN) 的 SMEM 占用 = BK*BN*sizeof(type)
  BK=64, BN=64, fp16 -> 64*64*2 = 8 KB (A tile) + 8 KB (B tile) = 16 KB (小)
  BK=128, BN=128 -> 32 KB + 32 KB = 64 KB (适中)
  BK=256, BN=128 -> 64 KB + 32 KB = 96 KB (需注意 L1/SMEM 分区)

(3) Occupancy vs tile 尺寸的 trade-off
  Tile 越大 -> 计算密度越高 -> 大矩阵好
  Tile 越小 -> more tiles in flight -> 更好的 latency hiding -> 小矩阵好
  所以 CUTLASS/A100 上常见 tile: 128x256x64 (MxNxK per CTA)

最佳实践：
  用 cuBLAS / CUTLASS 的 auto-tuning 选 tile size，不同形状有不同最优 tile。
  面试中关键说清"三个约束"：regs / SMEM / occupancy。
```

### 1.3 MoE TopK 最佳 Token 数 (Roofline 推导)

```
MoE 场景：每个 token 选 top-K experts，每个 expert 做一个 GEMM。

单 expert GEMM: X (N_tokens_per_expert x d_model) x W (d_model x d_ff)
  N_tokens_per_expert = N_total_tokens * (K / E_experts) * (load balance factor)

计算密度 N_tokens_per_expert 是关键：

以 DeepSeek-V3 (d_model=7168, d_ff=2048, E=256, K=8, 每卡 160 experts) 为例：

  每个 expert 的 GEMM 计算量 = 2 * N_tok * d_model * d_ff
  访存量 = N_tok * d_model + d_model * d_ff
  计算密度 I = 2 * N_tok * d_model * d_ff / (N_tok * d_model + d_model * d_ff)
             = 2 * N_tok * d_ff / (N_tok + d_ff)

  当 N_tok >> d_ff (如 N_tok = 4096, d_ff = 2048)：
    I ~ 2 * d_ff = 4096 FLOPs/byte -> compute-bound

  当 N_tok << d_ff (如 N_tok = 16, d_ff = 2048)：
    I ~ 2 * N_tok = 32 FLOPs/byte -> memory-bound

  以 H100 (P/B ~ 591) 为分界：
    N_tok_optimal >= 591 才能达到 compute-bound。
    但 N_tok = 总 token / expert 数，增大 N_tok 意味着减少 expert 数或增加 batch。

  TopK 实际工程含义：
    若每 expert 分到的 token 太少（< 256），GEMM 变成 memory-bound，
    需要在 early dispatch 前攒够 token（fused dispatch + big batch）。

  优化方向：DeepSeek-V3 的 prefill 用大 batch 让 N_tok_per_expert > 512；
    decode 阶段用 small batch，但重叠 dispatch-compute，隐藏 memory-bound 的损失。
```

---

## 2. PrefixCache vs KVCache

### 2.1 概念区分

```
KVCache（KV 缓存）：
  - 存储 attention 中 key/value 的中间结果
  - 每个 sequence 独享，随着 decode 逐步追加
  - 存在显存中，key/value 的形状: [num_layers, 2, batch, num_heads, seq_len, head_dim]
  - 核心目的：避免 decode 阶段重复计算历史 token 的 KV 值

PrefixCache（前缀缓存）：
  - 缓存 *多个请求之间复用的* 公共前缀（system prompt / 用户 prefix）
  - 若请求 A 和 B 共享前缀 (e.g. 同一个 system prompt)，
    请求 B 不必重新计算前缀的 KV -> 直接从 cache 加载
  - 通过 hash 匹配前缀内容（SGLang 的 RadixAttention 核心）
```

### 2.2 关键区别

```
| 维度 | KVCache | PrefixCache |
|------|---------|-------------|
| 粒度 | per-token per-layer | per-prefix (连续 segment) |
| 复用范围 | 同一 sequence 的 decode | 不同 sequence 的公共前缀 |
| 实现方式 | vLLM 的 PagedAttention block table | SGLang 的 RadixTree (前缀树) |
| 缓存策略 | all tokens (无选择) | 仅缓存公共前缀段 |
| 解决核心问题 | decode 阶段重计算避免 | prefill 阶段重计算避免 |
| 命中方式 | 自动命中（同 sequence） | 前缀哈希匹配（跨 sequence） |
| 淘汰策略 | 类似 OS LRU/SJF（vLLM 的 Garbage Collection） | LRU 或引用计数（RadixTree 自动维护） |
```

### 2.3 实现方式对比

```
PagedAttention (vLLM):
  - KV cache 分 page (block = 16 token)，block table 映射逻辑 page -> 物理 block
  - PrefixCache 在 SGLang 中由 RadixAttention 实现：
    共享前缀存储在 RadixTree 中，新请求通过 trie 匹配最长的公共前缀，
    跳过已计算的前缀，只计算未匹配的部分
  - 显存节省效果：system prompt 长时（如 Agent 场景），可省 80%+ prefill 计算

实现接口：
  SGLang: radix_cache = RadixCache(), cache.get_prefix_node(hash(input_ids))
  vLLM: hash-based prefix caching (>= v0.4.0)

面试要点：PrefixCache 实质是在 KVCache 之上增加"跨请求共享"的逻辑层；
  底层存储仍是同一个 KVCache block pool，只是 block 的生命周期和引用计数变了。
```

---

## 3. Chunk Prefill + MLA 激活值优化

### 3.1 问题：Chunk Prefill 下 MLA 的 KV proj 激活爆炸

```
DeepSeek-V3 的 MLA (Multi-head Latent Attention)：
  - 标准 attention 的 KV 先做 low-rank projection，从 d_model -> kv_lora_rank
  - Chunk Prefill 时，KV 一次性计算完毕，再做 attention
  - 问题：
    KV proj 后的激活值形状 = [batch, n_heads, chunk_size, head_dim]
    如果 chunk_size = 4096, batch = 64, n_heads = 128, head_dim = 128:
      激活值大小 = 64 * 128 * 4096 * 128 * 2(fp16) = 8 GB per layer！
    即使只存 1 层也超出 HBM (80 GB) 可承受范围。

根本原因：chunk prefill 为了利用 tensor core，把整段的 KV 一次性投影，
  而 MLA 的 KV proj 输出是后续所有 head 都共享的，中间不压缩。
```

### 3.2 解决方案

```
方案 A：分块计算 + 内存复用（最主流）
  不把整个 chunk 的 KV proj 结果存下来，而是：
  1. 把 chunk 分成多个 sub-chunk (如 512 tokens)
  2. 对每个 sub-chunk 分别做 KV proj
  3. 做 sub-chunk 内部的 attention
  4. 积累输出结果（online softmax 技巧，参见 FlashAttention）
  代价：多次 KV proj 调用（计算量增加），但避免了大中间 tensor

方案 B：FlashAttention V3 的 persistent kernel
  1. KV proj 和 attention 在同一个 kernel 里 fuse
  2. KV proj -> SMEM -> 直接做 attention（无需落 HBM）
  3. 激活值从 8 GB 降到 SMEM 内 ~200 KB
  代价：kernel 复杂度高、occupancy 可能下降

方案 C：利用 MLA 的 low-rank 特性
  MLA 本身就是对 KV 做压缩。如果计算图中 KV proj 后的 latent vector
  本身就在 latent space 里保持很小（如 kv_lora_rank=512），
  可以让 attention 直接在这个 latent space 里做（MLA 的原始设计），
  不需要展开到 head_dim 维度。这样中间激活 = [batch, chunk, kv_lora_rank] 而非
  [batch, n_heads, chunk, head_dim]，可大幅减少。

    标准 MLA: latent (dkv) -> 展开到 n_heads * head_dim -> attention
    优化 MLA: 注意力直接在 latent space 做 (W_k and W_v 合并为 W_kv_latent，
              Q 也投影到 latent space，inner product 在 latent 计算)
    这实质就是 DeepSeek-V3 MLA 节省 KV cache 的同一套思路延伸到 prefill 激活。

面试标准回答：
  先陈述问题（chunk prefill 把 KV proj 输出变得很大, 数学上推激活尺寸），
  再给方案（sub-chunk + online softmax 最实用，fuse kernel 次之，
  利用 MLA latent 最优雅但需要模型设计配合）。
```

---

## 4. Agent 场景下 KV Cache 空间不足的优化

### 4.1 Agent 场景特征

```
Agent: 模型反复调用工具/搜索，每次调用是独立的 generation
特征：
  - 每次 generation 的 input tokens 累计（history + new observation）
  - conversation length 快速增长（10K ~ 100K+ tokens）
  - 对延迟敏感（用户等待 agent 思考路径）
  - 多个 agent session 并存时显存压力极大
```

### 4.2 优化方案

```
(1) Prefix Caching (RadixAttention)
   - 缓存历史对话的公共前缀 KV（每次 agent 调用时 system prompt 不变）
   - SGLang 实现：trie-based prefix matching
   - 节省 prefill 计算和部分 KVCache（不重复存共享前缀）

(2) KV Cache Eviction / Compression
   - 不存全部 KV，只存最近 N 个 token + 重要历史 token
   - H2O (Heavy Hitter Oracle): 观测到 attention score 大的 token 更重要
   - StreamingLLM: 保留初始 token (attention sink) + 最近窗口
   - SnapKV: 压缩历史 KV 到固定预算（适合长 context）

(3) 共享 KV Cache (Cross-agent KV sharing)
   - 如果多个 agent 用了相同的 tool output / search result 作为 context，
     只存一份 KV，多个 session 共享引用（CO2 和 CacheGen 等方案）
   - GPU -> CPU offloading：少用 KV 挪到 CPU RAM，用时载回
   - 代价：offload 延迟约 10-50 GB/s (PCIe) vs 3.35 TB/s (HBM)

(4) 4-bit KV Cache 量化
   - KV cache 量化到 FP8 (H100) 或 INT4 (Blackwell)
   - 直接砍 2-4x KV cache 占用
   - 代价：精度损失，但在 agent 场景通常可接受

面试举例：
  假设 70B, 64 agent sessions, avg 20K tokens/session, GQA (8 KV heads)：
    KV per session = 2 * 64_layers * 8_KV_heads * 20K * 128 * 2(fp16) = 5.24 GB
    64 sessions -> 335 GB，远超 H100 80GB。
  优化后（FP8 + prefix caching + heavy hitter eviction）：
    ~ 335 * 0.5(FP8) * 0.5(eviction) * 0.7(prefix共享) ~ 58 GB -> 可容纳。
```

---

## 5. PD 分离的权衡

### 5.1 PD 分离为什么不一定好

```
PD 分离（Disaggregated Prefill/Decode）将 prefill 和 decode 分到不同 GPU：

优点（公认）：
  - 消除 P/D 干扰（prefill compute-bound 不污染 decode memory-bound）
  - 各自独立扩缩容（prefill 对算力敏感，decode 对带宽敏感）
  - 便于优化（prefill 池可配大 batch + 高算力卡，decode 池配大带宽卡）

缺点（面试考点——PD 分离不是银弹）：

(1) RDMA 传输延迟
    Prefill 输出的 KV cache 需从 prefill GPU 传送到 decode GPU (RDMA)
    KV cache 传输量 = 2 * layers * kv_heads * seq_len * head_dim * sizeof(type)
    以 70B (GQA, 8 KV heads, 128 dim, fp16) 为例，prefill 2048 token：
      传输量 = 2 * 64 * 8 * 2048 * 128 * 2 = 512 MB
    RDMA 400 GB/s -> 需 ~1.3 ms（理想），但实际因拓扑/带宽竞用可能 3-5ms
    这个延迟直接加在 TTFT（首 token 延迟）上。

(2) 负载不均衡
    Prefill 池偏 compute（加卡易），decode 池偏 memory（加卡收益递减）
    decode 池 GPU 可能因 batch 不足利用不充分，prefill 池可能 burst 超载

(3) Agent / Long Context 场景的致命问题
    Agent 场景：每个 agent step 输出很短（几十 token），但输入的 KV cache 很长
    -> prefill 计算量很小（短输入），但 KV cache 传输量很大（长历史）
    -> PD 分离的传输开销与 prefill 计算开销比例失调
    -> 可能不如 P/D 在同一卡上直接用 prefix cache 快

(4) 扩容复杂度
    需要 load balancer、kv cache migration、consistent hashing
    增加系统复杂度，小规模部署（单机 8 卡）收益不高
```

### 5.2 Agent + Long Prefix 场景的 PD 策略

```
长 prefix + 短 generation（Agent 工具调用）：
  方案 A: Prefill on Demand（不分离，共享 prefix cache）
    当前卡已有 prefix KV（来自历史 agent 调用）=> 无需传输
    P/D 同卡，无传输开销，适合小规模 agent 部署（< 4 机）

  方案 B: Hybrid PD（部分分离）
    大 prefill（> 4K tokens）走 PD 分离
    小 prefill（agent 中间步骤）本地做 prefix cache + decode
    传 KV 时只传 delta（新增的观测 token），不传全部历史

  方案 C: Agent-specific scheduler
    根据 agent session 的 KV cache 所在位置调度到同一 decode node
    减少 KV cache 迁移，特别适合 agent 场景（session ephemeral affinity）
```

---

## 6. Context Parallelism (CP): AG KV vs AG Q

### 6.1 两种方案对比

```
CP (Context Parallelism): 把 sequence 维度分到多张 GPU 上。
每张卡只持有部分 sequence token，各自做部分的 attention。

AG (All-Gather) 是两种常见的 CP 通信方案：

AG KV:
  - 每卡 AllGather KV（把跨卡的 KV 收集到所有卡）
  - 每卡本地做 full attention（Q 本地有，KV 是全的）
  - 通信量：每卡发出 (seq_len/CP_size * kv_dim)，收到完整 seq_len * kv_dim
  - 通信模式：AllGather
  - 优点：不需要额外的 reduce，每卡独立计算
  - 缺点：KV 全收集，通信量大（KV 通常比 Q 大，尤其 MLA 的 KV latent 也大）

AG Q:
  - 每卡 AllGather Q（把跨卡的 Q 收集到所有卡）
  - 每卡本地算 part of attention（KV 是本地的，Q 是全的）
  - 最后 ReduceScatter 合并 attention output
  - 通信量：Q 的 AllGather + output 的 ReduceScatter
  - 通信模式：AllGather + ReduceScatter
  - 优点：Q 通常比 KV 小（尤其在 GQA/MLA 场景）
  - 缺点：两轮通信；attention output 需要 reduce

性能对比：
  GQA: KV heads < Q heads -> KV < Q（近似）-> AG Q 两轮通信可能略高
  MLA: KV latent 维度小 -> KV < Q -> AG Q 不一定优于 AG KV
  MHA: KV heads == Q heads -> KV ~ Q -> 看 seq_len 和 head_dim 比例
```

### 6.2 Agent 场景选型

```
Agent 场景：prefix cache 很长，但每次 agent step generation 很短。

AG KV：
  - 每卡收集所有 KV（完整长序列的 KV everywhere）
  - 生成新 token 时无需再通信（因为 KV 已在本地）
  - 适合：long prefix + short decode
  - 代价：每卡显存 = 完整 KV（大）+ 本地 Q 部分（小）

AG Q：
  - 每次新 token 生成都要 Q AllGather + output ReduceScatter
  - decode 阶段反复做两轮通信
  - 适合：short prefill + long generation（多轮通信成本高）

结论（面试回答）：
  Agent 场景 prefix cache 极长 -> 应选 AG KV，一次通信换来整个序列无需再通信，
  AG Q 需要每步通信，在长 prefix decode 场景中累积延迟令人不可接受。

  如果是 prefill 爆发（而不是 long generation）：
    短 prefix（< 4K）-> AG Q 或 AG KV 差别不大
    长 prefix（> 32K）-> AG KV 更优（AG Q 的 ReduceScatter 延迟随 seq 增长）
```

---

## 7. Decode 阶段长输出时的负载均衡

### 7.1 问题本质

```
长输出 decode（如长文本生成 / Agent 多步推理）：
  - 同一 batch 内的请求长度差异大
  - 短的先 finish -> GPU 计算单元闲置（under-utilization）
  - 长的后 finish -> 拖慢整个 batch 的可见延迟
  - 负载不均衡具体形式：
    (1) 请求间输出长度差异 -> GPU 利用率锯齿
    (2) 层间或 expert 间计算量差异 -> 通信等待
    (3) PD 分离下 decode 池内 batch 大小波动
```

### 7.2 优化方法

```
(1) Dynamic Batching (Continuous Batching)
   标准解法：请求完成后立即插入新请求
   不等待整个 batch 完成 -> 减少 GPU 空闲时间
   vLLM / SGLang / TensorRT-LLM 都支持

(2) Preemptive Scheduling（抢占式调度）
   长 decode 请求的后续 token 可以延迟生成
   优先服务即将完成的新请求（提高 burst 下的 tail latency）
   类似 OS 的时间片调度（SGLang 的 preemption mode）

(3) Split-fuse / Chunk Prefill 的负载视角
   在 decode 池里，把长 prefill 拆成 chunk，插入 decode 的间隙
   利用 decode 的"空闲算力"来做 chunk prefill（但代价是干扰 decode 延迟）

(4) Expert-level Load Balancing (MoE 场景)
   某些 expert 负载过高（router 偏向它们）
   通过 auxiliary loss 让 router 尽量均匀分配
   DeepSeek-V3 还用了 device-level auxiliary loss -> 设备间负载更均衡

(5) Speculative Decoding 减少步数
   通过 draft model 一次生成多个候选 token -> 减少实际 decode 步数
   等同于"减少长输出请求占用 GPU 的 window size"

(6) 工作窃取 (Work Stealing) - 分布式场景
   空闲 decode 节点从负载高的节点"偷"请求
   需要 KV cache 迁移 + 请求状态同步
   SGLang 和 vLLM 的分布式调度器有相关实现
```

---

## 8. EP/DP 大小计算：以 DeepSeek-V3 为例

### 8.1 大 EP 和大 DP 的原因

```
EP (Expert Parallelism)：每个 expert 分配到不同 GPU
DP (Data Parallelism)：整个模型复制到不同 GPU，数据分配到各卡

为什么需要大 EP（多 expert）：
  - DeepSeek-V3: 256 experts, d_ff=2048 each
  - EP 越大 -> 每卡的 expert 数越少 -> 每卡计算量减小
  - 但 A2A (All-to-All) 通信量随 EP size 增大
  - 最优 EP 在 计算/通信 交叉点

为什么需要大 DP（多数据副本）：
  - 训练中全局 batch size 需足够大以保证收敛
  - 推理中 DP 用于 parallel prefill（不同请求分到不同 DP 副本）
  - DP 不涉及通信（每个副本独立），但消耗显存（完整权重 + KV cache 多份）
```

### 8.2 DeepSeek-V3 的 EP 大小计算

```
已知：
  - 每张 H100: 80 GB HBM, 1979 TFLOPS FP8
  - DeepSeek-V3: d_model=7168, d_ff=2048, E=256, K=8
  - 每个 token 激活 8 experts (K=8)
  - 单卡 batch_size=16, seq_len=4096 (prefill)

每卡计算需求（per expert per layer per token）：
  GEMM: 2 * d_model * d_ff = 2 * 7168 * 2048 = 29.4 MFLOPs

每卡存储需求（per expert）：
  权重: d_model * d_ff * 2(fp16) = 7168 * 2048 * 2 = 28 MB

EP=8: 每卡 256/8=32 experts
  计算: 16 * 4096 * 8(active) * 29.4M / 8(EP分工) / 1979T
    = 16 * 4096 * 8 * 29.4M / (8 * 1979T)
    = 每个 token 激活 8 experts，256/8=32 experts/卡，所以每卡 32 个 expert
    实际 prefill 中一个 token 只激活 K=8 个 expert，但分布在 256 张卡上
    EP=8 意味着这 8 个 expert 分布在 8 张不同的卡上 -> 1 expert/卡
    每卡 prefill 计算 = batch * seq * (1 expert GEMM) / 卡
    = 16 * 4096 * 29.4M = 1.93 TFLOPs
    耗时 ~ 1.93T / 1979T * sm_util = ~1ms (乐观)

  显存: 权重 = 32 * 28M = 896 MB (far from HBM limit)

EP=32: 每卡 256/32=8 experts
  每卡 prefill 计算 = (K/EP) * batch * seq * GEMMs
    = 8/32 * 16 * 4096 * 29.4M = 0.48 TFLOPs (更少计算)
  A2A 通信量 = (EP-1)/EP * d_model * batch * seq * type_size (约 512 MB per layer)
  当 EP 增大 -> 每卡计算量下降 -> 但 A2A 通信增大 -> 可能变成通信瓶颈
```

### 8.3 固定单卡 batch=16 时求最优 EP

```
批量公式：EP 的通信-计算比 = 通信时间 / 计算时间

  A2A 通信量（每 token 每 layer 每卡）:
    每卡发送: d_model * batch * seq (hidden state)
    A2A 总通信 = (EP-1) * d_model * batch * seq * sizeof(fp8)
    以 fp8 算： (EP-1) * 7168 * 16 * 4096 * 1 = (EP-1) * 470 GB per layer
    实际 A2A 是 collective，用 NVLink + RDMA（假设 900 GB/s NVLink）
    通信时间 = 470e9 * (EP-1) / EP / 900e9 = 0.52 * (EP-1)/EP ms (per layer per token 近似)

    每层每卡的计算时间 = (K/EP) * 29.4M FLOPs / (1979T FLOPs/s)
    (需考虑实际利用率约 60-80%)

    通信计算比 = 1 时最优
    -> (EP-1)/EP * 0.52ms = K/EP * 29.4M / 1979T * (1/0.7利用率)
    -> 近似求解得 EP ~ 16-32 为 optimal region

面试中关键说清计算过程而不是数字，展示你理解 trade-off：
  EP 小 -> 每卡计算多，通信少（适合 prefill，compute-bound 怕通信）
  EP 大 -> 每卡计算少，通信多（适合 decode，怕计算 idle）
  DeepSeek-V3 的配置: prefill EP=32, decode EP=256（充分利用各 expert 的独立计算）
```

---

## 9. PD 分配合比与 Decode Batch Size

### 9.1 PD 分配合比确定

```
问题：给定集群 M 张 GPU，多少给 prefill 池，多少给 decode 池？

方法：用吞吐模型推导

已知：
  - 每秒请求到达率 R (req/s)
  - 每个请求平均 prompt tokens = P, 平均 output tokens = O
  - Prefill 吞吐: T_prefill (tokens/s/GPU) = f(batch_size, seq_len)
  - Decode 吞吐: T_decode (tokens/s/GPU) = g(batch_size, seq_len)

系统稳态条件：
  Prefill 消费速度 = Decode 需求的 token 产生速度
  N_prefill * T_prefill = N_decode * T_decode / (output_tokens_per_req)

在最优分配合比下：
  N_prefill : N_decode = (P * 算力成本) : (O * 带宽成本)
  粗略估计：prefill GPU 数是 decode GPU 数的 P/O * (算力/带宽比)

  举例：7B 模型，prefill 1000 tok/s per GPU, decode 100 tok/s per GPU,
  P=2048, O=512 -> 每个 request 产生的计算需求：
    prefill 侧 = 2048 tok / 1000 tok/s = 2.04s 计算
    decode 侧 = 512 tok / 100 tok/s = 5.12s 计算
  GPU 数比 = P/O * (T_decode/T_prefill) = 2048/512 * 100/1000 = 0.4
  即 prefill : decode = 0.4 : 1 -> 对于 140 GPU, prefill 约 40, decode 约 100
```

### 9.2 Decode 最佳 Batch Size + 反推 Prefill 数量

```
Decode 最适 batch_size 的 Roofline 计算：
  decode 一个 token 的计算量 = 2 * d_model * d_ff (忽略 attention)
  访存量 = d_model * d_ff (reuse weight across batch, 不计 input/output tensor)

  batch_size 增大时：
    计算量 = batch * 2 * d_model * d_ff
    访存量 = d_model * d_ff (几乎不变，因为 weight 被 batch 共享)
    计算密度 I = batch * 2

  以 H100 (P/B ~ 591) 为参考：
    batch=1: I=2 -> 深度 memory-bound
    batch=16: I=32 -> 仍 memory-bound
    batch=128: I=256 -> 接近 compute-bound
    batch=256: I=512 -> compute-bound

  所以 decode batch 越大，计算利用越充分。
  但 batch 受三方面限制：
  (1) KV cache 显存 (每 token 约 2 * layers * kv_heads * head_dim * sizeof)
  (2) 延迟 SLA (大 batch 下单个 token 延迟增加)
  (3) 实际工作负载 (请求到达率能否支撑大 batch)

反推 Prefill GPU 数：
  设 decode pool 有 N_dec 张 GPU，每卡 batch = B_decode
  Decode 总吞吐 = N_dec * B_decode / (decode 延迟)
  若 decode 延迟 = 10ms per token, B_decode=32:
    Decode 总吞吐 = N_dec * 32 / 0.01 = 3200 * N_dec tokens/s
  若每 request 平均输出 O=256 tokens:
    每 req 平均 decode 时间 = 256 * 0.01 = 2.56s
    系统需 R_steady = N_dec * 32 / 256 = 0.125 * N_dec req/s

  Prefill 侧需匹配：
    Prefill throughput = R_steady * P / T_prefill_per_gpu
    若 T_prefill = 2000 tokens/s/GPU, P=2048:
      N_prefill = R_steady * P / T_prefill = 0.125 * N_dec * 2048 / 2000 = 0.128 * N_dec

  所以粗略比例：
    Prefill : Decode ~ 0.13 : 1
    即 decode 100 张 GPU 配约 13 张 prefill GPU。

  实际中这个比与模型大小、硬件、SLA 密切相关。
  面试关键：展示如何从 decode batch_size 反推 prefill 数量——通过稳态流量守恒。
```

---

## 10. MoE 并行 Overhead 与 A2A 通信量

### 10.1 MoE 并行的三阶段 Overhead

```
MoE (MoE layer) 在 EP 下的执行三阶段：
  1. Dispatch (前 A2A): token 根据 router 结果发送到对应 expert 所在卡
  2. Compute: 每卡算自己分到的 expert 的 FFN
  3. Combine (后 A2A): 每个 expert 的输出发回原 token 所属卡

各阶段延迟：
  Dispatch: 每卡发送 token_count * d_model * sizeof(fp8) 字节
  Compute: K/EP * 2 * d_model * d_ff FLOPs
  Combine: 与 dispatch 对称

  A2A 通信量公式：
    每次 A2A 通信量 = token_per_card * d_model * sizeof(type)
    Dispatch: send -> 每卡发 (total_tokens * K/EP) * d_model
    Combine: receive -> 每卡收 (total_tokens * K/EP) * d_model

  Overhead 分析：
    prefill (大 batch): compute 大，A2A 占比低 (< 10%)
    decode (小 batch): compute 小，A2A 占比高 (可 > 50%)
```

### 10.2 A2A 通信量详细计算

```
以 DeepSeek-V3 (d_model=7168, E=256, K=8) prefill 场景：

batch=64, seq=4096, EP=32:
  每卡总 tokens = batch * seq / EP = 64 * 4096 / 32 = 8192 tokens/卡
  每卡需 expert 计算 token = 8192 * K/EP = 8192 * 8/256 ... 这里不对

  正确：
    总 tokens = 64 * 4096 = 262144 tokens
    每个 token 激活 K=8 experts
    总 expert 调用 = 262144 * 8 = 2.1M 次 expert FFN
    EP=32 -> 每卡处理 = 2.1M / 32 = 65536 expert FFN 次
    但每个 expert FFN 由 d_ff=2048 决定，每个 expert FFN 计算量 = 2*d_model*d_ff

  简化版（面试中用的通用公式）：
    每卡 A2A 前通信（dispatch）:
      每 card 发送 = batch * seq * d_model * K / EP / (E/EP)
      实际上 dispatch 不是 K/EP，而是每个 token 把 hidden states 发给所有包含其被选 expert 的卡
      更精确：每卡发（收）的 token 总量 = batch * seq * K / E * (EP-1) * d_model (worst-case)

  实际面试标准回答路径（不用背数字，只要通路）：
    1. MoE 并行 = 前 A2A + compute + 后 A2A
    2. 每步 A2A 通信量 = token_count * d_model * sizeof(type) * (EP-1)/EP
    3. 通信时间 = 通信量 / 互联带宽 (NVLink 900 GB/s, RDMA 400 GB/s)
    4. 计算时间 = FLOPs / 峰值算力 * utilization
    5. 典型 result: prefill 通信 ~10-15% 时间，decode 通信可到 30-50%
    6. 优化: overlap 通信-计算（NVSwitch 和 NCCL 支持）
```

---

## 11. 4 台服务器的矩阵乘并行方案

### 11.1 全连接拓扑：C=AB 的分布式方案

```
问题：4 台服务器，每台内存 NN，全连接互联（每台直接连其他 3 台），
计算 C(N,N) = A(N,N) x B(N,N)。

假设 N 大到无法放入单台内存（N*N*sizeof > NN）。

方案 1：2D Block Cyclic（ScaLAPACK 风格）
  将矩阵分成 P_rows x P_cols = 2x2 块（4 台机器）
  每台存 A 和 B 的对应分块
  计算时，每台需要 A 的整行块和 B 的整列块 -> 需要广播/收集

  通信：
    A 的广播: 每台需要 A_block_row 从其他行机器来
    B 的广播: 每台需要 B_block_col 从其他列机器来

  实际上 canonical 方案是 SUMMA (Scalable Universal Matrix Multiplication Algorithm):
  1. 把 A 按列分块 [A1|A2]，B 按行分块 [B1; B2]
  2. 每台 server 负责 C 的一个 2x2 block
  3. 分 2 步：每步 broadcast A的一块和 B的一块
  4. 每步做局部 GEMM -> accumulation

  4 台 SUMMA:
    拓扑：4 台连成 2D torus (2x2 grid)
    每台 rank (i,j)
    第 k 步: broadcast A(i,k) 到 row i, broadcast B(k,j) 到 col j
    局部计算: C(i,j) += A(i,k) * B(k,j)
    总步数: 2 (因为 block=2 columns/rows)
    通信量: 每台每步收发 N/2 * N 的矩阵块

方案 2：1D 行分块（简化）
  每台存 N/4 行 A 和全部 B
  每台计算 C 的 N/4 x N 部分
  通信：B 需要全 broadcast -> 每台发 N*N 收 N*N
  显存：每台存 A: N*N/4, B: N*N -> 共 1.25*N*N
  局限：B 要全放内存，不满足"每台内存 NN"可能不够

方案 3：混合 1D+2D（实际常用）
  把 A 和 B 都做 2D 分块
  每次计算时，A 的 column block 按 row broadcast，
  B 的 row block 按 column broadcast
  CUDA-aware MPI + NCCL 实现
```

### 11.2 缓存限制 N*N/8 时的并行

```
问题：每个节点缓存（fast memory）只有 N*N/8，
放不下 A 或 B 的完整分块。

分析：
  N*N/8 意味着缓存只能存 1/8 的 A 或 B。
  若 N=4096, N*N=16M elements, fp32=64 MB
  N*N/8 = 8M elements = 32 MB（典型 L2 cache 大小）

这意味着任何单节点不能 hold A 和 B 的连续大块 -> 必须分得更细。

方案：Tiled SUMMA with Pipelining

1. Tile size T 满足 T*T < N*N/8:
   T = sqrt(N*N/8) = N / sqrt(8) ~ N/2.8
   但 T 不能太小（通信效率低）

2. 实际选择 T = N/4 或 N/8
   让每个 tile 适合 L2 cache

3. 执行：
   对每个 k-tile (A的列分块, B的行分块):
     (a) 加载 A 的一个 vertical panel (T x N) 到 node 的缓存
     (b) 加载 B 的一个 horizontal panel (N x T) 到缓存
     (c) 做局部 GEMM (C += A_panel * B_panel)
     (d) C (N x N) 的对应部分累积在寄存器或 L1/SMEM

4. 如果 T = N/8:
    每个 tile A: N/8 * N = N^2/8 (正好等于缓存)
    每个 tile B: N * N/8 = N^2/8 (也正好等于缓存)
    C 的 tile: N/8 * N/8 = N^2/64 (小)
    计算/通信比: tile 大小适合缓存不能太小 -> 维持高计算密度

面试关键：说清"缓存吃不下完整矩阵 -> 分 tile -> tile 大小刚好把缓存填满最有效率"
  这个设计思路和 GPU GEMM 里利用 shared memory 的 block tiling 完全一致。
```

---

## 12. AtomicAdd 顺序随机与算子非确定性

### 12.1 相同输入是否输出随机？

```
GPU 上的算子多数是确定性的（deterministic），但以下情况会随机：

(1) AtomicAdd 顺序随机
    当多个 thread 对同一显存地址做 AtomicAdd 时：
    - hardware 不保证 atomic 操作的顺序
    - 每次 kernel launch，thread 的 warp schedule 可能不同
    - 所以多个 AtomicAdd 的累加结果顺序不同，但 *最终总和相同*（因为加法可交换）
    - 但如果是非可交换操作（如 atomicCAS 实现链表插入），结果也会非确定

    解决方案：
      NVIDIA 提供了 cudaStreamSynchronize + cudaMemcpy 保证 host-device ordering
      但不提供 GPU 内的 atomic ordering（Atomics only guarantee atomicity, not ordering）
      如果需要确定性：手动做 warp-level reduce 代替 global atomic

(2) FP32 / FP16 累加顺序不同
    a + b + c: (a+b)+c != a+(b+c) 因为浮点不满足结合律
    不同 warp 的 reduce 顺序不同 -> 累加结果在最低几位不同
    NVIDIA 提供了 cublasSetMathMode(CUBLAS_DETERMINISTIC)（但性能下降）

(3) 动态并行（CUDA Dynamic Parallelism）
    子 kernel launch 顺序非确定

(4) CUDA Graph 重放（replay）时是确定性的
    graph capture 已确定了所有 warp 的 schedule -> 重放结果一致
    但 graph capture 的过程可能非确定（如果涉及 atomic 操作）

面试标准回答：
  一般并行 reduce -> 非确定（atomic 或浮点结合律）
  单 thread 计算 -> 确定
  CUDA Graph -> 确定
  需要确定性时：用 warp shuffle + 手动指定 reduce tree，或 cuBLAS deterministic mode
  实际训练/推理中：通常不要求 bit-level 确定（噪声影响可忽略），debug 时才开
```

---

## 13. 两个进程同时操作 Global Memory

### 13.1 基本问题

```
两个进程能否同时操作同一个 global memory 地址？

可以，但有条件：

(1) 统一虚拟地址空间 (UVA, Unified Virtual Addressing)
    如果两个进程共享 CUDA context（通过 cudaIpcOpenMemHandle / MPS / MIG）：
      - 两个进程可以访问同一个 GPU 物理地址
      - 谁先抢到总线谁先执行（类似 CPU 上的 shared memory）
      - 需要 atomic 操作来同步

(2) 通过 CUDA IPC
    Process A: cudaIpcGetMemHandle -> 传 handle 给 B
    Process B: cudaIpcOpenMemHandle -> 获得同一 GPU 物理内存映射

(3) MPS (Multi-Process Service)
    多个 CPU 进程共享同一个 CUDA context -> 默认共享 GM 地址空间
    MPS 模式下，不同进程的 kernel 可以同时跑在相同 SM 上
    MPS 提交到同一个 command queue -> kernel 执行顺序不确定

(4) 虚拟地址 vs 物理地址
    GPU 有虚拟地址空间，与 CPU 类似：
      - 每个 context 有独立虚拟地址空间（默认）
      - 通过 cudaIpc 共享时，双方虚拟地址可以不同（映射到相同物理页）
      - GM 地址指的是 global memory 空间（物理显存）
      - 进程间操作"同一个 GM 地址"，是指操作同一个*物理*显存地址
      - 写同一物理地址无 cache coherence 问题？Yes GPU L2 cache 是统一的，
        但 write through / write back 策略可能不同 -> 需要 cudaDeviceSynchronize
```

---

## 14. 算子的常量参数存放在哪里

```
问题：kernel 中常量的参数（如 tile size、grid dim、scaling factor）存在哪个 memory？

答案分层：

(1) 编译期常量（template parameter / constexpr）
    -> 直接编译进指令的 immediate field
    -> 存放在 instruction memory（不在 HBM 读写路径上）
    -> 零访存开销
    例：template <int BLOCK_SIZE> __global__ void kernel()

(2) 调用时传入的参数（kernel<<<grid, block, shared_mem, stream>>>(param1, param2...)
    -> 存放在 kernel 参数空间 (parameter memory)
    -> 通过 constant memory bank 读取（有 constant cache）
    -> 所有线程读到相同值 -> broadcast 到 warp（一个 cycle, 不像 global 需 32 个）
    -> 大小限制：4096 bytes（取决于架构）

(3) __constant__ 数组
    -> 编译器分配到 constant memory（64 KB，所有 context 共享）
    -> cached in constant cache per SM (约 8-10 KB)
    -> 同 warp 内所有 thread 读同一地址 -> 单 cycle
    -> 不同地址 -> serialized（按 warp 内线程顺序读取，变慢）

(4) 场景判断：
    | 参数类型 | 最佳位置 | 原因 |
    |----------|---------|------|
    | tile size, grid dim | 编译期模板参数 | 零开销 |
    | scaling factor, threshold | kernel 参数 | constant cache broadcast |
    | large lookup table (< 64KB) | __constant__ | cached per SM |
    | large lookup table (> 64KB) | texture memory / global | 64KB 不够时 |

常见常量的具体位置（以 HGEMM 为例）：
  - M, N, K (矩阵维度): kernel 参数（与 batch 相关，运行时变）
  - BLOCK_M, BLOCK_N, BLOCK_K (tile 大小): 模板参数（编译期定）
  - alpha, beta (scaling): kernel 参数
  - lookup table for dequant: __constant__ (< 64KB) 或 constant cache via kernel args
```

---

## 15. 单核下两个线程比一个线程更快的场景

```
问题（CPU 多线程）：单核上两个线程执行相同任务，为什么可能比一个线程快？

答案：

(1) 指令级并行 (ILP) vs 线程级并行 (TLP) 的 trade-off
    单线程受 ILP 限制（寄存器依赖、cache miss）
    如果单线程内存在大量 cache miss / memory stall，
    多线程让硬件（超线程 / SMT）在一个线程等内存时切换到另一个线程

(2) 超线程 / SMT (Simultaneous Multi-Threading)
    CPU 单核有多个逻辑 thread context (e.g. Intel: 2 per core)
    当一个 thread stall（L3 cache miss ~ 300 cycles, RAM ~ 5000 cycles）
    另一个 thread 可以继续执行
    两个线程的 IPC (instruction per cycle) 之和 > 单线程 IPC
    典型场景: memory-bound workload, 数据访问模式不规则

(3) GPU 类比
    Warp 之间的 latency hiding 就是 GPU 版的"多线程比单线程快"
    当一个 warp stall (global memory access ~ 200-800 cycles)，
    warp scheduler 切换到另一个可执行的 warp
    这就是为什么需要足够的 occupancy 来隐藏访存延迟

(4) 条件：单线程受限于 memory latency（而非计算）
    纯 compute-bound 任务 -> 两个线程不一定更快（共享执行单元）-> 可能反而变慢（上下文切换开销）
    memory-bound 任务 -> 两个线程通过 overlap latency 提升吞吐

GPU 版本（面试关联）：
  单 SM 上一个 warp 独占 -> 可能大量 stall
  多个 warp interleave -> 隐藏访存延迟
  所以 SM 上需要至少 8-16 active warps 来达到满带宽（这是 occupancy 重要性的根源）
```

---

## 16. 面试题与现有 Notebook 对照

```
以下标记现有 notebook 已覆盖（无需本文件补充的内容）：

问题 | 覆盖位置
-----|---------
FlashAttention + FlashDecoding | 10_flashattention_deep_dive.md
MHA / MQA / GQA / MLA 对算子影响 | 11_attention_variants.md
GPU Memory 种类 | 03_gpu_memory_hierarchy.md
算子优化方法 | 14_kernel_routes.md, hgemm_optimization.md
Roofline 模型 | 06_roofline_and_flops.md
Continuous Batching | 13_scheduling.md
PagedAttention / RadixAttention | 12_kv_cache_management.md
NVLink / PCIe / RDMA | 16_distributed.md (6.2 节)
Speculative Decoding | 22_sampling_decoding.md
FP32 精度阈值 | 07_numerical_formats.md
Constant Memory | 03_gpu_memory_hierarchy.md (Q5)
Distributed TP/PP/EP/DP 概览 | 16_distributed.md
MPS / MIG | 20_cuda_streams_async.md
```

---

## 索引

> 本文件覆盖的面试题速查索引

```
| 编号 | 问题 | 本文件章节 |
|:---:|------|:--------:|
| 1 | Matmul 最优分块 + Roofline 最佳 token 数 | 1.1-1.3 |
| 2 | PrefixCache 与 KVCache 区别 | 2.1-2.3 |
| 3 | Chunk Prefill + MLA 激活值大 | 3.1-3.2 |
| 4 | Agent 场景 KV Cache 空间不足 | 4.1-4.2 |
| 5 | PD 分离是否一定好 | 5.1-5.2 |
| 6 | CP: AG KV vs AG Q | 6.1-6.2 |
| 7 | Decode 长输出负载均衡 | 7.1-7.2 |
| 8 | 大 EP/DP 原因 + DeepSeek 实例计算 | 8.1-8.3 |
| 9 | PD 分配合比 + Decode batch 反推 Prefill | 9.1-9.2 |
| 10 | MoE 并行 Overhead + A2A 通信量 | 10.1-10.2 |
| 11 | 4 台服务器矩阵乘并行 | 11.1-11.2 |
| 12 | AtomicAdd 随机 + 算子非确定性 | 12.1 |
| 13 | 两个进程操作同一 GM 地址 | 13.1 |
| 14 | 算子里常量参数存放位置 | 14.1 |
| 15 | 单核两个线程更快的情况 | 15.1 |
| 16 | 已有 Notebook 覆盖核对 | 16.1 |
```
