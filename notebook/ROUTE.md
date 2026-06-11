# AI Infra 推理 / 算子岗 面试学习路线（依赖排序 · 可检索学习版）

> 目标：达到 Tier-1 厂（字节 AML、阿里通义/PAI、腾讯混元、华为诺亚等）推理 / 算子岗面试水平。
> 用法：每个模块给出 **依赖 → 核心知识点 → 能力要求 → 自测/面试题 → 资料关键词**。
> 你可以直接拿每条「知识点」和「资料关键词」去搜论文、博客、源码。

---

## 0. 能力分级（贯穿全文，每个知识点标注目标层级）

- **🟢 L1 认知**：能准确复述"是什么 / 为什么有它 / 解决了什么"，能在简历自述里提到。
- **🟡 L2 推导**：能在白板上推公式、画架构、讲清两条方案的 **trade-off**，能扛住一轮追问。
- **🔴 L3 实现**：能动手写出来（Triton/CUDA/玩具实现）或读懂主流框架源码并改。

算子岗最终要求：**Module 2 / 6 达 L3，其余至少 L2，你列的四块全部 L2 以上。**

---

## 依赖关系总图

```
[M0 硬件+数值地基]  ──┬─────────────────────────────┐
   roofline/精度/SM    │                             │
                       v                             v
              [M1 推理工作负载+指标]          [M6 Kernel实现路线]
              prefill/decode 二分              CUDA/CUTLASS/Triton/编译器
                       │                             │
        ┌──────────────┼───────────────┐            │
        v              v               v            │
  [M2 Attention数学  [M5 调度]    [M7 量化]          │
   +FlashAttention] (你的第1块) (你的第4块)          │
   (前置:flash是算子核心)  │           │             │
        │              │           （M7 的反量化 kernel 依赖 M6）
        v              │
  [M3 Attn变体+KV结构]  │
   MHA/MQA/GQA/MLA      │
        │              │
        v              │
  [M4 KV Cache管理]─────┘
  (你的第2块,依赖M3的KV大小+M2的flash kernel改造)

[M8 分布式并行]  依赖 M0+M1，与 M5 强相关（P/D 分离）
[M9 Profiling/工程]  横切，越早建立越好
```

**关键依赖逻辑（务必理解，这是路线"为什么这样排"的根据）：**

1. M0 的 **roofline + prefill/decode 二分** 是所有讨论的语言，不过这关后面全是死记硬背。
2. 你列的"调度/KV/Kernel/量化"四块，**没有一块能脱离 M2(FlashAttention) 和 M3(KV 结构) 单独讲清**——所以这两个前置模块必须先学。
3. M4(KV 管理) 同时依赖 M3（KV 有多大）和 M2（attention kernel 怎么从非连续内存取数），所以排在两者之后。
4. M5(调度) 的最大热点 **P/D 分离** 直接由 M1 的二分推出，并和 M8(分布式) 耦合。

---

## M0 — 硬件与数值地基

**依赖**：CUDA 基础、GEMM 优化（你的 9 版 SGEMM 已覆盖大部分）。

### 核心知识点
- 🟢 GPU 存储层级：Register / Shared Memory(SMEM) / L1 / L2 / HBM(全局显存)，各级带宽与延迟数量级差异。
- 🟢 执行模型：SM、warp(32 线程)、SIMT、occupancy、warp scheduler、bank conflict、memory coalescing。
- 🟡 Tensor Core：MMA 指令演进 `mma` → `wgmma`(Hopper, warpgroup 异步) → `tcgen05`/Tensor Memory(Blackwell)；为什么 matmul 要走 tensor core 而 elementwise 走 CUDA core。
- 🔴 **Roofline 模型**：算术强度（arithmetic / operational intensity = FLOPs / bytes）、ridge point、compute-bound vs memory-bound 的判定。
- 🟡 数值格式全家桶及用途：FP32 / TF32 / FP16 / BF16（指数位多、训练友好）/ FP8(E4M3 推理 / E5M2 范围大) / INT8 / INT4 / **FP4(E2M1)**；理解"指数位 vs 尾数位"决定动态范围 vs 精度。
- 🟡 Transformer decoder 结构件：MHA、FFN/MLP（gate+up+down，SwiGLU）、RMSNorm vs LayerNorm、residual、RoPE 位置编码、KV cache 的物理含义。

### 能力要求
- 🔴 **能徒手对一个具体模型（如 Llama-7B）分别算出 prefill 和 decode 的 FLOPs、HBM 访存字节、算术强度，并判定卡算力还是卡带宽。** 这是整条路线的"入场券"，必须能在白板上做。
- 🟡 能解释为什么 BF16 在训练取代 FP16（动态范围）、为什么 E4M3 用于推理权重/激活。

### 自测 / 面试题
- A100 显存带宽约 2TB/s、BF16 算力约 312 TFLOPS，ridge point 的算术强度是多少？一个算术强度=2 的 kernel 卡在哪？
- 为什么 decode 单步算术强度极低？把它写成 (FLOPs)/(权重字节+KV字节) 推一遍。

### 资料关键词
`GPU roofline model`、`arithmetic intensity`、`memory coalescing bank conflict`、`Hopper wgmma`、`Blackwell tcgen05 tensor memory`、`FP8 E4M3 E5M2`、`Llama architecture RMSNorm SwiGLU RoPE`、NVIDIA CUDA C++ Programming Guide。

---

## M1 — 推理工作负载特性与指标体系

**依赖**：M0（roofline）。

### 核心知识点
- 🔴 **Prefill vs Decode 二分**（全路线最重要的一句话）：
  - Prefill：一次处理整个 prompt，大矩阵 GEMM，**compute-bound**，决定 **TTFT**。
  - Decode：逐 token 自回归，每步把 **整模型权重 + 全部 KV** 从 HBM 搬一遍只算 1 个 token，**memory-bound**，决定 **TPOT/ITL**。
- 🟡 为什么 batching 对 decode 近乎"免费提吞吐"：权重搬运成本被 batch 内多请求摊薄，算术强度随 batch 增大而上升，直到把 decode 推成 compute-bound 为止（ridge point）。
- 🔴 指标体系：**TTFT**（首 token 延迟）、**TPOT/ITL**（每 token / token 间延迟）、**throughput**（tokens/s）、**goodput**（满足 SLO 约束下的有效吞吐，2024+ 高频考点）、QPS、SLO/SLA。
- 🟡 延迟 vs 吞吐的根本张力：大 batch 提吞吐但伤单请求延迟；这是后面所有调度策略的优化目标。

### 能力要求
- 🟡 **能用 roofline 论证"batch 从 1 加到 8，decode 吞吐近线性涨而单请求延迟几乎不变"。** 讲清这个，调度和 KV 两块就有了根。
- 🟢 能说出每个优化技术主要打哪个指标（如 chunked prefill 打 TPOT 抖动、prefix cache 打 TTFT）。

### 自测 / 面试题
- 为什么 throughput 高不等于 goodput 高？举一个吞吐高但违反 SLO 的场景。
- prefill 和 decode 放在同一块卡同一步里跑，会互相怎么干扰？（引出 M5 的 chunked prefill 与 P/D 分离）

### 资料关键词
`LLM inference prefill decode`、`TTFT TPOT ITL goodput`、`memory-bound decode batching`、`roofline LLM serving`。

---

## M2 — Attention 数学 + FlashAttention 家族【算子核心 · 必达 L3】

**依赖**：M0、M1。**这是算子岗最高频的现场手写/优化题。**

### 核心知识点
- 🟡 Attention 数学回顾：Q/K/V、scaled dot-product、causal mask、softmax 数值稳定（减最大值）、N×N 中间矩阵为什么是访存大头。
- 🔴 **Online / Streaming Softmax**：用 running max + running sum 流式归一化，不物化整个 attention 矩阵。**能独立推导这个递推公式是硬通货。**
- 🔴 **FlashAttention v1**：IO-aware、tiling（Q/K/V 分块进 SMEM）、用重计算换显存（backward 不存中间矩阵）、复杂度从 O(N²) 显存降到 O(N)。
- 🟡 **v2**：减少非 matmul FLOPs、改进 warp 间 work partitioning、减少 SMEM 读写、支持更大 head_dim。
- 🟡 **v3**（Hopper）：warp specialization（生产者/消费者）、TMA 异步搬运、wgmma 异步 matmul、与 softmax overlap、FP8 支持。
- 🟡 **v4**（Blackwell，2026/03 论文）：CuTe-DSL 实现；**新 online softmax——仅当新行最大值显著大于旧值才做 rescale，跳过约 90% 输出 rescaling**；用 FMA 多项式近似软件实现 exp（绕开 SFU 瓶颈）；针对 tcgen05/Tensor Memory 的异步 MMA。了解"为什么 Blackwell 要重写"= asymmetric hardware scaling。
- 🔴 **FlashDecoding / FlashDecoding++**：decode 阶段 query 长度=1，batch×heads 喂不满 SM，必须沿 **KV 序列维做 split-K（split-KV）** 并行再 reduce。**很多人只懂 prefill 的 flash、不懂 decode 为什么换并行策略——这是区分度所在。**

### 能力要求
- 🔴 **用 Triton 写一个 FlashAttention forward**（causal + 非 causal）。直接对应你 Tinygrad 能力，是现场最可能让你写的 kernel。
- 🟡 **能在白板推 online softmax 递推**，并讲清 v1→v2→v3→v4 每代到底改了什么、为什么。
- 🟡 能解释 prefill flash 与 decode flash（FlashDecoding）并行维度的差异。

### 自测 / 面试题
- 推导：分块计算 softmax 时，新块到来如何修正之前块的 running sum / 输出累加？
- v2 相对 v1 具体改了哪几点？为什么 "减少非 matmul FLOPs" 在 tensor core 上收益大？
- decode 时 batch=1、heads=32、head_dim=128，为什么直接用 prefill 的 flash kernel 会浪费 GPU？怎么改？

### 资料关键词
`online softmax derivation`、`FlashAttention v1 v2 v3 paper Tri Dao`、`FlashAttention-4 CuTe Blackwell`、`FlashDecoding split-KV`、`Triton flash attention tutorial`、Triton 官方 tutorials（fused-softmax → matmul → flash-attention）。

---

## M3 — Attention 变体与 KV Cache 结构（MHA / MQA / GQA / MLA）

**依赖**：M2。**这是 2024-2025 新增的高频考点，你之前的清单没覆盖，务必补。**

### 核心知识点
- 🔴 KV Cache 大小公式：`2 (K和V) × num_layers × num_kv_heads × head_dim × seq_len × batch × dtype_bytes`。理解它如何决定可开 batch / 可支持上下文长度。
- 🔴 **KV 头数压缩演进路线**（核心矛盾：KV cache 太大）：
  - **MHA**：每个 query head 配独立 K/V head，KV 最大。
  - **MQA**（Multi-Query）：所有 query head 共享 1 组 K/V，KV 缩 num_heads 倍，但质量略降。
  - **GQA**（Grouped-Query）：分组共享，介于两者之间；**Llama-2/3、Qwen 等主流模型采用**。
  - **MLA**（Multi-head Latent Attention，DeepSeek-V2/V3）：把 K/V 低秩压缩到一个小 **latent 向量**缓存，用时上投影还原；KV cache 显著更小。难点：RoPE 与低秩压缩不兼容 → **解耦 RoPE（decoupled RoPE，拆出单独的位置编码维度）**。
- 🟡 各变体对 attention kernel 的影响（如 GQA 需要 kernel 支持 KV head 广播到多个 Q head）。

### 能力要求
- 🟡 **能给定模型配置算各变体的 KV cache 大小**，并解释为什么 MQA/GQA/MLA 能开更大 batch。
- 🟡 **能讲清 MLA 怎么省 KV、又怎么解决 RoPE 兼容问题**——这是 DeepSeek 相关面试的必问点。

### 自测 / 面试题
- 同样 70B 模型，MHA 改成 GQA(8 组)，KV cache 缩多少倍？对最大 batch 的影响？
- MLA 的 latent 压缩为什么不能直接套 RoPE？decoupled RoPE 怎么解决？

### 资料关键词
`KV cache size formula`、`MQA multi-query attention`、`GQA grouped-query attention paper`、`DeepSeek MLA multi-head latent attention`、`decoupled RoPE`、DeepSeek-V2 / V3 技术报告。

---

## M4 — KV Cache 内存管理【你的第 2 块】

**依赖**：M3（KV 多大）、M2（attention kernel 如何从非连续内存取数）。

### 核心知识点（按粒度递进，核心矛盾=**碎片化 vs 管理开销**）
- 🟡 **连续分配（HF transformers）**：按 max_seq_len 预留连续显存 → 严重**内部碎片**（占了没用满），且无法跨请求共享。单请求快、多请求差。
- 🔴 **PagedAttention（vLLM, SOSP'23）**：照搬 OS 虚拟内存分页。
  - KV 切成固定 **block**（默认 16 token），**block table** 做逻辑块→物理块映射，碎片只剩每个序列最后一块的零头。
  - **Copy-on-Write**：并行采样 / beam search 共享前缀 block，写时复制。
  - attention kernel 必须改成**从非连续 block gather**（这就是为什么 M4 依赖 M2）。
  - **block_size 权衡**（高频追问）：block 小 → 省显存但 block table 大、kernel 寻址开销升；block 大 → 反之。
- 🟡 **前缀复用层**：
  - **RadixAttention（SGLang）**：基数树（radix tree）管理 KV，LRU 驱逐，**cache-aware scheduling**（把共享前缀的请求路由到一起）。chat / few-shot 场景吞吐再升。
  - **Automatic Prefix Caching, APC（vLLM）**：基于 block hash 复用，目标相同、实现不同。**能对比 Radix tree vs block-hash 两种实现是加分项。**
- 🟡 **Token Attention（LMDeploy）**：本质 block_size=1 的极细粒度分页，碎片最小、bookkeeping 开销最大——又一个 trade-off。
- 🟡 **KV Cache 量化**（INT8 / FP8 KV）：KV 是 decode 显存大头，量化它直接扩 batch / 上下文，2024-25 高频。
- 🟢 **KV offloading / 分层缓存**：热 KV 在 HBM、冷 KV 卸到 CPU/NVMe；与前缀缓存、P/D 分离结合（如 Mooncake 的全局 KV pool）。

### 能力要求
- 🔴 **实现一个玩具版 paged KV + 改 attention kernel 从非连续 block 取数**（接到你 tinygrad 线，极强的简历项）。
- 🟡 能算碎片率、能对比 Radix vs APC、能讲 block_size 取舍。

### 自测 / 面试题
- PagedAttention 把内部碎片降到多少？为什么仍有零头？block_size=16 vs 1 各自代价？
- 两个请求共享 100 token 系统 prompt，RadixAttention 和 APC 分别怎么复用？驱逐策略？
- 为什么 paged 之后 attention kernel 不能再假设 KV 连续？kernel 要改什么？

### 资料关键词
`PagedAttention vLLM paper`、`block table copy-on-write`、`RadixAttention SGLang`、`vLLM automatic prefix caching block hash`、`LMDeploy TokenAttention`、`KV cache quantization INT8 FP8`、`Mooncake KV cache pool`、vLLM `block_manager` / paged attention kernel 源码。

---

## M5 — 调度粒度【你的第 1 块】

**依赖**：M1（二分+指标）、M4（KV 管理是调度的物理约束）。**当前面试重心已从"continuous batching"后移到"P/D 分离 + chunked prefill + 投机解码"。**

### 核心知识点
- 🟡 **Request-level（旧）→ Iteration-level**：源头是 **Orca（OSDI'22）**，提出 iteration-level scheduling + **selective batching**。问题不止 padding 浪费，更关键是**队头阻塞**（短请求被长请求拖着等整批结束）。
- 🔴 **Continuous Batching（连续批处理）**：每个 decode step 动态把已完成请求踢出、新请求加进 batch。vLLM/SGLang 主流。理解它和 PagedAttention 的配合。
- 🔴 **Chunked Prefill（Sarathi-Serve）**：朴素 continuous batching 把长 prefill 和 decode 混在一步，长 prefill 一来占满算力、decode 的 TPOT 抖动。把 prefill 切块、与 decode "捎带"（piggyback）同跑，平滑延迟。打 **TPOT 抖动**。
- 🔴 **Prefill / Decode 分离（P/D 分离，当前最大热点）**：
  - 动机直接来自 M1 二分：prefill compute-bound 且 care TTFT，decode memory-bound 且 care TPOT，混在一起互相干扰。
  - 做法：拆到不同 GPU 池，各自独立扩缩、可异构硬件；**prefill 算完的 KV cache 需跨节点传到 decode 池**（KV transfer 是工程难点，走 NVLink/RDMA/NCCL）。
  - 代表：**DistServe、Splitwise、Mooncake（以 KV cache 为中心）**；**已成行业标准**——vLLM/SGLang/TRT-LLM/LMDeploy 全支持，**NVIDIA Dynamo** 作为其上的编排层（路由+自动扩缩+KV 传输协调），**DeepSeek 大规模生产落地**（如 DeepSeek-R1 大集群 P 池/D 池分离）。
  - 路由：P/D-aware load balancer、**consistent hashing 做前缀亲和**（同会话粘到持有其 KV 的 worker）。
- 🟡 **Speculative Decoding（投机解码）**：你说的 "token-level 决策工程化难"，现实落地形态就是它——小 **draft model** 一次猜多 token，大模型一次并行验证、接受/回退。变体：**Medusa**（多头并行预测）、**EAGLE**（特征级自回归 draft）、**lookahead decoding**、self-speculative。打 decode 的延迟，本质是把 memory-bound 的 decode 变得更"compute 密集"。
- 🟢 调度的其它维度：优先级/抢占（preemption + recompute vs swap）、公平性、SLO-aware 调度。

### 能力要求
- 🟡 **能画 P/D 分离架构图**，讲清动机、KV 怎么跨节点传、P 池和 D 池如何独立扩缩。
- 🟡 能讲清 chunked prefill 解决什么、speculative decoding 的验证机制与"为什么不掉精度"。

### 自测 / 面试题
- 不用 chunked prefill，长 prompt 进来时 decode 用户的 TPOT 为什么抖动？切块怎么平滑？
- P/D 分离相比合并部署，省了什么、又新增什么成本（KV 传输带宽/延迟）？什么负载下不值得分？
- 投机解码为什么不损失精度？draft 接受率低时为什么反而变慢？

### 资料关键词
`Orca iteration-level scheduling selective batching`、`continuous batching vLLM`、`Sarathi-Serve chunked prefill`、`DistServe Splitwise Mooncake prefill decode disaggregation`、`NVIDIA Dynamo disaggregated serving`、`speculative decoding Medusa EAGLE`、`KV cache transfer NIXL RDMA`、vLLM scheduler 源码。

---

## M6 — Kernel 实现路线【你的第 3 块 · 算子岗必达 L3】

**依赖**：M0、M2。按一条轴理解：**控制力/性能上限 ↔ 开发效率/可维护性**。

### 核心知识点
- 🟡 **手写 CUDA**：极致控制、性能上限最高，但开发慢、维护贵。TRT-LLM、vLLM 早期关键 kernel。
- 🟡 **CUTLASS / CuTe**：模板化 GEMM 构件 + **layout/Tensor 抽象**（CuTe 的 `Layout`/`Tensor` 是理解现代 kernel 的关键）、**epilogue fusion**（GEMM 后直接融激活/bias/量化）。近峰值且可维护，**FlashInfer、FA4** 走这条。
- 🔴 **Triton（DSL）**：tile 级编程模型（你写 block 级逻辑、编译器管线程/SMEM/流水）、`autotune`、`tl.load/store/dot`。开发效率高，常达手写 80-90%。**OpenAI、字节火山等内部大量使用**——对你目标厂尤其值得 L3。
- 🟡 **编译器化**：TVM/Relax、**MLIR**、IREE、MLC-LLM、`torch.compile`/TorchInductor、TensorRT(engine 构建)。可移植/跨平台，但前沿优化天花板相对低。
- 🔴 **算子融合（fusion）**：为什么融合 = 减少 kernel launch 开销 + 减少 HBM 往返（把中间结果留在 SMEM/寄存器）。典型：bias+GELU、RMSNorm+matmul、RoPE 融进 attention、dequant 融进 GEMM。**这是算子岗日常工作本身。**

### 能力要求
- 🔴 **用 Triton 写并 autotune 一个 fused kernel**（如 fused RMSNorm、或 fused dequant-GEMM）。
- 🟡 **能就一个具体场景论证选 Triton 还是手写 CUDA 还是 CUTLASS**（答案就在那条轴上）。
- 🟡 能解释 CuTe 的 layout 抽象解决了什么、epilogue fusion 是什么。

### 自测 / 面试题
- 为什么 elementwise 算子（如 add+gelu）一定要融合？不融合多花在哪？
- Triton 相比手写 CUDA，你放弃了什么控制权、换来了什么？什么情况下这笔交易不划算？
- CUTLASS 的 epilogue 能做什么？为什么 FA4 选 CuTe 而不是纯手写 PTX？

### 资料关键词
`Triton language tutorial autotune`、`CUTLASS CuTe layout tensor`、`epilogue fusion`、`FlashInfer`、`kernel fusion reduce HBM traffic`、`torch.compile inductor`、`TVM MLIR IREE`、`TensorRT-LLM`、OpenAI Triton 官方文档与 GitHub。

> **Triton L3 已落地为完整教程：[triton/00_README.md](triton/00_README.md)（6 篇，含 12 道带验收的练习）；
> 生态术语全景（Triton/TVM/TensorRT/DeepSpeed/K8s/LLVM 的分层定位）见 [19_ai_infra_ecosystem.md](19_ai_infra_ecosystem.md)。**

---

## M7 — 量化策略【你的第 4 块】

**依赖**：M0（数值格式）、M6（反量化 kernel 怎么落地）。

### 核心知识点
- 🟡 **量化基础**：对称/非对称、per-tensor / per-channel / **per-group(block)**、静态 vs 动态、校准(calibration)、PTQ vs QAT、**W?A?** 记法（W=权重位宽，A=激活位宽）。
- 🔴 **W4A16（GPTQ / AWQ）—— 主流的根因**：decode 是 memory-bound，权重压到 4bit 把**访存砍约 4×**，激活留 16bit，kernel 边算边**反量化（dequant-on-the-fly）**。
  - **GPTQ**：基于二阶信息（Hessian / OBQ 思路）逐层做误差补偿量化。
  - **AWQ**：activation-aware——用激活幅度找出"显著权重通道"，靠 per-channel scaling 保护它们；低 bit 下常优于 GPTQ 且更简单。
- 🔴 **W8A8（SmoothQuant）—— 纠正你原清单的"精度损失大"**：SmoothQuant 的卖点恰恰是**低精度损失**。它把激活里难量化的 **outlier 按通道平滑迁移到权重上**，使 W8A8 可行。它和 W4A16 的真正区别不是"损失大小"，而是 **目标阶段不同**：W8A8 走 **INT8 tensor core** 给 **compute-bound/prefill** 提速；W4A16 主要救 **memory-bound/decode**。
- 🟡 **FP8（H100+）**：E4M3(推理常用) / E5M2(范围大)；per-tensor scaling、**delayed scaling**；硬件原生 matmul，动态范围比 INT8 友好；可配 **FP8 KV cache**。TRT-LLM 主推。
- 🟡 **FP4 微缩放（Blackwell, 2025-2026 前沿）**：**MXFP4**（block=32，2 的幂 scale）与 **NVFP4**（block=16，**两级 scaling**：per-block E4M3 + per-tensor FP32），元素是 E2M1。瞄准 **W4A4**（连激活也 4bit）以吃满 tensor core 吞吐。坑：小 group 会**抵消传统 outlier 缓解手段**；需要 ModelOpt/MR-GPTQ 等专门算法。仅 Blackwell（B200/B300/RTX 5090/PRO 6000）原生支持；TRT-LLM 最成熟，vLLM/SGLang 已支持。
- 🟢 极端/边缘：GGUF(llama.cpp) 各档量化、INT4 KV、BitNet(1-bit) 了解即可。
- 🔴 **反量化在 kernel 里怎么落地**：dequant 与 GEMM 融合（避免先反量化整张权重写回 HBM），这正是 M6 的 fusion。

### 能力要求
- 🟡 **能解释 GPTQ vs AWQ 的机理差异**、**论证 W4A16 vs W8A8 分别适用哪个阶段**、**FP8 vs INT8 的取舍**。
- 🟡 能讲 NVFP4 的两级 scaling 为什么需要、解决什么动态范围问题。

### 自测 / 面试题
- 为什么 decode 场景普遍 W4A16 而不是 W8A8？反过来 prefill 重的离线批处理为什么可能选 W8A8？
- SmoothQuant 的"平滑"在数学上做了什么？为什么把难量化迁到权重就 OK？
- FP8 相比 INT8 在量化上最大的好处是什么？（动态范围/无需复杂校准）
- NVFP4 为什么要 per-block + per-tensor 两级 scale？只用一级会怎样？

### 资料关键词
`GPTQ paper`、`AWQ activation-aware weight quantization`、`SmoothQuant outlier migration`、`FP8 quantization E4M3 delayed scaling`、`NVFP4 MXFP4 microscaling Blackwell`、`NVIDIA ModelOpt`、`dequant GEMM fusion`、`llm-compressor / AutoGPTQ / AutoAWQ`。

---

## M8 — 分布式 / 并行（推理向 · 相邻必考）

**依赖**：M0、M1；与 M5（P/D 分离）强耦合。你原清单没列，但 Tier-1 推理岗常考。

### 核心知识点
- 🔴 **张量并行 TP（Megatron 式）**：attention 按 head 切、FFN 按列/行切；理解 **all-reduce 插在哪两处**、为什么 TP 适合卡内 NVLink。
- 🟡 **流水并行 PP**：micro-batch、bubble；推理里不如训练常用但要懂。
- 🟡 **专家并行 EP（MoE）**：expert 分到不同卡，**all-to-all** 通信；DeepSeek/Mixtral 类必备。
- 🟡 **序列/上下文并行 SP/CP**：长上下文场景沿 seq 维切。
- 🟡 **通信原语**：all-reduce / all-gather / reduce-scatter、NCCL、NVLink vs PCIe vs RDMA 带宽量级；**计算-通信 overlap**。

### 能力要求
- 🟡 能画 Megatron TP 对 attention + FFN 的切法，标出 all-reduce 位置，估算每层通信量。
- 🟢 能讲 P/D 分离里 KV 跨节点传输走什么链路、为什么 KV transfer 是瓶颈。

### 资料关键词
`Megatron-LM tensor parallelism`、`pipeline parallelism bubble`、`MoE expert parallel all-to-all`、`sequence parallelism context parallel`、`NCCL collective communication`、`compute communication overlap`。

---

## M9 — Profiling 与工程横切（越早建立越好）

**依赖**：能跑起来即可，建议在 M2 动手时就用上。

### 核心知识点
- 🔴 工具：**Nsight Systems**（timeline、kernel/通信重叠）、**Nsight Compute**（单 kernel 的 occupancy/访存吞吐/SM 利用率/roofline）、PyTorch profiler。
- 🟡 看什么：achieved occupancy、memory throughput vs compute throughput、kernel launch 间隙、是否 memory-bound。
- 🟡 **CUDA Graph**：消除大量小 kernel 的 launch 开销（decode 每步几十个小 kernel，launch 开销占比高）。
- 🟢 benchmark 方法论：warmup、锁频、固定 batch/seq、区分 TTFT/TPOT 测量、避免被 CPU 调度噪声污染。

### 能力要求
- 🔴 **能用 Nsight 定位一个 kernel 是 memory-bound 还是 launch-bound，并说出优化方向。** 面试常问"你怎么知道瓶颈在哪"。

### 资料关键词
`Nsight Systems Nsight Compute tutorial`、`CUDA graph reduce launch overhead`、`pytorch profiler`、`kernel occupancy memory throughput`。

---

## M10 — 前沿雷达（2025-2026，横切 · 面试聊"最近在关注什么"）

**依赖**：M0-M9 全部（建议最后学）。详见 **[18_frontier_2025_2026.md](18_frontier_2025_2026.md)**。

- 🟢 工作负载范式：reasoning/test-time scaling（decode 占比暴增）、agentic 负载（prefix cache 一级指标）、RL rollout（推理引擎成训练部件）。
- 🟡 架构新方向：细粒度 MoE 标配（grouped GEMM）、可训练稀疏 attention（**NSA/MoBA/DSA**）、线性/混合 attention（Qwen3-Next、Kimi Linear——打破 paged KV 假设）、MTP 即原生投机解码。
- 🟡 系统格局：P/D 分离 + 大 EP 成旗舰标配（DeepSeek 全家桶：FlashMLA/DeepEP/DeepGEMM/EPLB）、NVIDIA Dynamo、KV cache 分层/全局化（Mooncake/LMCache/NIXL）、vLLM V1。
- 🟡 硬件与精度：Blackwell tcgen05/TMEM、GB200 NVL72（72 卡 NVLink 域）、**不对称缩放**论点、FP8 训练（DeepSeek-V3）→ FP4 推理（NVFP4/MXFP4）。
- 🟢 Kernel 范式：CuTe-DSL / Gluon / TileLang / ThunderKittens / Helion 的共同主线；AI 写 kernel（KernelBench）。

### 资料关键词
`Native Sparse Attention NSA`、`MoBA block attention`、`DeepSeek-V3.2 DSA`、`Gated DeltaNet hybrid`、`DeepEP EPLB FlashMLA`、`NVIDIA Dynamo NIXL`、`Mooncake LMCache`、`CuTe DSL Gluon TileLang`、`NVFP4 W4A4`、`multi-token prediction speculative`。

---

## 动手项目清单（直接挂到你的 slimyo/tinygrad-notebook）

> **已落地为可执行项目计划：见 [projects/P0_overview.md](projects/P0_overview.md)（6-8 周，
> 含 RTX 2060/SM75 环境约束、每周任务、ncu 验收标准）。下面 1/2/3/5 分别对应
> P3 / P4 / P5 / P4，并新增了 P1（Profiling 训练馆）和 P2（HGEMM+CUTLASS）两个地基项目。**

按"投入产出比 + 简历叙事强度"排序：

1. 🔴 **Triton FlashAttention forward**（M2）——现场最可能考，先做。
2. 🔴 **玩具版 Paged KV + 改 attention kernel 从非连续 block 取数**（M4）——能讲"我自己实现过 PagedAttention 的取数路径"，远胜复述 vLLM。
3. 🔴 **Fused dequant-GEMM（W4A16）Triton kernel**（M6+M7）——把量化与 kernel 两块一次性打通。
4. 🟡 **极简 continuous batching 调度器 + chunked prefill 模拟**（M5）——不必真跑大模型，用 mock 时延模型展示 TPOT 平滑效果即可。
5. 🟡 **FlashDecoding（split-KV）vs 朴素 decode attention 的 benchmark**（M2）——配 Nsight 截图，叙事完整。

> 每个项目都配一段 README：动机（用 roofline/二分讲清为什么）+ 实现要点 + Nsight profiling 结果 + trade-off 讨论。**面试官要的就是这套"我懂原理 + 我动过手 + 我会测"的闭环。**

---

## 论文 / 源码精读清单（按模块）

- **M2**：FlashAttention v1/v2/v3 论文 + FA4 论文(2026)；FlashDecoding 博客；Triton 官方 tutorials。
- **M3**：MQA、GQA 论文；DeepSeek-V2/V3 技术报告（MLA + decoupled RoPE）。
- **M4**：PagedAttention(vLLM, SOSP'23)；SGLang/RadixAttention；vLLM APC 文档与源码。
- **M5**：Orca(OSDI'22)；Sarathi-Serve(chunked prefill)；DistServe、Splitwise、Mooncake；NVIDIA Dynamo 文档；Speculative Decoding / Medusa / EAGLE。
- **M6**：CUTLASS/CuTe 文档；FlashInfer；Triton 论文。
- **M7**：GPTQ、AWQ、SmoothQuant、FP8(LLM.int8 背景)、NVFP4/MXFP4(Blackwell 白皮书 + ModelOpt)。
- **M8**：Megatron-LM 论文。
- **源码**：vLLM（scheduler + block_manager + paged attention kernel）、SGLang（radix tree + scheduler）、FlashInfer（CUTLASS kernel 风格）。

---

## 面试白板高频题（自检通过即达标）

1. 只用 roofline 论证：decode 为什么 memory-bound？batch 增大吞吐为何近线性涨、延迟为何几乎不变？（M0/M1）
2. 推导 online softmax；讲 FlashAttention v1→v4 每代改了什么。（M2）
3. decode attention 为什么要 FlashDecoding 的 split-KV？（M2）
4. GQA/MLA 各省多少 KV？MLA 怎么解决 RoPE 兼容？（M3）
5. PagedAttention 解决什么碎片？block_size 怎么取舍？paged 后 kernel 要改什么？（M4）
6. chunked prefill 解决什么？P/D 分离的动机、KV 怎么传、何时不值得分？（M5）
7. 某 attention/GEMM 场景，选 Triton 还是手写 CUDA 还是 CUTLASS，为什么？（M6）
8. W4A16 vs W8A8 分别打哪个阶段？GPTQ vs AWQ 机理？FP8 vs INT8 取舍？NVFP4 两级 scale 为什么？（M7）
9. Megatron TP 怎么切 attention+FFN，all-reduce 在哪？（M8）
10. 给一个慢的 kernel，你怎么用 Nsight 判断瓶颈、定优化方向？（M9）

---

## 一句话学习顺序建议

**M0/M1 打地基（1 周）→ M2 死磕到 L3 + 第一个 Triton flash 项目（2-3 周）→ M3→M4 串起 KV 主线 + paged KV 项目（2 周）→ M5 调度（含 P/D 分离重点，1.5 周）→ M6/M7 交织学 + fused dequant-GEMM 项目（2-3 周）→ M8/M9 补全 + 全程用 Nsight（穿插）。** 你列的四块分别落在 M5 / M4 / M6 / M7，全部建立在前面的 M0-M3 之上。