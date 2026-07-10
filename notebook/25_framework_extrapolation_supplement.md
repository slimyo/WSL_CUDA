# 面试题补充二：推理框架选型 / 长上下文外推 / 综合系统

> 对象: 冲刺 Tier-1 推理/算子岗
> 前置: 已完成 06-24 章，此文件补充剩余面试缺口
> 目标: 覆盖"推理框架对比、位置编码外推方法、综合优化清单、国产硬件"等此前未涉及的高频题

---

## 1. 位置编码与长上下文外推方法对比

### 1.1 位置编码基础

```
RoPE (Rotary Position Embedding)：
  在 Q/K 向量上施加旋转矩阵，使点积结果编码相对位置信息。
  旋转角度 = pos * theta_i, theta_i = 10000^(-2i/d)
  特性：相对位置编码、可外推至训练长度以上的序列。

ALiBi (Attention with Linear Biases)：
  不在 Q/K 上加位置编码，而是在 attention score 上加线性偏置：
    score(i,j) = Q_i * K_j / sqrt(d) - m * |i-j|
  m 是每 head 不同的斜率。
  特性：训练时无需位置 embedding、天然支持外推（BLOOM 等模型使用）。
```

### 1.2 长上下文外推方法

```
(1) PI (Position Interpolation, 2023)
    将长序列的位置索引压缩到训练范围：
      pos' = pos * (L_train / L_infer)
    等价于降低旋转频率（所有 theta_i 缩小 L_train/L_infer 倍）
    优点：实现简单，只需修改 cos/sin 表
    缺点：高频信息被压缩，短距离分辨率下降
    典型使用：Llama 2 从 4K -> 32K

(2) NTK-Aware Scaled RoPE (2023)
    只修改高频 theta（接近训练范围的频率），保留低频不压缩：
      theta_i' = theta_i * s^(-2i/(d-2))
    其中 s = (L_infer / L_train) > 1
    高频(theta 大)被拉升(频率加快)，低频(theta 小)几乎不变
    优点：保留局部分辨能力（高频控制短距离）
    缺点：实现比 PI 复杂，需要确定 scaling factor

(3) YaRN (Yet another RoPE extensioN method, 2023)
    结合 PI 和 NTK，分两路处理：
    - 高频 theta -> NTK-style scaling
    - 低频 theta -> PI-style interpolation
    加上一个 ramp 函数在 frequency domain 上平滑过渡：
      theta_i' = (1 - w_i) * theta_i / s + w_i * theta_i * s
      w_i 从高频到低频从 1 到 0 线性递减
    优点：综合 PI 和 NTK 优势，实践效果最好
    缺点：调参较复杂（ramp width, scale factor）

(4) 对比总结

| 方法 | 原理 | 实现复杂度 | 长上下文质量 | 典型适用 |
|------|------|:--------:|:----------:|:-------:|
| PI | 线性压缩位置 | 简单 | 中（短距精度下降） | Llama 2 Chat |
| NTK-Aware | 高频拉升+低频压缩 | 中 | 较好 | Code Llama |
| YaRN | PI+NTK 混合加权 | 复杂 | 最好 | Llama 3 / Mistral |
| ALiBi | 线性偏置 | 极简 | 好（但需训练时用） | BLOOM / MPT |

面试核心要点：
  RoPE 的外推方法本质是"频率调制"——高频负责局部精度，低频负责全局范围。
  外推质量取决于高/低频的平衡。
  越复杂的方法（YaRN）通常效果越好，但推理时需要 precompute cos/sin cache。
```

---

## 2. 主流推理框架对比

### 2.1 四大框架一表通

```
| 维度 | vLLM | TensorRT-LLM | SGLang | TGI |
|------|------|-------------|--------|-----|
| 开发者 | UC Berkeley | NVIDIA | Stanford | HuggingFace |
| 开源时间 | 2023.06 | 2023.10 | 2024.01 | 2023.08 |
| 核心调度 | Continuous Batching | In-flight Batching | RadixAttention | Simple Batching |
| KV Cache 管理 | PagedAttention (block=16) | 自定义 block (优化 block shape) | RadixTree (共享前缀) | 连续分配 |
| Prefix Caching | APC (hash-based, block 粒度) | 不支持原生 prefix cache | RadixAttention (trie) | 不支持 |
| 量化支持 | AWQ/GPTQ/FP8 | INT4/INT8/FP8/FBGEMM | AWQ/FP8 | GPTQ/AWQ |
| 投机解码 | 支持 (draft model) | 支持 (Medusa/Lookahead) | 支持 | 不支持 |
| P/D 分离 | 支持 (v1 引擎) | 通过 Triton 推理服务器 | 原生支持 | 不支持 |
| 分布式部署 | 多节点 (Ray) | 多节点 (NCCL + MPI) | 多节点 | 单机优先 |
| 生态绑定 | HuggingFace Transformers | TensorRT / ONNX | 自有前端 (Python) | HuggingFace 原生 |

性能倾向：
  vLLM: 吞吐优先，社区最活跃，通用场景首选
  TRT-LLM: 极端优化（内核级），延迟敏感场景（LLM 服务）首选
  SGLang: 前缀复用场景（Agent/Chat）压倒性优势，复杂的结构化 prompt
  TGI: 启动快配置简单，适合快速原型
```

### 2.2 选型建议

```
根据场景：

(1) 通用在线推理（大并发，多样请求）
  -> vLLM：社区大、生态好，custom model 适配快
  优点：支持 HuggingFace 模型最全，quant config 简单
  缺点：极端延迟优化不如 TRT-LLM

(2) 延迟敏感（智能客服、Copilot，要求 P99 TPOT < 30ms）
  -> TensorRT-LLM：极致延迟优化，kernel 级定制
  优点：对 H100/Blackwell 硬件利用最充分，FP8/int4 最优
  缺点：模型转换复杂，不支持所有 huggingface 模型格式

(3) 高前缀复用（Agent、Multi-turn Chat、RAG）
  -> SGLang：RadixAttention 带来的前缀复用优势
  优点：同样 system prompt 下 prefill 计算量减 80%+
  缺点：调度策略复杂，定制 kernel 不如 TRT-LLM 彻底

(4) 快速原型/小规模部署
  -> TGI：最简配置，一行指令启动
  优点：零配置上手
  缺点：大规模下吞吐和延迟都不如前三者
```

---

## 3. 为什么 LLM 推理不用 Beam Search

```
Beam Search 在 NLP 时代是 Seq2Seq 模型的标准解码方法（机器翻译、摘要），
在 LLM 推理中基本退役，原因：

(1) 开放生成的坏特性
  LLM 生成的是多样化的开放文本，Beam Search 寻找"全局最优"序列，
  但语言生成中"最优"不等同于"最自然"——Beam Search 倾向短序列、
  重复片段、事实错误的文本（因为高概率路径往往是"平庸"的）。

(2) KV Cache fork 成本
  Beam Search 需要维护 B 条路径，每条路径独立维护 KV cache。
  显存占用 = B * 正常显存（decode 的 KV 占大头）。
  即使有 Copy-on-Write（PagedAttention）减少重复前缀存储，
  分支后每条路径的 KV 是独立的，依然线性增大。

(3) 延迟 vs 收益不成比例
  同等计算量下，Beam Search (B=4) 的耗时为贪心解码的 4x（串行执行）
  或 4x 显存开销（并行执行）。
  Top-k/P sampling + Speculative Decoding 组合通常能达到更好的效果/延迟比。

(4) 场景例外
  - 代码生成（如代码补全）有时用 Beam Search 取 top-5 候选
  - 翻译任务中 Beam Search 仍有效（确定性任务）
  - 评分/排序场景（如 Reward Model 中找最佳候选）

结论：LLM 推理默认贪心解码或 Top-p/Top-k 采样，Beam Search 仅限特殊场景。
```

---

## 4. PTQ vs QAT 详细对比与选择

### 4.1 概念对比

```
| 维度 | PTQ (Post-Training Quantization) | QAT (Quantization-Aware Training) |
|------|:-------------------------------:|:---------------------------------:|
| 所需数据 | 少量校准集 (128-1024 samples) | 完整训练数据 + 训练流程 |
| 耗时 | 几十分钟到几小时 | 几周到数月（需完整训练/微调） |
| 精度损失 | 低 bit 损失大 (INT4 可能有 1-5% 任务精度下降) | 基本无损（模拟量化训练恢复精度） |
| 适用阶段 | 推理部署快速落地 | 需要极致低 bit 精度时 |
| 典型方法 | GPTQ / AWQ / SmoothQuant / FP8 PTQ | QAT for INT4 / QAT for FP8 |
| 工具 | AutoGPTQ / AutoAWQ / NVIDIA ModelOpt | NVIDIA TAO / PyTorch FX / TensorRT QAT |
| 量化算子 | 需要层上做 calibration | 训练时插入 fake quantize 节点 |

PTQ 关键选择因素：
  - Weight-only: W4A16 (GPTQ/AWQ) -> 精度好，显存降 4x，主流
  - Weight+Activation: W8A8 -> 精度可接受，需 SmoothQuant 处理 outlier
  - FP8: 硬件原生，精度好，H100+ 可选

QAT 的关键考量：
  - 训练时模拟量化噪声 -> 模型学到适应低精度
  - 对于 INT4 以下（INT3/INT2）和极低比特，QAT 通常是必要
  - DeepSeek-V3 的 FP8 训练本质上是 QAT（训练时就在 FP8 精度下）
```

---

## 5. 量化对 KV Cache 的影响

```
KV Cache 量化策略：

方案 A：FP8 KV Cache（H100+ 原生）
  每个 KV 元素从 FP16 (2 bytes) -> FP8 (1 byte)
  节省：KV cache 显存减半
  精度：E4M3 足够（KV 值的范围相对稳定，不像梯度有 outlier）
  收益：同等显存下 batch size 提升 2x，或 context length 翻倍
  H100 FP8 TC 可直接消费 FP8 KV -> matmul 加速

方案 B：INT8 KV Cache
  类似 weight quantization，需要 per-channel 或 per-token scaling
  精度：int8 range 127, 需要 calibration（类似 SmoothQuant）
  但 KV 的分布比 activation 更稳定（每个 token 的 KV 数值范围变化小）

方案 C：NVFP4 / MXFP4 KV Cache（Blackwell）
  NVFP4 每元素 4 bit + per-block E8M0 scale
  每个 block 32 元素: 32*4bit + 8bit = 136bit vs 512bit (FP16) -> 3.76x 压缩
  但需要 Blackwell 硬件支持 FP4 TC decode

重要工程考量（面试加分）：
  (1) K 和 V 分别的量化策略不同
      K 值通常范范更大（因为有 RoPE，随位置旋转，数值范围变化），
      V 值范围更稳定（与位置无关的 weighted sum）。
      所以有时对 K 和 V 用不同 quantization config（V 可更低 bit）。

  (2) 量化 KV Cache 在 decode 阶段 vs prefill 阶段
      Prefill: KV 大量生产，先量化再存 -> 节省显存
      Decode: 每次读已量化的 KV，参加 attention -> 需要 dequant
      Dequant 增加延迟（但 FP8 直接 TC matmul 不需 dequant）

  (3) KV Cache 量化的实际收益
      以 70B GQA (8 KV heads, 128 dim, 64 layers) 为例：
        FP16 KV per token = 2 * 64 * 8 * 128 * 2 = 0.26 MB
        32K context = 8.5 GB (FP16) -> 4.25 GB (FP8) -> 1.1 GB (FP4)
      对于 long context (>128K)，FP8/FP4 KV 是可行的前提条件。
```

---

## 6. 算子融合的收益与限制

### 6.1 收益（已知）

```
14 章已覆盖：
  - 减少 HBM 往返（每个 fused op 省 1 次 R/W -> wall clock 省 1-2x）
  - 减少 kernel launch overhead
  - 减少中间 tensor 显存占用
```

### 6.2 限制（面试要能说出"什么时候不该 fusion"）

```
(1) 融合后寄存器压力过大
    多个 op fused 在一起，中间结果全在 register/SMEM 中。
    若中间变量太多 -> 寄存器溢出 (register spill) -> 压到 local memory (L1/L2)
    -> 反而变慢。例：GELU + Matmul + Residual Add + LayerNorm 四合一可能溢出。

(2) 融合后 occupancy 下降
    每个 thread 负责的工作量变大（register 和 SMEM 用量增加）
    -> SM 能同时跑的 warp 数减少 -> latency hiding 能力下降
    -> 对 memory-bound 场景可能是致命的

(3) 融合 kernel 的可维护性
    一个 2000 行的 fused kernel 和 4 个 500 行的 unfused kernel
    测试、debug、调优难度完全不同。生产环境考虑工程可维护性。

(4) 融合边界
    哪些 op 适合融合？
      elementwise + elementwise: 几乎总是（如 silu + mul -> SwiGLU）
      reduce + elementwise: 可以（如 RMSNorm fused）
      GEMM + elementwise epilogue: 可以（CUTLASS epilogue fusion）
      GEMM + reduce / attention: 难（计算模式差异大，即 FlashAttention 已做到）

    不建议 fusion 的典型场景：
      - 两个 op 都是 compute-bound（融合不会提高 throughput）
      - 融合后 tensor 复用率不高（不会减少 HBM 访问）
      - 中间变量超大（溢出风险）

面试经典回答：fusion 是"用 register 换 HBM"，register 是 SM 最宝贵的资源，
  fusion 前要算 register budget，保证不溢出。
```

---

## 7. 端到端推理优化清单

### 7.1 按业务场景分类

```
智能客服 / Copilot（延迟敏感）：
  1. KV Cache: FP8 量化 + PagedAttention
  2. 调度: Continuous Batching + 动态 batch
  3. 量化: W4A16 (GPTQ/AWQ) decode 加速
  4. 投机采样 (Speculative Decoding) 减少步数
  5. 算子: fused kernel + CUDA Graph (减少 launch overhead)
  6. 部署: vLLM / TRT-LLM，PD 分离保持稳定性
  7. 监控: P50 TTFT < 300ms, P99 TPOT < 30ms

RAG / 长文档问答（Memory-bound）：
  1. Prefix Caching (SGLang RadixAttention) 共享 system prompt
  2. Long context: 滑动窗口 / StreamingLLM / AutoCompressors
  3. KV Cache: 压缩/eviction (H2O / SnapKV)
  4. Chunk Prefill 防止单次 prefill 卡死 decode
  5. 显存: KV cache offload to CPU (PCIe <-> HBM)

Agent / 多轮对话（长 prefix + 短 generation）：
  1. 强依赖 Prefix Caching（trie 匹配共享前缀）
  2. Agent session affinity（调度到已有 KV 的 GPU）
  3. KV Cache eviction (H2O: Heavy Hitter Oracle)
  4. Speculative Decoding（减少每步延迟）
  5. PD 分离: 只分离大 prefill，小 prefill 本地处理
```

### 7.2 量化策略选择树

```
Q: 给定模型和硬件，怎么选量化方案？

先问三个问题：
  1. 推理阶段: prefill (compute-bound) vs decode (memory-bound)？
  2. 部署硬件: A100 / H100 / B200 / 国产？
  3. 精度预算: 可接受多少 perplexity/task loss？

决策树：

  硬件是 H100+?
  ├── 是 -> prefill: FP8 W8A8 (原生 FP8 TC)
  │         decode: W4A16 (GPTQ/AWQ) 或 FP8 W8A8
  │         KV cache: FP8 量化
  ├── 否 (A100) ->
  │    prefill: INT8 W8A8 (SmoothQuant)
  │    decode: W4A16 (GPTQ/AWQ)
  │    KV cache: I8 量化 (per-channel)

  精度要求高（不能降 PPL > 0.5）?
  ├── 是 -> W4A16 (AWQ 精度更好) + FP8 KV (if H100)
  ├── 否 -> W4A4 (NVFP4/MXFP4 on Blackwell) 或 INT4 KV

  Agent 场景长 prefix?
  ├── 是 -> 先上 Prefix Caching，再量化
  ├── 否 -> 直接量化
```

---

## 8. 线上推理慢 / OOM 排查路径

### 8.1 推理慢排查

```
现象：P99 TPOT 从 25ms 飙到 80ms，用户反馈慢。

排查路径（从外到内）：

(1) 系统层面
  - 请求队列长度? (是否 burst 打满 GPU)
  - Batch 大小? (过大导致 batch 延迟上涨)
  - 是否发生了显存 OOM / swap? (nvidia-smi 看显存使用)
  - 网络: RDMA 是否降级 (链路丢包重传)?

(2) 调度层面
  - Continuous batching 调度是否健康? (新请求插入频率)
  - 是否有长 prefill 阻塞 decode? (Chunk Prefill 的 chunk 不合理)
  - P/D 分离: prefill KV 传输是否成为瓶颈?

(3) Kernel 层面
  - Nsight Systems timeline: 哪个 kernel 耗时最长?
  - Nsight Compute: 是 memory-bound 还是 compute-bound?
  - 是否存在线程束 stall? (stall reason: long scoreboard / not selected / etc.)
  - 是否发生 bank conflict / uncoalesced access?

(4) 模型层面
  - 量化: 是否有 dequant 瓶颈? FP8->FP16 转换?
  - Attention: 是否因为长 seq 导致 attention O(n^2) 计算?
  - MoE: A2A 通信是否成为瓶颈?

分层定位（面试能说出这条路径）：
  系统级 -> 调度级 -> Kernel 级 -> 模型级
```

### 8.2 OOM 排查

```
现象：请求返回 500 / OOM / GPU 显存耗尽

检查顺序：

(1) 显存分配
  - nvidia-smi 看每卡显存使用: 是否接近 80 GB (H100) 满?
  - 有显存泄漏? (每次请求后释放? torch.cuda.empty_cache())
  - 是权重还是 KV cache 占大头?

(2) KV Cache 问题（最常见）
  - Max sequence length 是否超出预期?
  - Prefix caching: radix tree 没有修剪? (LRU 失效)
  - PagedAttention: block table 没回收? (引用计数)
  - 显存碎片: 连续分配请求过大时分配失败?

(3) 重新估算
  - 用公式 recheck: weights + KV cache + activation 总显存
    权重大小 = model_params * dtype_size (70B fp16 = 140 GB 需要 TP 拆分)
    KV cache = 2 * layers * kv_heads * max_seq * head_dim * batch * dtype
  - 估算 batch 极限: (HBM - weights - overhead) / KV_per_seq / num_seq_per_gpu

(4) 解决方案
  - 减小 batch / max_seq / 开启 prefix caching
  - KV cache offload (CPU RAM) 或 quantization (FP8/INT4)
  - 水平扩展（加卡 TP/EP/DP）
```

---

## 9. 新硬件架构对推理的影响

### 9.1 NVIDIA 架构演进

```
| 架构 | 代表 GPU | 关键特性 | 对推理的影响 |
|------|---------|---------|------------|
| Volta (2017) | V100 | 首代 FP16 TC, 32 GB HBM | 混合精度训练起点 |
| Turing (2018) | T4 | INT8 TC, 16 GB, 低功耗 | 推理专用卡起点 |
| Ampere (2020) | A100 | TF32/BF16 TC, 80 GB HBM | LLM 训练的起点，仍广泛用于推理 |
| Hopper (2023) | H100 | FP8 TC, Transformer Engine, 3.35 TB/s | FP8 推理成为主流，KV cache 量化 |
| Blackwell (2025) | B200 | FP4 TC, NVLink 5, 192 GB | W4A4 推理，大模型推理成本降低 |
| Rubin (2026) | R100 (预计) | 下一代 | 更高带宽，更大 HBM，更先进的互联 |

关键推理代际变化：
  - V100 -> A100: HBM 从 32 GB -> 80 GB，让 70B 模型在单机上部署成为可能
  - A100 -> H100: 加入 FP8，KV cache 半量化，吞吐翻倍
  - H100 -> B200: FP4 TC + 更大显存，70B decode 延迟减半
```

### 9.2 推理专用 GPU / 国产 NPU

```
英伟达推理专用卡：
  T4 (16 GB, INT8 TC) -> L4 (24 GB, FP8) -> L40S (48 GB, FP8)
  特点：无 HBM（用 GDDR），功耗低，推理效率高
  适用：小模型、低并发场景

国产推理芯片（面试可能存在差异）：
  - 昇腾 (Ascend) 910B: 对标 A100, 支持 INT8/FP16, 通过 CANN 框架
  - 寒武纪 (Cambricon) MLU370: 推理卡, 支持 INT8/INT4
  - 曦望 (S3): 号称推理吞吐数倍于 T4/同等精度，24 GB GDDR6
  - 摩尔线程 (Moore Threads) MTT S4000: 通用 GPU，兼容 CUDA（PTX 翻译层）
  - 壁仞 (Biren Technology) BR100: 通用 GPU, 支持 FP32/FP16/INT8

国产卡的关键考量（面试加分）：
  - 软件生态: CUDA 的成熟度国内短时间难以追赶，算子库、量化工具、框架适配
  - 推理框架适配: vLLM/TensorRT-LLM 的国产卡移植工作量（通常需要做一层硬件抽象）
  - 互联: NVLink/NVSwitch 的替代方案（通常只有 PCIe，限制了 TP）
  - 精度: 国产卡在 FP8/INT4 的低 bit 精度可能不如 NVIDIA 稳定
  - 编译器: 自研编译器（如 CANN、TopsCompiler）的成熟度和 debug 能力

面试回答策略：
  不主动提国产卡细节（避免暴露不熟悉的领域），但如果被问到：
  承认差距（软件生态、互联），指出关键待解决点（算子适配、低 bit 精度），
  表示关注趋势（国产化替代是大势，但工程落地仍需时间）。
```

---

## 10. 推理服务化：QPS / P99 / 调度

### 10.1 关键指标

```
| 指标 | 定义 | 典型值 | 说明 |
|------|------|:-----:|------|
| QPS | Queries Per Second | 10-1000 | 系统每秒处理请求数 |
| TTFT | Time To First Token | 200-2000ms | 用户感知的"响应速度" |
| TPOT | Time Per Output Token | 15-60ms | 每秒生成 token 数 (1000/TPOT) |
| P50 / P99 | 中位数 / 99 百分位延迟 | P99 TPOT < 30ms | SLA 的关键衡量指标 |
| Goodput | 满足 SLO 的吞吐 | QPS * (1 - SLO_violation_rate) | 服务质量加权的真实吞吐 |

为什么 P99 比平均延迟重要：
  平均延迟可能很好但 1% 请求严重超时（tail latency）
  LLM 推理 P99 通常受 longest-prefill / longest-decode 请求影响
  优化 P99 需要关注长序列和 burst 调度
```

### 10.2 推理服务架构

```
请求进入 -> 调度器 -> 排队 -> 批处理 -> GPU 推理 -> 输出

(1) 调度策略
  - FIFO: 简单但公平性差（长请求堵住短请求）
  - SJF (Shortest Job First): 短请求优先，优 for P99，但长请求可能饿死
  - MLFQ (Multi-Level Feedback Queue): 动态优先级，长请求逐步升级优先级
  - 带超时机制的 SJF: 长请求超过等待时间后提升优先级

(2) 排队 + 并发
  - 单队列: 简单，前后依赖（head-of-line blocking）
  - 多队列: 不同 SLO 等级隔离（如 VIP 队列、普通队列）
  - 动态 batch: 等待 max_batch_size 或 max_batch_timeout (如 2ms) 后推理

(3) 负载均衡
  - Round-robin: 简单但不考虑当前负载
  - Least connections: 分配给最空闲的节点
  - KV-aware scheduling: 偏好调度到已有该请求 KV cache 的 GPU 节点
  - 一致性哈希: 保证相同前缀的请求路由到同一节点（提高 prefix cache 命中）

(4) QPS 估算
  最大 QPS = GPU 数 * 批量大小 / (每批量处理时间)
  例: 8 * H100, decode batch=64, 每处理时间=2s (假设 output=256 tokens)
    最大 QPS = 8 * 64 / 2 = 256 QPS
  实际 QPS 需考虑: prefill 占用的 GPU 时间、排队、网络、模型的大小等
```

---

## 11. Prefix Cache 命中率度量与优化

### 11.1 命中率度量

```
Prefix Cache 命中率 = 命中前缀长度 / 总输入长度

度量维度：
  (1) Token 命中率: 从 cache 中直接读取的 token 数 / 总 prompt token
      例: 请求 prompt 4096 tokens, cache 命中 3072 -> 命中率 75%
      直接反映 prefill 计算节省

  (2) Request 命中率: 有至少 1 个 cache hit 的请求 / 总请求
      反映 cache 对请求的影响面

  (3) Cache 命中延迟收益:
      cache hit 的 request TTFT = 未命中部分 prefill 时间 + decode 时间
      cache miss 的 request TTFT = 完整 prefill 时间 + decode 时间
      命中率提升 10% 可减少平均 TTFT 5-15%（看场景）

常见失效原因：
  - 请求前缀不同（即使相同 system prompt，假设微调版本不同也 miss）
  - Cache 被淘汰（LRU 策略，超过 max cache 大小）
  - 序列中有随机噪声（如 timestamp、random id 嵌入 prompt）
```

### 11.2 命中率优化

```
(1) Cache Key 设计
  SGLang RadixAttention: 用 input_ids + model_id + model_version 做 key
  更大的关键考虑：cache key 的 hash 计算不能太复杂（否则命中检测本身变成开销）

(2) 顺序固定
  对 system prompt 做 canonicalization: 移除 timestamp、request_id 等随机变量
  例如: 
    坏: "你是 xxx 客服，当前时间 2026-07-10 14:30:22，用户问..."
    好: "你是 xxx 客服，用户问..."
  通过去除噪声前缀使相同前缀命中

(3) 摘要压缩 (Prefix Cache 的辅助技术)
  如果 prompt 过长（> 64K），连 prefix cache 也存不下
  用摘要压缩（如 AutoCompressor / Gisting）把长前缀压缩成几个 summary tokens
  prefill 时只需计算 summary tokens 的 KV -> 大幅节省 prefill 时间

(4) 缓存分层
  第一层: 全局共享前缀（system prompt、few-shot examples）
  第二层: 会话级别（session 内的聊天历史）
  第一层在多个 GPU 之间同步（cross-instance cache sharing），第二层在单个 GPU 上
```

---

## 12. PTQ/QAT/量化全量对比表

```
| 方法 | 是否需要训练 | 校准数据 | 精度损失 | 适用 bit | 典型工具 |
|------|:----------:|:-------:|:-------:|:-------:|---------|
| GPTQ (PTQ) | 否 | 128-1024 samples | 低 (W4A16 几乎无损) | W4A16 | AutoGPTQ |
| AWQ (PTQ) | 否 | 128 samples | 低 (优于 GPTQ 在低 bit) | W4A16 | AutoAWQ |
| SmoothQuant (PTQ) | 否 | 512 samples | 低-mid | W8A8 | NVIDIA ModelOpt |
| FP8 PTQ | 否 | 512-1024 samples | 很低 (FP8 范围大) | FP8 | TRT-LLM / TE |
| QAT | 是 | 完整训练集 | 极低 (模拟量化恢复) | INT4/FP8 | NVIDIA TAO |
| QAT + Knowledge Distillation | 是 | 完整训练集+教师 | 基本无损 | INT3/INT2 | 定制流程 |

工程选型：
  7B-13B 模型: PTQ (AWQ/GPTQ) 即可，W4A16 decode 加速 3-4x
  70B+ 模型: PTQ (AWQ/FP8) 起步，若精度不满足则用 QAT
  FP8 训练: 本质是 QAT（Mixture of Experts 用 FP8 训练时自动适应）
```

---

## 13. 影响吞吐/延迟的关键因素综合表

```
| 因素 | 影响哪个指标 | 原理 | 优化手段 |
|------|:----------:|------|---------|
| Batch size | Throughput / TPOT | 大 batch 提高计算密度但增大单步延迟 | 动态/continuous batching |
| Sequence length | TTFT / KV cache | 长 seq 增大 prefill 计算量和 KV 存 | Prefix cache / Chunk prefill |
| Quantization (weight) | TPOT / 吞吐 | W4A16 访存 4x 减少 -> decode 加速 | AWQ / GPTQ |
| Quantization (KV cache) | Batch / 长 ctx | KV cache 减半 -> 可支持更大 batch 或更长的上下文 | FP8 KV / INT4 KV |
| Speculative Decoding | TPOT | 一步生成多个 token，减少 decode 步数 | Draft model / Medusa |
| FlashAttention | TTFT | 减少 prefill 的 HBM 访问 | FA v2/v3 |
| Operator Fusion | TTFT / TPOT | 减少 kernel launch + 中间 tensor 的 HBM 访问 | GPU 核融合 |
| Prefix Caching | TTFT | 跳过公共前缀的 prefill 计算 | RadixAttention (SGLang) |
| PD 分离 | TTFT / TPOT 稳定性 | 消除 P/D 干扰，各自独立优化 | 分布式部署分离 |
| CUDA Graph | TPOT (decode) | 减少 kernel launch overhead (可达 30-50%) | 静态图捕获 |
```

---

## 索引

```
| # | 面试题 | 章节 |
|:--:|-------|:----:|
| 1 | RoPE/ALiBi + 长上下文外推 (PI/NTK/YaRN) | 1.1-1.2 |
| 2 | 推理框架对比 (vLLM/TRT-LLM/SGLang/TGI) | 2.1-2.2 |
| 3 | 为什么 LLM 不用 Beam Search | 3 |
| 4 | PTQ vs QAT 详细对比 | 4, 12 |
| 5 | 量化对 KV Cache 的影响 | 5 |
| 6 | 算子融合的限制 | 6.1-6.2 |
| 7 | 端到端推理优化清单 | 7.1-7.2 |
| 8 | 线上推理慢/OOM 排查 | 8.1-8.2 |
| 9 | 新硬件架构对推理的影响 | 9.1 |
| 10 | 国产 GPU / 推理专用 NPU | 9.2 |
| 11 | 推理服务化 QPS/P99/调度 | 10.1-10.2 |
| 12 | Prefix Cache 命中率度量与优化 | 11.1-11.2 |
| 13 | PTQ/QAT 全量对比表 | 12 |
| 14 | 影响吞吐/延迟关键因素综合表 | 13 |
```
