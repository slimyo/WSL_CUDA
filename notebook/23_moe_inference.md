# MoE 推理深入：从 Router 到 Grouped GEMM

> 对象: 推理/算子岗（2025 后旗舰开源模型全是 MoE，16/18 章只讲了通信侧，
>       本章补齐"单卡/kernel 视角的 MoE forward 到底怎么算"）
> 前置: 16_distributed.md §4, 18_frontier_2025_2026.md §2.1, 08_transformer_architecture.md
> 目标: 面试能画 MoE 层的完整数据流、讲清 grouped GEMM 为什么是专门问题、
>       算清 MoE 的 roofline 账

---

## 1. MoE 层 forward 的完整数据流（必须能画）

```
x[T, H]  (T = batch×seq 个 token)
  ↓ ① Router(gate): logits = x @ W_g [H, E] → softmax → top-k 选择
        输出: topk_ids[T, k], topk_weights[T, k]        （小 GEMM + topk, 便宜）
  ↓ ② Permute / Dispatch: 按 expert id 把 token 重排分桶
        → 排序/计数(histogram!) 得到每个 expert 的 token 段
        （EP 跨卡时这步就是 all-to-all, 16 章; 单卡时是显存内 gather）
  ↓ ③ Expert FFN: 对每个 expert e, 它分到的 token 段做
        h_e = silu(x_e @ W1_e) * (x_e @ W3_e);  y_e = h_e @ W2_e
        ← 这就是 grouped GEMM: E 组形状不同(每组 token 数不等)的 GEMM!
  ↓ ④ Unpermute / Combine: 散射回原 token 顺序, 按 topk_weights 加权求和
        y[t] = Σ_i w_i · expert_out(t, i)     (+ shared expert 的输出, 若有)
```

**与 dense FFN 的本质区别**：dense 是一个大 GEMM；MoE 是
"一次数据重排 + E 个小 GEMM + 一次重排回来"——计算省了（每 token 只过 k 个
expert），但引入了**排序/gather/scatter 这些访存型操作**和**形状不定的 GEMM**。

## 2. Grouped GEMM：MoE 时代的核心算子（kernel 岗新宠）

```
问题定义: 同时算 E 个 GEMM_e: [m_e, K] × [K, N]，其中 m_e 运行时才知道且严重不均
为什么不能用现成方案:
  逐个 launch:    E=256 个 kernel, m_e 平均很小 → launch 开销 + SM 喂不满
  batched GEMM:   cuBLAS batched 要求所有 m 相同 → 得 padding 到 max(m_e), 浪费巨大
Grouped GEMM 解法（CUTLASS group 模式 / 工业实现的共同骨架）:
  把所有 expert 的输出 tile 统一编号塞进一个 grid，
  persistent kernel: 每个 thread block 循环领取 "下一个 tile"（经查表知道
  自己属于哪个 expert、对应哪段 token、用哪份权重）→ 一次 launch 吃完全部
```
实现地标（按可读性排序）：
- vLLM `fused_moe` Triton kernel：①②③④ 全融合的单文件教材（`sorted_token_ids`
  + 对齐到 BLOCK 的分桶 → 一个 matmul kernel 服务所有 expert）
- CUTLASS `examples/24_gemm_grouped`
- **DeepGEMM**（DeepSeek, 2025）：FP8 grouped GEMM，JIT 生成，decode 用
  masked 布局避免 CPU 同步（值得读 README 理解设计动机）

## 3. MoE 的 roofline 账（06/09 章方法论的延伸，面试硬通货）

```
以 DeepSeek-V3 形态抽象: E=256 routed + 1 shared, top-8, 总参 671B / 激活 37B

decode batch=1: 每 token 激活 8+1 个 expert → 只需搬这 9 个 expert 的权重?
  错! 下一个 token 选哪 8 个不可预测 → 权重全集必须常驻显存/集群
  → MoE 省的是 FLOPs 和【单 token 权重带宽】，不省显存容量 ← 必考辨析

batch 变大后的新现象（与 dense 相反!）:
  dense:  batch ↑ → 权重搬运摊薄 → 算术强度线性升（09 章）
  MoE:    batch ↑ → 命中的 expert 越来越多 → 要搬的权重也涨
          直到 batch 足够大每个 expert 都摊到足够 token 才进入 dense 式摊薄
  → MoE decode 需要【更大的 batch】才能吃满算力 → 这就是为什么
    DeepSeek 用跨节点大 EP 聚合海量请求（18 章 §4.1）——
    把全集群的 token 汇到每个 expert 上，重建"批量摊薄"效应
```

## 4. 负载均衡：MoE 特有的系统问题

```
训练侧(了解): aux loss / DeepSeek 的 auxiliary-loss-free 偏置法,
             capacity factor 与 token dropping
推理侧(重点):
  热 expert 问题: 真实流量下 expert 命中高度不均 → EP 时热 expert 所在卡成长尾,
                  整个 batch 等最慢的卡（all-to-all 是同步点）
  EPLB (DeepSeek): 统计命中率 → 热 expert 多放几个副本（冗余专家）,
                  路由时在副本间分摊; 周期性重排放置
  单卡推理(你的 2060 跑不动 MoE, 但要懂): 权重放不下 → expert offload 到 CPU,
                  按预测预取（Mixtral offloading 一类工作）
```

## 5. 与其他模块的组合（面试连环问预演）

| 组合 | 要点 |
|------|------|
| MoE + 量化 | expert 权重是显存大头 → W4 收益极大；但每 expert 校准数据少（命中不均）→ 量化更难 |
| MoE + P/D 分离 | P 池 token 多、expert 命中均匀（适合大 EP）；D 池靠大 batch 聚合，attention 和 FFN 的最优并行度撕裂 → Attention-FFN 分离（18 章 §7 MegaScale-Infer）|
| MoE + 投机解码 | draft 多猜的 token 让 batch 内 token 数增加 → 正好缓解 §3 的"批量不足"，组合收益超线性 |
| MoE + KV | MoE 只改 FFN，attention/KV 部分与 dense 完全相同（MLA 与 MoE 正交，DeepSeek 两个都用）|

## 6. 学习检查清单

- [ ] 能画 router→dispatch→grouped GEMM→combine 四步数据流，标出每步的计算/访存类型
- [ ] 能解释 grouped GEMM 为什么不能用 batched GEMM 替代、persistent kernel 怎么解
- [ ] 能讲清"MoE 省 FLOPs 不省显存"、"MoE decode 需要更大 batch"两个反直觉点
- [ ] 能说出热 expert 长尾问题与 EPLB 的解法
- [ ] 读过 vLLM fused_moe Triton kernel（哪怕只读 token 分桶部分）

## 7. 自测 / 面试题

1. 手画 MoE forward。其中哪些步骤是 memory-bound？哪步引入了 CPU-GPU 同步风险？
2. E=64、top-2、batch=4 的 decode：期望激活多少个 expert？这对权重带宽意味着什么？
3. grouped GEMM 和 batched GEMM 的差别？m_e 不均时 padding 方案浪费多少？
4. 为什么 MoE 推理偏好大 EP 而不是把每张卡都放全部 expert（纯 TP/DP）？
5. shared expert 的作用是什么？它对 kernel 实现有什么便利？（形状固定、必命中→可与 routed 部分并行流水）

## 8. 参考

- vLLM: `vllm/model_executor/layers/fused_moe/`（Triton 实现 + 配置调优文件）
- DeepGEMM / DeepEP / EPLB 三个 repo 的 README（DeepSeek 开源周）
- CUTLASS example 24 (grouped GEMM)
- DeepSeekMoE / DeepSeek-V3 论文（细粒度专家 + shared expert + 无辅助损失均衡）
- notebook: 16 章 §4（EP 通信）、18 章 §2.1/§4.1（趋势）、06 章（roofline 方法）
