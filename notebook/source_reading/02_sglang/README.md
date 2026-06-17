# SGLang 源码精读

> 仓库：[sgl-project/sglang](https://github.com/sgl-project/sglang)，默认分支 `main`，
> Python 代码在 `python/sglang/srt/`（srt = SGLang Runtime，serving 引擎本体）。
> fetch 时间：2026-06-16。

为什么排第二而不是第一：vLLM 是更通用的对照基准，但 **RadixAttention（基数树前缀复用）
和当前的 P/D 分离/投机解码生态在 SGLang 里实现得更"显式"**，适合拿来和 vLLM 的对应模块
做横向对比，这正是面试爱问的"两者怎么不一样"。

---

## 1. 代码地图：概念 → 目录 → 关键文件

| 概念（notebook 章节） | 目录 | 关键文件 |
|------|------|------|
| 调度 / continuous batching（[13](../../13_scheduling.md)） | `srt/managers/` | `scheduler.py`（主循环：collect new requests → 决定 batch → 跑一步）、`schedule_policy.py`（选哪些请求入 batch 的策略，含 chunked prefill 相关逻辑）、`schedule_batch.py`（batch 数据结构） |
| RadixAttention / 前缀缓存（[12](../../12_kv_cache_management.md) §3.3） | `srt/mem_cache/` | `radix_cache.py`（基数树本体，插入/匹配/驱逐）、`hiradix_cache.py`（分层 radix，结合 CPU/磁盘 offload）、`evict_policy.py`（LRU 等驱逐策略）、`memory_pool.py`（底层 KV 物理存储池，等价于 vLLM 的 block_pool） |
| 滑窗/混合架构 KV（对应 [18](../../18_frontier_2025_2026.md) 线性/混合 attention） | `srt/mem_cache/` | `swa_radix_cache.py`、`base_swa_memory_pool.py`、`mamba_radix_cache.py` |
| P/D 分离（[13](../../13_scheduling.md) §3） | `srt/disaggregation/` | `prefill.py` / `decode.py`（两侧角色逻辑）、`mooncake/`（接 Mooncake KV transfer，对应你笔记里月之暗面的归属纠正）、`nixl/`（接 NVIDIA NIXL）、`common/` |
| 投机解码（[22](../../22_sampling_decoding.md)） | `srt/speculative/` | `eagle_worker_v2.py`/`eagle_info.py`（EAGLE）、`ngram_worker.py`（无草稿模型）、`multi_layer_eagle_*`（多层 EAGLE 变体）、`frozen_kv_mtp_*`（MTP 风格、对应 [18](../../18_frontier_2025_2026.md) §2.4 "MTP 即原生投机解码"） |
| 量化（[15](../../15_quantization.md)） | `srt/layers/quantization/` | `fp8.py`/`fp8_kernel.py`、`awq/`、`gptq/`、`marlin_utils_fp4.py`/`marlin_utils_fp8.py`、`mxfp4.py`/`mxfp4_marlin_moe.py`（Blackwell MXFP4 + MoE 融合） |
| MoE / EP（[23](../../23_moe_inference.md) [16](../../16_distributed.md)） | `srt/` | `eplb/`（专家负载均衡，对照 vLLM `distributed/eplb/` 和 DeepSeek 的 EPLB） |
| EAGLE/n-gram 等草稿用的自定义算子 | `srt/speculative/triton_ops/`、`srt/speculative/cpp_ngram/` | — |

---

## 2. 精读路径

1. **调度器主循环**：`srt/managers/scheduler.py` 通读一遍 `event_loop` 相关函数，
   对照 [13](../../13_scheduling.md) §2 的 continuous batching 流程图逐行对号。
2. **RadixAttention vs vLLM APC**：读 `srt/mem_cache/radix_cache.py` 的 `match_prefix`/`insert`/`evict`，
   对照 [12](../../12_kv_cache_management.md) §3.3，写下两句话——
   RadixAttention 是显式基数树结构（前缀天然就是树的祖先链），
   vLLM APC 是 block hash 表（前缀复用靠 hash 命中），这是数据结构层面的真实区别。
3. **P/D 分离**：`srt/disaggregation/prefill.py` + `decode.py` 配合 `mooncake/` 子目录看一遍 KV 怎么从 P 侧发到 D 侧。
4. **投机解码**：`srt/speculative/eagle_info.py` + `eagle_worker_v2.py`，
   对照 vLLM 的 `vllm/v1/spec_decode/eagle.py` 做一次两边实现对比笔记（面试加分项："我对比过两家怎么实现 EAGLE"）。

---

## 3. 常见面试追问 → 这份代码怎么答

- **"RadixAttention 和 vLLM 的 Automatic Prefix Caching 有什么本质区别？"**
  → 数据结构不同（radix tree vs block hash），由此带来的差异：radix tree 天然支持
  **任意长度前缀的最长公共前缀匹配** 和 LRU 驱逐时按树结构做的"子树整体驱逐"，
  而 block-hash 方式匹配粒度固定在 block_size，实现更简单但灵活性略低。代码证据见上表。
- **"SGLang 的 P/D 分离怎么对接 Mooncake？"** → `srt/disaggregation/mooncake/` 整个子目录就是答案，
  对照 [13](../../13_scheduling.md) §3 讲清"以 KV cache 为中心的传输池"这个设计点。
- **"混合/线性 attention 模型（如 Kimi Linear）怎么塞进现有 KV 管理框架？"**
  → `srt/mem_cache/mamba_radix_cache.py` 是真实证据：线性注意力的"状态"不是传统 KV，
  需要专门的 cache 类，呼应 [18](../../18_frontier_2025_2026.md) §2.3 "线性/混合 attention 打破 paged KV 假设"。

---

## 4. 备注

- SGLang 的 `srt/managers/scheduler_components/` 是新拆出来的子模块，内容会比较快变化，先看 `scheduler.py` 主体再钻进去。
- 与 vLLM 对比着读收益最大；建议每读完一个模块就去 [01_vllm/README.md](../01_vllm/README.md) 找对应小节互相补充。
