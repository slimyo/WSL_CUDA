# vLLM 源码精读

> 仓库：[vllm-project/vllm](https://github.com/vllm-project/vllm)，默认分支 `main`。
> fetch 时间：2026-06-16（通过 GitHub API 拉取目录树确认）。
> **重要背景**：vLLM 已完成 V0→V1 引擎重写，所有新代码在 `vllm/v1/` 下；
> 旧的 `vllm/core/`、`vllm/worker/` 之类的 V0 路径已基本退役/精简，**不要再按网上 2023-2024 年的老博客找路径**。

---

## 1. 代码地图：概念 → 目录 → 关键文件

| 概念（notebook 章节） | 顶层目录 | 关键文件 |
|------|------|------|
| 调度 / continuous batching（[13](../../13_scheduling.md)） | `vllm/v1/core/sched/` | 调度器主循环；决定每步选哪些 request 进 batch |
| PagedAttention / block 管理（[12](../../12_kv_cache_management.md)） | `vllm/v1/core/` | `kv_cache_manager.py`（block 分配/释放总入口）、`block_pool.py`（物理 block 池/free list）、`kv_cache_coordinator.py`、`single_type_kv_cache_manager.py`（按 attention 类型分派，因为 V1 要同时支持 full-attn/SWA/Mamba 等不同 KV 形状） |
| KV cache 规格/异构层（MLA、滑窗等） | `vllm/v1/` | `kv_cache_interface.py`、`kv_cache_spec_registry.py` |
| KV offloading（HBM↔CPU/磁盘分层，对应 [18](../../18_frontier_2025_2026.md) Mooncake/LMCache 思路） | `vllm/v1/kv_offload/` | `base.py`（抽象接口）、`worker/`（实际搬运）、`tiering/`（分层策略）、`cpu/` |
| Attention kernel 后端选择（[10](../../10_flashattention_deep_dive.md) [11](../../11_attention_variants.md)） | `vllm/v1/attention/` | `selector.py`（按硬件/dtype/是否 MLA 选 backend）、`backends/`（FlashAttention/FlashInfer/Triton/XFormers 等具体后端）、`backend.py`（统一接口定义） |
| GPU 执行循环（model runner，把调度结果真正喂进模型） | `vllm/v1/worker/` | `gpu_model_runner.py`（每步 forward 的核心，prepare input + 跑模型 + 处理输出）、`gpu_worker.py`（单卡 worker 进程）、`block_table.py`（V1 侧的逻辑→物理 block 映射表，对应 PagedAttention 的 block table） |
| 投机解码（[22](../../22_sampling_decoding.md)） | `vllm/v1/spec_decode/` | `eagle.py`（EAGLE）、`medusa.py`、`ngram_proposer.py`/`ngram_proposer_gpu.py`（无草稿模型的 n-gram 猜测）、`draft_model.py`（独立小模型 draft）、`metadata.py`（接受/拒绝逐 token 校验的元数据结构） |
| 量化（[15](../../15_quantization.md)） | `vllm/model_executor/layers/quantization/` | `awq_marlin.py`/`awq.py`（AWQ + Marlin kernel）、`auto_gptq.py`（GPTQ）、`fp8.py`（FP8 W8A8/动态&静态 scale）、`mxfp4.py`（Blackwell MXFP4）、`modelopt.py`（NVIDIA ModelOpt 产出的 FP4/FP8 checkpoint 加载） |
| 分布式并行（[16](../../16_distributed.md)） | `vllm/distributed/` | `parallel_state.py`（TP/PP/DP/EP 进程组管理）、`device_communicators/`（NCCL/自定义 all-reduce）、`eplb/`（专家负载均衡，呼应 DeepSeek EPLB，见 [04_deepseek_infra](../04_deepseek_infra/README.md)）、`kv_transfer/`（P/D 分离时跨节点传 KV，对接 NIXL） |
| MoE 推理（[23](../../23_moe_inference.md)） | `vllm/model_executor/layers/` | 搜 `fused_moe`（grouped GEMM 路由+专家计算融合实现） |

---

## 2. 精读路径（建议顺序）

1. **先看一次完整请求的生命周期骨架**：`vllm/v1/engine/`（请求怎么进来）→
   `vllm/v1/core/sched/`（怎么被选中进 batch，对应 [13](../../13_scheduling.md) continuous batching）→
   `vllm/v1/worker/gpu_model_runner.py`（怎么真正跑一步 forward）→ `vllm/v1/outputs.py`。
2. **PagedAttention 数据结构**：`vllm/v1/core/kv_cache_manager.py` 的 block 分配逻辑 +
   `vllm/v1/worker/block_table.py` 的逻辑块→物理块映射，对照 [12](../../12_kv_cache_management.md) §2 的 block table 图自己画一遍。
3. **Attention backend 怎么被选出来**：`vllm/v1/attention/selector.py`，
   重点看它怎么根据 head_dim/是否 MLA/硬件代际 在 FlashAttention 后端和 FlashInfer 后端之间切换——
   这是 [11](../../11_attention_variants.md) "MLA 需要专门 kernel" 在工程上的落地证据。
4. **量化任选一条线**：从 `vllm/model_executor/layers/quantization/__init__.py` 的注册表进去，
   挑 `fp8.py` 或 `awq_marlin.py` 通读一遍 `apply()`/`process_weights_after_loading()`，
   对照 [15](../../15_quantization.md) §2-4 看"反量化在哪一步融进 GEMM"。
5. **分布式**：`vllm/distributed/parallel_state.py` 看 TP/PP/DP/EP 四种 group 如何并存，
   再看 `eplb/` 目录下专家负载怎么统计与重排，对照 [23](../../23_moe_inference.md) §3。
6. **投机解码**：`vllm/v1/spec_decode/eagle.py` 通读一遍，对照 [22](../../22_sampling_decoding.md) §3
   的拒绝采样证明，找出代码里对应"接受概率 min(1, p_target/p_draft)"的那一行。

---

## 3. 常见面试追问 → 这份代码怎么答

- **"vLLM 的 KV block 默认多大？在哪改？"** → block size 是调度/KV 管理的全局配置项，
  顺着 `vllm/v1/core/kv_cache_manager.py` 找 block_size 的来源（来自 `vllm/config/` 的 cache 配置），
  能讲清 [12](../../12_kv_cache_management.md) §2 的 block_size trade-off 就够，不需要背具体默认值（不同版本会变）。
- **"V1 为什么要把 KV cache manager 拆成 coordinator + single_type_manager？"**
  → 因为同一引擎现在要同时服务 full attention / 滑窗 / Mamba（线性注意力，[18](../../18_frontier_2025_2026.md) §2.3 提到的混合架构）
  这些 **KV 形状完全不同** 的层，必须按 attention 类型分派管理策略，这是 `kv_cache_spec_registry.py` 存在的原因。
- **"vLLM 怎么做 P/D 分离的 KV 传输？"** → `vllm/distributed/kv_transfer/`，
  对照 [13](../../13_scheduling.md) §3 讲清"prefill 算完 KV 要跨节点搬到 decode 池"这一步具体走哪个目录。
- **"AWQ/GPTQ 的 dequant 在 vLLM 里怎么和 GEMM 融合？"** → 答案在 `quantization/awq_marlin.py` /
  `marlin_utils*.py`，Marlin 是专门为 4bit 反量化+GEMM 融合写的 kernel，这就是 [15](../../15_quantization.md) "反量化融进 GEMM" 的具体落地。

---

## 4. 备注

- V1 目录里还有大量未在上表列出的子系统（`compilation/` 对应 `torch.compile` 集成、`structured_output/`
  对应 [22](../../22_sampling_decoding.md) §4 constrained decoding、`lora/` 等），先吃完上表再按需扩展。
- 路径可能随版本漂移，使用前用 `curl -s https://api.github.com/repos/vllm-project/vllm/contents/<path>`
  或直接在 GitHub 网页确认文件还在。
