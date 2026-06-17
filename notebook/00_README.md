# CUDA 学习笔记 · Notebook

> 从零开始, 达到 Tier-1 面试水平的 CUDA/GPU 知识体系。
> 参考 LeetCUDA 源码、NVIDIA 官方文档、多篇经典博客。

---

## 阅读顺序

教程按 ROUTE.md 的 M0-M9 路线组织。新文件编号 06+ 是对应 ROUTE 各模块的系统学习资料。

### 第一阶段: 地基 (M0 — 硬件与编程基础)

| 编号 | 文件 | 主题 | 依赖 |
|:---:|------|------|------|
| 01 | [01_gpu_hardware_architecture.md](01_gpu_hardware_architecture.md) | GPU 硬件架构 | — |
| 02 | [02_cuda_programming_model.md](02_cuda_programming_model.md) | CUDA 编程模型 | 01 |
| 03 | [03_gpu_memory_hierarchy.md](03_gpu_memory_hierarchy.md) | GPU 内存层级 | 01 |
| 04 | [04_warp_execution_model.md](04_warp_execution_model.md) | Warp 执行模型 | 01,02 |
| 05 | [05_warp_shuffle_primitives.md](05_warp_shuffle_primitives.md) | Warp Shuffle 原语 | 04 |
| 06 | [06_roofline_and_flops.md](06_roofline_and_flops.md) | Roofline 模型与 FLOPs 计算 | 01,03 |
| 07 | [07_numerical_formats.md](07_numerical_formats.md) | 数值格式全家桶 | 01,06 |
| 08 | [08_transformer_architecture.md](08_transformer_architecture.md) | Transformer Decoder 架构 | 03,06,07 |

### 第二阶段: Shared Memory & Warp-Level Reduce/Scan (M0)

| — | [bank_conflict_learning.md](bank_conflict_learning.md) | Shared Memory Bank Conflict | 01,03 |
| — | [reduce_warp_learning.md](reduce_warp_learning.md) | Warp Reduce 实战 | 05 |
| — | [scan_warp_learning.md](scan_warp_learning.md) | Warp Scan 实战 | 05,reduce_warp_learning |

### 第三阶段: Softmax / Normalization / Tensor Core (M0)

| — | [softmax_learning.md](softmax_learning.md) | Softmax: Naive→Safe→Online | reduce_warp_learning |
| — | [layernorm_rmsnorm_learning.md](layernorm_rmsnorm_learning.md) | LayerNorm & RMSNorm fused | reduce_warp_learning |
| — | [tensor_cores_intro.md](tensor_cores_intro.md) | Tensor Cores: WMMA→wgmma→tcgen05 | reduce_warp_learning |
| — | [hgemm_optimization.md](hgemm_optimization.md) | HGEMM 优化全流程 | tensor_cores_intro |

### 第四阶段: 推理工作负载 (M1) + FlashAttention (M2)

| — | [09_inference_workload.md](09_inference_workload.md) | LLM 推理工作负载 | 06,08 |
| — | [flash_attention_learning.md](flash_attention_learning.md) | FlashAttention 原理 | softmax_learning,tensor_cores_intro |
| — | [10_flashattention_deep_dive.md](10_flashattention_deep_dive.md) | FA v2/v3/v4 + FlashDecoding | flash_attention_learning |

### 第五阶段: Attention 变体 (M3) + KV Cache 管理 (M4)

| — | [11_attention_variants.md](11_attention_variants.md) | MHA/MQA/GQA/MLA | 10,08 |
| — | [12_kv_cache_management.md](12_kv_cache_management.md) | PagedAttention/RadixAttention | 11,10 |

### 第六阶段: 调度 (M5) + Kernel 路线 (M6)

| — | [13_scheduling.md](13_scheduling.md) | Continuous Batching/P/D 分离 | 09,12 |
| — | [14_kernel_routes.md](14_kernel_routes.md) | CUDA/CUTLASS/Triton/Compiler | 06,01 |

### 第七阶段: 量化 (M7) + 分布式 (M8) + Profiling (M9)

| — | [15_quantization.md](15_quantization.md) | GPTQ/AWQ/SmoothQuant/FP8 | 07,14 |
| — | [16_distributed.md](16_distributed.md) | 分布式并行 (TP/EP/NCCL) | 09,08,13 |
| — | [17_profiling.md](17_profiling.md) | Nsight/CUDA Graph/Benchmark | 06,04 |

### 第八阶段: 前沿雷达 + 生态全景

| — | [18_frontier_2025_2026.md](18_frontier_2025_2026.md) | Reasoning 负载/稀疏与线性 Attention/大 EP/FP4/Kernel DSL/AI 写算子 | 全部前序章节 |
| — | [19_ai_infra_ecosystem.md](19_ai_infra_ecosystem.md) | 生态全景: Triton/TVM/TensorRT/DeepSpeed/K8s/LLVM 等术语的分层地图与工业定位 | 可随时读, 建议早读 |

### 第九阶段: 工程纵深（补全"单 kernel"之外的世界）

| — | [20_cuda_streams_async.md](20_cuda_streams_async.md) | CUDA 并发: Stream/Event/Pinned/三路 Overlap/CUDA Graph | 02,03,17 |
| — | [21_pytorch_op_integration.md](21_pytorch_op_integration.md) | 算子接入 PyTorch: 扩展/autograd/compile 兼容/数值验证方法论 | 14,triton/ |
| — | [22_sampling_decoding.md](22_sampling_decoding.md) | 采样与解码: top-k/p、投机解码拒绝采样证明、constrained decoding | 08,09,13 |
| — | [23_moe_inference.md](23_moe_inference.md) | MoE 推理深入: Router/Grouped GEMM/负载均衡/roofline 账 | 16,18,06 |

### 速查

| — | [99_interview_cheatsheet.md](99_interview_cheatsheet.md) | 面试速查卡: 数字阶梯/公式卡/一句话答案 Top20/最后一晚清单 | 面试前 1-2 天刷 |

### OpenAI Triton 专题教程 (必达 L3, 6 篇)

| 篇 | 文件 | 内容 |
|:---:|------|------|
| 00 | [triton/00_README.md](triton/00_README.md) | 总览 + 安装 + SM75 注意事项 + 学习路径 |
| 01 | [triton/01_programming_model.md](triton/01_programming_model.md) | 编程模型: program/grid/mask, vs CUDA |
| 02 | [triton/02_fused_softmax.md](triton/02_fused_softmax.md) | 归约范式: fused softmax / RMSNorm |
| 03 | [triton/03_matmul_autotune.md](triton/03_matmul_autotune.md) | tl.dot / L2 swizzle / autotune |
| 04 | [triton/04_flash_attention.md](triton/04_flash_attention.md) | flash attention 默写训练 (面试件) |
| 05 | [triton/05_internals_debug.md](triton/05_internals_debug.md) | 编译管线 (TTIR/TTGIR/PTX)、调试、性能排查 |
| 06 | [triton/06_exercises.md](triton/06_exercises.md) | 12 道阶梯练习 + 验收标准 |

### 开源项目源码精读 (理论学完后，对照真实代码)

| 优先级 | 文件 | 项目 | 一句话 |
|:---:|------|------|------|
| 🥇 | [source_reading/01_vllm/README.md](source_reading/01_vllm/README.md) | vLLM | V1 引擎，覆盖调度/PagedKV/量化/分布式/投机解码 |
| 🥈 | [source_reading/02_sglang/README.md](source_reading/02_sglang/README.md) | SGLang | RadixAttention + P/D 分离生产实现，对比 vLLM |
| 🥉 | [source_reading/03_flashinfer/README.md](source_reading/03_flashinfer/README.md) | FlashInfer | vLLM/SGLang 共用的 attention/MLA/MoE kernel 底层 |
| 4 | [source_reading/04_deepseek_infra/README.md](source_reading/04_deepseek_infra/README.md) | FlashMLA/DeepEP/DeepGEMM | MLA/大EP/FP8 GEMM 工业实现 |

> 总览见 [source_reading/00_README.md](source_reading/00_README.md)。

### 实战项目 (1-2 月，理论学完后做)

| 文件 | 项目 | 周期 |
|------|------|:---:|
| [projects/P0_overview.md](projects/P0_overview.md) | 总览：环境约束(RTX 2060/SM75)、时间线、工程规范、报告模板 | — |
| [projects/P1_profiling_gym.md](projects/P1_profiling_gym.md) | Profiling 训练馆 (nsys/ncu 方法论) | 1.5-2 周 |
| [projects/P2_hgemm_cutlass.md](projects/P2_hgemm_cutlass.md) | HGEMM: WMMA→优化→CUTLASS/CuTe | 2-2.5 周 |
| [projects/P3_flashattention.md](projects/P3_flashattention.md) | FlashAttention forward (CUDA+Triton) | 2-2.5 周 |
| [projects/P4_flashdecoding_paged_kv.md](projects/P4_flashdecoding_paged_kv.md) | FlashDecoding split-KV + 玩具 Paged KV | 1.5 周 |
| [projects/P5_w4a16_dequant_gemm.md](projects/P5_w4a16_dequant_gemm.md) | W4A16 Dequant-GEMM (Triton) | 1-1.5 周 |

---

## 快速导航: 按面试题查找 (新增)

| 面试题 | 去哪个文件 |
|------|------|
| "算术强度怎么算? 瓶颈在哪?" | [06](06_roofline_and_flops.md) §3-4 |
| "decode 为什么 memory-bound?" | [06](06_roofline_and_flops.md) §3, [09](09_inference_workload.md) §1 |
| "BF16 vs FP16 区别?" | [07](07_numerical_formats.md) §1.3 |
| "W4A16 vs W8A8 分别打哪个阶段?" | [07](07_numerical_formats.md) §2, [15](15_quantization.md) §2-3 |
| "Llama decoder layer 手画?" | [08](08_transformer_architecture.md) §1 |
| "RoPE 怎么工作的?" | [08](08_transformer_architecture.md) §2.2 |
| "Prefill vs Decode 二分?" | [09](09_inference_workload.md) §1 |
| "TTFT / TPOT / goodput 区别?" | [09](09_inference_workload.md) §2 |
| "Online Softmax 递推推导?" | [10](10_flashattention_deep_dive.md) §1, [softmax_learning.md](softmax_learning.md) §2 |
| "FA v2→v3→v4 改了什么?" | [10](10_flashattention_deep_dive.md) §3-5 |
| "FlashDecoding split-KV 为什么?" | [10](10_flashattention_deep_dive.md) §6 |
| "MHA / MQA / GQA / MLA 对比?" | [11](11_attention_variants.md) §2-4 |
| "KV cache 大小公式?" | [11](11_attention_variants.md) §1 |
| "MLA decoupled RoPE 怎么解?" | [11](11_attention_variants.md) §3.3 |
| "PagedAttention 解决什么?" | [12](12_kv_cache_management.md) §2 |
| "RadixAttention vs APC?" | [12](12_kv_cache_management.md) §3.3 |
| "Chunked Prefill 怎么平滑 TPOT?" | [13](13_scheduling.md) §2.3 |
| "P/D 分离架构?" | [13](13_scheduling.md) §3 |
| "Speculative Decoding 原理?" | [13](13_scheduling.md) §4 |
| "选 Triton 还是手写 CUDA?" | [14](14_kernel_routes.md) §1, §4.2 |
| "CuTe layout 解决了什么?" | [14](14_kernel_routes.md) §3.2 |
| "Epilogue Fusion 是什么?" | [14](14_kernel_routes.md) §3.3 |
| "GPTQ vs AWQ 差异?" | [15](15_quantization.md) §2 |
| "NVFP4 两级 scale 为什么?" | [15](15_quantization.md) §5 |
| "Megatron TP 怎么切?" | [16](16_distributed.md) §2 |
| "Nsight 怎么看瓶颈?" | [17](17_profiling.md) §2-3 |
| "CUDA Graph 什么时候用?" | [17](17_profiling.md) §4 |
| "MLA 推理为什么要矩阵吸收?" | [11](11_attention_variants.md) §3.2b |
| "Reasoning 模型对 infra 的影响?" | [18](18_frontier_2025_2026.md) §1 |
| "NSA / MoBA / 稀疏 attention?" | [18](18_frontier_2025_2026.md) §2.2 |
| "线性/混合 attention 对 serving 的影响?" | [18](18_frontier_2025_2026.md) §2.3 |
| "MTP 和投机解码什么关系?" | [18](18_frontier_2025_2026.md) §2.4, [13](13_scheduling.md) §4 |
| "怎么看 FP4 / AI 写 kernel?" | [18](18_frontier_2025_2026.md) §6, §5.2 |
| "OpenAI Triton 和 NVIDIA Triton 区别?" | [19](19_ai_infra_ecosystem.md) §3 |
| "讲讲推理的全链路（token 生命周期）?" | [19](19_ai_infra_ecosystem.md) §6 |
| "TVM/torch.compile/手写 kernel 边界?" | [19](19_ai_infra_ecosystem.md) §3, [14](14_kernel_routes.md) §1 |
| "现场写 Triton softmax/matmul/attention?" | [triton/02](triton/02_fused_softmax.md), [03](triton/03_matmul_autotune.md), [04](triton/04_flash_attention.md) |
| "ZeRO 和 TP 的区别?" | [19](19_ai_infra_ecosystem.md) §8 题4, [16](16_distributed.md) |
| "怎么 overlap 计算和拷贝? pinned 为什么必须?" | [20](20_cuda_streams_async.md) §2-3 |
| "你的 kernel 怎么接进 PyTorch / 不破坏 compile?" | [21](21_pytorch_op_integration.md) §1-3 |
| "kernel 怎么验证是对的? 容差怎么定?" | [21](21_pytorch_op_integration.md) §4 |
| "top-p 采样为什么贵? 怎么优化?" | [22](22_sampling_decoding.md) §2 |
| "证明投机解码无损?" | [22](22_sampling_decoding.md) §3 |
| "JSON mode / constrained decoding 怎么实现?" | [22](22_sampling_decoding.md) §4 |
| "MoE forward 数据流 / grouped GEMM?" | [23](23_moe_inference.md) §1-2 |
| "MoE 省什么不省什么?" | [23](23_moe_inference.md) §3 |

---

## 建议学习路径 (时间估计)

```
Day 1:  01 + 02                                   GPU 架构 + 编程模型           ≈ 3h
Day 2:  03 + 04                                   内存层级 + Warp 执行模型       ≈ 4h
Day 3:  05 + bank_conflict + reduce/scan           Shuffle + Reduce + Scan       ≈ 4h
Day 4:  06 + 07                                   Roofline + 数值格式            ≈ 3h
Day 5:  08 + softmax + layernorm                   Transformer + Norm            ≈ 3h
Day 6:  tensor_cores + hgemm                       Tensor Core + GEMM 优化       ≈ 4h
Day 7:  flash_attention_learning + 10              FlashAttention v1→v4          ≈ 4h
Day 8:  09 + 11                                   推理负载 + Attention 变体      ≈ 3h
Day 9:  12 + 13                                   KV 管理 + 调度                ≈ 4h
Day 10: 14 + 15                                   Kernel 路线 + 量化            ≈ 4h
Day 11: 16 + 17                                   分布式 + Profiling            ≈ 3h
Day 12: 20 + 21                                   CUDA 并发 + 算子接入           ≈ 3h
Day 13: 22 + 23                                   采样解码 + MoE 推理            ≈ 3h
Day 14: 18 + 19                                   前沿雷达 + 生态全景            ≈ 3h
Day 15: 99 + 复习 + 面试自测                       速查卡 + ROUTE.md 第9节        ≈ 全部
（triton/ 教程与 projects/ 实战穿插在理论学习之后, 见各自 README）
```

---

## LeetCUDA 源码映射速查

| LeetCUDA 目录 | 对应 Notebook 文件 |
|------|------|
| `kernels/sgemm/` | 06 (roofline), 14 (kernel routes) |
| `kernels/hgemm/` | tensor_cores_intro, hgemm_optimization, 06 |
| `kernels/flash-attn/` | flash_attention_learning, 10 |
| `kernels/rope/` | 08 (transformer) |
| `kernels/rms-norm/`, `kernels/layer-norm/` | layernorm_rmsnorm_learning, 08 |
| `kernels/gelu/`, `kernels/swish/`, `kernels/relu/` | 08, 07 |
| `kernels/reduce/` | reduce_warp_learning |
| `kernels/softmax/` | softmax_learning, 10 |
| `kernels/cutlass/` | 14 (kernel routes) |
| `kernels/openai-triton/` | 14 (kernel routes) |
| `kernels/ws-hgemm/` | tensor_cores_intro, hgemm_optimization |
| `kernels/nvidia-nsight/` | 17 (profiling) |
| `kernels/elementwise/`, `kernels/embedding/` | 03, 08 |
| `kernels/sgemv/`, `kernels/hgemv/` | 08 (GEMM variants), 09 (decode 的 GEMV 本质) |
| `kernels/mat-transpose/` | 03 (memory access patterns) |
| `kernels/swizzle/` | hgemm_optimization |
| `kernels/dot-product/` | reduce_warp_learning |
| `kernels/openai-triton/merge-attn-states/` | 10 §6 (FlashDecoding split-KV 的 merge 步骤) |
| `kernels/cutlass/cute_dsl/` | 14, 18 §5.1 (CuTe-DSL) |
| `ffpa-attn/` (仓库顶层) | 10, 11 (大 head_dim attention, MLA 场景) |
| `kernels/histogram/`, `kernels/nms/` | 进阶示例 |

> 注意：`kernels/transformer/` 是空目录；量化 (15)、调度 (13)、KV 管理 (12)、
> 分布式 (16) 在 LeetCUDA 中没有对应实现，请按各章"源码映射"小节读
> vLLM / SGLang / FlashInfer / DeepSeek 开源库（FlashMLA、DeepEP 等）。

---

## 参考资源

- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) — 200+ CUDA Kernels, 100+ LLM/CUDA 博客
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA Ampere Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [ROUTE.md](ROUTE.md) — 完整学习路线与面试白板高频题
