# CUDA 学习笔记 · Notebook

> 从零开始, 达到面试水平的 CUDA/GPU 知识体系。
> 参考 LeetCUDA 源码、NVIDIA 官方文档、多篇经典博客。

---

## 阅读顺序

教程按 **"从硬件到软件, 从基础到实战"** 组织。新文件编号 01-05 是零基础入门路线, 之后是现有的专题笔记。

### 第一阶段: 地基 (零基础必读)

| 编号 | 文件 | 主题 | 关键收获 |
|:---:|------|------|------|
| 01 | [01_gpu_hardware_architecture.md](01_gpu_hardware_architecture.md) | GPU 硬件架构 | SM 内部结构、CUDA Core vs Tensor Core、各代演进 |
| 02 | [02_cuda_programming_model.md](02_cuda_programming_model.md) | CUDA 编程模型 | `<<<>>>` 语法、threadIdx/blockIdx、`__syncthreads()` |
| 03 | [03_gpu_memory_hierarchy.md](03_gpu_memory_hierarchy.md) | GPU 内存层级 | Register/Shared/Global/Constant 六种内存、coalesced access |
| 04 | [04_warp_execution_model.md](04_warp_execution_model.md) | Warp 执行模型 | SIMT、warp divergence、zero-overhead switching、occupancy 计算 |
| 05 | [05_warp_shuffle_primitives.md](05_warp_shuffle_primitives.md) | Warp Shuffle 原语 | `shfl_down/xor/up` 四个原语、mask、butterfly reduce、block reduce 两阶段 |

### 第二阶段: Shared Memory & Bank Conflict

| 编号 | 文件 | 主题 | 依赖 |
|------|------|------|------|
| — | [bank_conflict_learning.md](bank_conflict_learning.md) | Shared Memory Bank Conflict | 01, 03 |

理解 32 bank 结构、stride=1 vs stride=32 vs padding、n-way conflict 计算。

### 第三阶段: Warp-Level Reduce & Scan

| 编号 | 文件 | 主题 | 依赖 |
|------|------|------|------|
| — | [reduce_warp_learning.md](reduce_warp_learning.md) | Warp Reduce 实战 | 05 |
| — | [scan_warp_learning.md](scan_warp_learning.md) | Warp Scan 实战 | 05, reduce_warp_learning |

从 shared memory 版本升级到 warp shuffle 版本; `shfl_down` vs `shfl_xor` 选型; block-level 两阶段设计; Hillis-Steele prefix sum。

### 第四阶段: Softmax & Normalization

| 编号 | 文件 | 主题 | 依赖 |
|------|------|------|------|
| — | [softmax_learning.md](softmax_learning.md) | Softmax: Naive→Safe→Online | reduce_warp_learning |
| — | [layernorm_rmsnorm_learning.md](layernorm_rmsnorm_learning.md) | LayerNorm & RMSNorm fused | reduce_warp_learning, softmax_learning |

Online safe softmax → FlashAttention 的数学底座; LayerNorm 的两次 reduce broadcast。

### 第五阶段: Tensor Cores & HGEMM

| 编号 | 文件 | 主题 | 依赖 |
|------|------|------|------|
| — | [tensor_cores_intro.md](tensor_cores_intro.md) | Tensor Cores 精度与 WMMA | reduce_warp_learning, bank_conflict |
| — | [hgemm_optimization.md](hgemm_optimization.md) | HGEMM 优化全流程 | tensor_cores_intro |

FP16/BF16/TF32 精度体系; WMMA → MMA PTX; Multi-Stage Pipeline; Block Swizzle。

### 第六阶段: FlashAttention

| 编号 | 文件 | 主题 | 依赖 |
|------|------|------|------|
| — | [flash_attention_learning.md](flash_attention_learning.md) | FlashAttention 原理 | softmax_learning, tensor_cores_intro |

O(N²) 内存墙 → Tiling + Online Softmax; Split KV vs Split Q; Shared Memory 布局。

---

## 快速导航: 按面试题查找

| 面试题 | 去哪个文件 |
|------|------|
| "GPU 和 CPU 有什么区别?" | [01](01_gpu_hardware_architecture.md) §1, §7 |
| "SM 是什么? 里面有什么?" | [01](01_gpu_hardware_architecture.md) §2, §3 |
| "thread/block/grid 怎么对应硬件?" | [01](01_gpu_hardware_architecture.md) §5, [02](02_cuda_programming_model.md) §4 |
| "GPU 有哪些内存? 速度排序?" | [03](03_gpu_memory_hierarchy.md) §1, §4 |
| "什么是 coalesced access?" | [03](03_gpu_memory_hierarchy.md) §2.3 |
| "寄存器 spilling 是什么?" | [03](03_gpu_memory_hierarchy.md) §2.1, §2.6 |
| "什么是 warp? 为什么是 32?" | [04](04_warp_execution_model.md) §1, [05](05_warp_shuffle_primitives.md) §2 |
| "SIMT 和 SIMD 的区别?" | [04](04_warp_execution_model.md) §2 |
| "什么是 warp divergence?" | [04](04_warp_execution_model.md) §4 |
| "Occupancy 怎么算?" | [04](04_warp_execution_model.md) §5 |
| "GPU 为什么能零开销线程切换?" | [04](04_warp_execution_model.md) §3.2 |
| "Warp Shuffle 四个原语的差别?" | [05](05_warp_shuffle_primitives.md) §3-7 |
| "shfl_down vs shfl_xor 怎么选?" | [05](05_warp_shuffle_primitives.md) §4.4 |
| "shfl_up prefix sum 为什么要 tmp?" | [05](05_warp_shuffle_primitives.md) §5.3 |
| "Block Reduce 两阶段架构?" | [05](05_warp_shuffle_primitives.md) §10 |
| "Bank Conflict 是什么? 怎么解?" | [bank_conflict_learning.md](bank_conflict_learning.md) |
| "Online Softmax 公式怎么推?" | [softmax_learning.md](softmax_learning.md) §2 |
| "LayerNorm vs RMSNorm 区别?" | [layernorm_rmsnorm_learning.md](layernorm_rmsnorm_learning.md) §1 |
| "Tensor Core 怎么工作?" | [tensor_cores_intro.md](tensor_cores_intro.md) §2 |
| "FP16/BF16/TF32 精度对比?" | [tensor_cores_intro.md](tensor_cores_intro.md) §1 |
| "FlashAttention 核心思想?" | [flash_attention_learning.md](flash_attention_learning.md) §2 |
| "FlashAttention Split Q vs Split KV?" | [flash_attention_learning.md](flash_attention_learning.md) §3 |

---

## 建议学习路径 (时间估计)

```
Day 1:  01 (GPU 架构)       + 02 (编程模型)         ≈ 3h
Day 2:  03 (内存层级)       + 04 (Warp 执行模型)     ≈ 4h
Day 3:  05 (Warp Shuffle)   + bank_conflict_learning ≈ 4h
Day 4:  reduce_warp_learning  + scan_warp_learning    ≈ 4h
Day 5:  softmax_learning    + layernorm_rmsnorm       ≈ 3h
Day 6:  tensor_cores_intro  + hgemm_optimization      ≈ 4h
Day 7:  flash_attention_learning                       ≈ 3h
```

---

## 参考资源

- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) — 200+ CUDA Kernels, 100+ LLM/CUDA 博客
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA Ampere Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
