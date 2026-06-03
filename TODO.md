# AI Infra CUDA 学习路线图

> 目标：达到 AI 推理岗位能力要求 —— 掌握 CUDA 算子编写、kernel 融合、nsys 性能分析
> 当前进度：L0~L2 基础完成，L3 部分完成；已搭建 LeetCUDA submodule 环境

---

## 阶段 1: Warp-Level Primitive 补课 (1~2 周)

当前你的 reduce/dot-product 仍用 atomicAdd + 串行循环，效率低。这一步升级到 warp shuffle：

- [ ] **warp reduce with `__shfl_down_sync`**
  - 改写 puzzle 8/10/11 中的 reduce，用树形 warp shuffle 替代 `for` + `atomicAdd`
  - 目标：256 元素 block 内 reduce 只需 log2(32)=5 轮 shuffle + 1 次 atomic
  - 文件：新建 `src/Puzzle/reduce_warp.cu`

- [ ] **warp scan (prefix sum) with `__shfl_up_sync`**
  - 用 warp-level inclusive/exclusive scan 改写 puzzle 10
  - 文件：新建 `src/Puzzle/scan_warp.cu`

- [ ] **bank conflict 实验**
  - 构造一个 bank conflict demo，在 nsys/ncu 中观测 bank conflict 计数
  - 理解 padding 消除冲突的效果
  - 文件：新建 `src/Puzzle/bank_conflict.cu`

---

## 阶段 2: LeetCUDA Easy/Medium Kernels (2~3 周)

开始使用 LeetCUDA 仓库中的 kernel 代码，逐个精读并在本地编译运行：

- [ ] **elementwise 系列** — 理解激活函数 kernel 的标准写法
  - `kernels/elementwise/` 下的 relu, gelu, swish, silu, sigmoid
  - 注意：bound check、向量化加载 (float4)、fused kernel 模式
  - 自己写一个 fused `gelu + mul + add` kernel

- [ ] **softmax** ★重点
  - 先实现 naive softmax（会溢出）
  - 再实现 online safe softmax（max-reduce + exp-sum-reduce + rescale）
  - 参考 `kernels/softmax/`
  - 这是面试最高频算子之一

- [ ] **layer-norm / rms-norm** ★重点
  - 实现：warp reduce 算 mean/variance → normalize → affine
  - fused kernel：避免多次 global memory round-trip
  - 参考 `kernels/layer-norm/` 和 `kernels/rms-norm/`
  - 使用 nsys 对比 fused vs unfused 性能

- [ ] **rope** (RoPE)
  - 理解 LLM 中 RoPE 的数学原理
  - 实现 in-place fused RoPE kernel
  - 参考 `kernels/rope/`

- [ ] **sgemm (朴素 tiled matmul)**
  - 你已经有了 puzzle 12，现在对比 LeetCUDA 的 `kernels/sgemm/`
  - 重点：register tiling, 双缓冲, 边界处零填充

---

## 阶段 3: 精度体系与 Tensor Cores 入门 (2~3 周)

进入 AI 推理的核心 —— 混合精度计算。

- [ ] **理解 FP16/BF16/TF32/FP8 精度体系**
  - FP16: 5-bit exponent, 10-bit mantissa, range ~65504
  - BF16: 8-bit exponent, 7-bit mantissa, 与 FP32 相同 range
  - TF32: 8-bit exponent, 10-bit mantissa (Tensor Core 内部格式)
  - FP8: E4M3 / E5M2, Hopper 引入
  - 阅读 `kernels/hgemm/` 中的相关文档

- [ ] **HGEMM WMMA 版本**
  - 使用 `nvcuda::wmma` API 写一个半精度矩阵乘
  - `wmma::fragment`, `wmma::load_matrix_sync`, `wmma::mma_sync`
  - 参考 `kernels/hgemm/` 中的 WMMA 实现
  - 文件：新建 `src/hgemm_wmma.cu`

- [ ] **HGEMM MMA PTX 版本**
  - 从 WMMA 升级到 MMA PTX 内联汇编
  - `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
  - 目标：接近 cuBLAS 性能
  - 参考 `kernels/hgemm/` 中的 MMA 实现

---

## 阶段 4: HGEMM 优化全流程 (3~4 周)

将 HGEMM 从「能跑」优化到「98% cuBLAS 性能」：

- [ ] **Multi-Stage Pipeline (2~4 stages)**
  - 理解 producer-consumer 流水线
  - `cp.async` + commit group + wait group
  - 隐藏 global→shared 访存延迟

- [ ] **Register Double Buffer**
  - A/B fragment 各两份，交错计算

- [ ] **Shared Memory Padding**
  - 消除 bank conflict，padding 到奇数行

- [ ] **Block Swizzle / Warp Swizzle**
  - 理解 swizzle 如何提升 L2 cache 复用率
  - 参考 `kernels/swizzle/` 和 `kernels/hgemm/`

- [ ] **性能测量与 nsys/ncu 分析**
  - 每个优化步骤后用 nsys/ncu 测量
  - 关注：SM occupancy, memory throughput, compute throughput
  - 记录到 `bench/` 目录

---

## 阶段 5: FlashAttention (3~4 周)

LLM 推理最核心的 kernel：

- [ ] **理解 FlashAttention 算法**
  - 阅读原始论文 (Dao et al. 2022)
  - 理解 online softmax + tiling + recomputation 的设计

- [ ] **从 LeetCUDA FA-2 MMA 代码学习**
  - 参考 `kernels/flash-attn/`
  - 先理解 Split KV (FA-1 风格) 的简化版
  - 再理解 Split Q (FA-2 风格) 的正向实现

- [ ] **实现一个简化版 FlashAttention**
  - FP32，single head，无 causal mask
  - 目标：理解 QKV tiling 循环结构
  - 文件：新建 `src/flash_attn_naive.cu`

---

## 阶段 6: 进阶主题 (持续)

- [ ] **CUTLASS / CuTe**
  - 阅读 `kernels/cutlass/` 示例
  - 理解 CuTe 的 Layout/Tile/Copy 抽象
  - 写一个 CuTe GEMM 示例

- [ ] **Triton 对照**
  - 用 Triton 重写已掌握的 kernel
  - 对比 CUDA vs Triton 的开发效率与性能

- [ ] **实际模型推理场景**
  - 为一个简单 Transformer block 写 fused attention + layernorm
  - 参考 `kernels/transformer/`

- [ ] **开源贡献**
  - 向 LeetCUDA 提交一个自己写的 kernel PR

---

## 每日/每周习惯

- [ ] 每天写至少一个能编译运行的小 kernel
- [ ] 每个 kernel 附 nsys profile，记录到 `bench/`
- [ ] 阅读 LeetCUDA README 中的 LLM/CUDA blog 链接（100+ 篇）
- [ ] 维护本 TODO.md，打勾完成项

---

*最后更新: 2026-05-30*

