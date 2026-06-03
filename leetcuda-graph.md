# LeetCUDA 学习图谱

> LeetCUDA 是一个面向初学者的现代 CUDA 学习项目，包含 200+ kernels 和 PyTorch 对照实现。
> 本文档描述 LeetCUDA 的层次化学习路线，按依赖关系组织，从基础到前沿。

---

## 全景结构

```
                    ┌─────────────────────────────────────┐
                    │        FlashAttention-MMA △          │  前沿：生产级 attention kernel
                    │  (Split KV/Q, QKV Tiling, Prefetch)   │
                    └──────────────┬──────────────────────┘
                                   │ 依赖
                    ┌──────────────▼──────────────────────┐
                    │        HGEMM (Tensor Cores) △         │  高阶：WMMA/MMA/CuTe, 98% cuBLAS
                    │  Multi-Stage, Swizzle, Double Buffer │
                    └──────────────┬──────────────────────┘
                                   │ 依赖
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼──────┐  ┌─────────▼─────────┐  ┌───────▼──────────┐
    │  Layer Norm    │  │  Softmax △        │  │  Matmul Tiled △  │
    │  RMS Norm      │  │  (online safe)    │  │  (shared memory) │
    └────────┬───────┘  └────────┬──────────┘  └──────┬───────────┘
             │                   │                    │
    ┌────────▼───────────────────▼────────────────────▼───────────┐
    │                  Shared Memory Primitives                     │
    │  Reduce/Scan △  |  Elementwise △  |  Transpose △  |  Conv1D  │
    └────────────────────────────┬─────────────────────────────────┘
                                 │ 依赖
    ┌────────────────────────────▼─────────────────────────────────┐
    │                   CUDA Core Fundamentals                      │
    │  Thread/Block/Grid  |  Mem H2D/D2H  |  Kernel Launch  |  Sync │
    └───────────────────────────────────────────────────────────────┘
```

图例：△ 标注为 LeetCUDA 中的重点模块，也是 AI Infra 面试高频考点。

---

## 层次详解

### L0 — CUDA 核心基础
| 概念 | LeetCUDA 对应 | 你已完成 |
|------|:------------:|:--------:|
| `threadIdx / blockIdx / blockDim / gridDim` | elementwise | ✅ Puzzle 1-5 |
| `cudaMalloc / cudaMemcpy / cudaFree` | 全部 | ✅ Puzzle 1-3 |
| Kernel launch `<<<grid, block>>>` | 全部 | ✅ |
| `cudaDeviceSynchronize / cudaGetLastError` | 全部 | ✅ |
| Index 与边界检查 | elementwise | ✅ Puzzle 3-5 |
| 2D grid/block 映射 | mat-transpose | ✅ Puzzle 4 |

### L1 — 共享内存与 Block 内协作
| 概念 | LeetCUDA 对应 | 你已完成 |
|------|:------------:|:--------:|
| `__shared__` 声明与加载 | reduce, dot-product | ✅ Puzzle 6-8 |
| `__syncthreads()` 屏障 | reduce, softmax | ✅ |
| Bank conflict 意识 | — | ⬜ 待学习 |
| Block 内 Reduce (树形归约) | reduce | ✅ Puzzle 8/11 |
| Block 内 Prefix Sum (scan) | — | ✅ Puzzle 10 |
| Halo 区域加载 (卷积) | — | ✅ Puzzle 9 |

### L2 — 分块 (Tiling) 与多维 Grid
| 概念 | LeetCUDA 对应 | 你已完成 |
|------|:------------:|:--------:|
| 2D Tiled Matmul (共享内存分块) | sgemm | ✅ Puzzle 12 |
| 双缓冲思想 | — | ⬜ |
| 2D Axis Reduction | reduce | ✅ Puzzle 11 |
| 1D Conv with Halo | — | ✅ Puzzle 9 进阶版 |

### L3 — 基础算子 (LeetCUDA Easy/Medium)
**这一层是进入 LeetCUDA 主力内容的分界线。**

| Kernel | 难度 | 核心知识点 |
|--------|:--:|-----------|
| `elementwise` (relu/gelu/swish/silu/sigmoid) | ★ | 逐元素操作，激活函数 |
| `reduce` (sum/max) | ★ | warp shuffle 归约 |
| `softmax` | ★★ | online safe softmax, 数值稳定性 |
| `layer-norm / rms-norm` | ★★ | reduce + broadcast, warp shuffle |
| `rope` (RoPE 位置编码) | ★★ | 三角函数、复数旋转 |
| `sgemv` (矩阵向量乘) | ★★ | 一维分块 |
| `sgemm` (朴素) | ★★ | 二维分块 + 共享内存 |
| `mat-transpose` | ★ | 合并访问 |
| `histogram` | ★ | atomic 操作 |
| `embedding` | ★ | 查表 |

### L4 — Tensor Cores 入门
| 概念 | LeetCUDA 对应 |
|------|--------------|
| FP16/BF16/TF32 精度体系 | hgemm 前半 |
| WMMA API (`nvcuda::wmma`) | kernels/hgemm |
| MMA PTX 指令 | kernels/hgemm |
| 数据布局 (row-major vs col-major) | hgemm |
| K 维分块循环 | hgemm |

### L5 — HGEMM (高阶 Matmul)
| 概念 | 说明 |
|------|------|
| Tile Block / Tile Warp / Tile Thread | 三级分块粒度 |
| Multi-Stage Pipeline (2~4 stages) | 隐藏访存延迟 |
| Register Double Buffer | 寄存器翻倍缓冲 |
| `cp.async` (异步拷贝) | SM 80+ |
| Shared Memory Padding | 避免 bank conflict |
| Block Swizzle / Warp Swizzle | 提升 L2 cache 命中率 |
| Collective Store (shfl) | warp shuffle 写回 |
| Layout NN / TN | 不同矩阵布局 |

### L6 — FlashAttention (MMA 实现)
| 概念 | 说明 |
|------|------|
| Split KV (FA-1 风格) | 沿 seqlen 分块 |
| Split Q (FA-2 风格) | 沿 headdim 分块 |
| Shared KV / QKV SMEM | 共享内存复用策略 |
| QK Tiling / QKV Tiling | 细粒度分块 |
| Prefetch (g2s / s2r) | prefetch 指令 |
| Causal Mask | 因果注意力遮罩 |

### L7 — CUTLASS 与 CuTe
| 概念 | 说明 |
|------|------|
| CuTe Layout / Tile / Copy | CUTLASS 3.x 抽象 |
| CUTLASS GEMM 模板 | 声明式 kernel 组合 |
| TMA (Tensor Memory Accelerator) | SM 90+ Hopper |

---

## LeetCUDA Kernel 分类速查

```
kernels/
├── elementwise/     ★     relu, gelu, elu, swish, sigmoid, hardswish, hardshrink
├── reduce/          ★     sum, max reduction
├── softmax/         ★★    online safe softmax
├── layer-norm/      ★★    LayerNorm forward
├── rms-norm/        ★★    RMSNorm forward
├── rope/            ★★    RoPE 位置编码
├── sgemv/           ★★    矩阵向量乘
├── sgemm/           ★★    朴素矩阵乘 (shared memory tiling)
├── dot-product/     ★     dot product + reduction
├── mat-transpose/   ★     矩阵转置 (coalesced access)
├── histogram/       ★     atomicAdd 直方图
├── embedding/       ★     词嵌入查表
├── hgemm/           ★★★   半精度矩阵乘 (WMMA/MMA/CuTe)
├── flash-attn/      ★★★   FlashAttention MMA 实现
├── swizzle/         ★★★   swizzle 访存模式
├── gelu/            ★     GELU 激活
├── elu/             ★     ELU 激活
├── hardshrink/      ★     HardShrink 激活
├── hardswish/       ★     HardSwish 激活
├── swish/           ★     Swish 激活
├── sigmoid/         ★     Sigmoid 激活
├── hgemv/           ★★   半精度矩阵向量乘
├── nms/             ★★   Non-Maximum Suppression
├── transformer/     ★★★  Transformer block 骨架
├── cutlass/         ★★★  CUTLASS GEMM 示例
├── openai-triton/   ★★★  Triton 语言对照
├── ws-hgemm/        ★★★  Weight Stationary HGEMM
├── nvidia-nsight/   -     Nsight Systems/Compute 使用
└── notes-v1.cu      -     学习笔记
```

---

## 你的当前位置

```
已完成：L0 ✅  L1 ✅  L2 ✅
进行中：L3（部分算子，但缺少 warp shuffle 版 reduce/softmax 等）
未启动：L4 ~ L7
```

你在 GPU Puzzles 12 个练习 + 部分 Thrust/CUB/stream 实践方面已经打下了不错的 CUDA 基础。下一步关键是：

1. **从 CPU-style reduction 升级到 warp-level reduction**（用 `__shfl_down_sync`）
2. **吃透 softmax / layer-norm / rms-norm**（这是 LLM 推理最高频的算子）
3. **进入 Tensor Cores 的世界**（HGEMM → FlashAttention）

