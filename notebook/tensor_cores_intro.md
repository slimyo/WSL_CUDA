# Tensor Cores 精度体系与 WMMA 入门

> TODO.md: 阶段3 — AI 推理核心
> 前置: `reduce_warp_learning.md`, `bank_conflict_learning.md`
> 参考: `third_party/LeetCUDA/kernels/hgemm/`

## 1. 精度体系总览

| 格式 | bits | exp | mantissa | range | 用途 |
|------|:---:|:---:|:---:|------|------|
| FP32 | 32 | 8 | 23 | ~3.4e38 | 训练基准 |
| TF32 | 19 | 8 | 10 | 同 FP32 | A100 Tensor Core 内部格式 |
| BF16 | 16 | 8 | 7 | 同 FP32 | 训练 (range=FP32) |
| FP16 | 16 | 5 | 10 | ~65504 | 推理 (注意 range 仅 65504) |
| FP8 E4M3 | 8 | 4 | 3 | ~448 | H100 推理 |
| FP8 E5M2 | 8 | 5 | 2 | ~57344 | H100 训练 |

**关键**: BF16 和 FP32 range 相同 (都 8-bit exponent), 所以 BF16 训练不会 overflow。
FP16 range 只有 65504, Adam 优化器等场景容易 overflow 需要 loss scaling。

## 2. Tensor Cores 原理

```
D = A × B + C   (矩阵乘加 fused)
```

一个 Tensor Core 一个 cycle 完成 tile 乘加 (如 m16n8k16), 比 CUDA Core 快 8-16×。

各代 GPU Tensor Core 能力:
- SM70 (V100): FP16
- SM75 (Turing): + INT8, INT4
- SM80 (A100): + TF32, BF16, FP64
- SM90 (H100): + FP8

## 3. WMMA API (高层)

```cuda
#include <mma.h>
using namespace nvcuda;

// fragment = 寄存器中存的小矩阵片
wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16,16,16, half> c_frag;

// 加载 shared/global memory → fragment
wmma::load_matrix_sync(a_frag, A_ptr, lda);
wmma::load_matrix_sync(b_frag, B_ptr, ldb);

// 矩阵乘 (在 fragment 上)
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// 存回 memory
wmma::store_matrix_sync(C_ptr, c_frag, ldc, wmma::mem_row_major);
```

Tile 大小: M×N×K 可选 16×16×16 或 32×8×16 或 8×32×16。

## 4. WMMA → MMA PTX 升级

WMMA 是封装好的 API, MMA PTX 是底层 inline assembly, 更灵活:

```cuda
// MMA PTX: m16n8k16, row-major A, col-major B, FP16 input/output
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
    : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
      "r"(b0),"r"(b1),
      "r"(c0),"r"(c1),"r"(c2),"r"(c3));
```

MMA PTX 优势: 可控精确布局, interleaved 指令, 自定义 tile, 更好的 register 控制。

## 5. 学习路线

1. 理解 FP16/BF16/TF32 精度差异
2. WMMA 写一个简单 HGEMM (参考 `kernels/hgemm/`)
3. 升级到 MMA PTX
4. 对比 cuBLAS 性能

## 6. 推荐阅读

1. `kernels/hgemm/` WMMA 版本 — 最先读
2. `kernels/hgemm/` MMA 版本 — 理解 PTX inline asm
3. NVIDIA Tensor Core 白皮书

