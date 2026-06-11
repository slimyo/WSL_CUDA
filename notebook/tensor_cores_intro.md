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

---

## 6b. Hopper wgmma — Warpgroup-Level MMA (SM90)

**WMMA 的问题：**
```
wmma::mma_sync 在 warp 内部做 matmul。
Hopper 引入 wgmma = warpgroup（4 warp = 128 threads）级 MMA。
优势：
  - 更大的 tile（如 64×16×16）：单次 matmul 吞吐更高
  - 异步执行：warpgroup 发射后继续执行其他指令
  - 与 TMA 异步加载结合 → pipeline overlap
```

```cuda
// wgmma PTX 伪代码 (Hopper):
// 一个 warpgroup 共 4 warp
// wgmma.fence 确保 SMEM 可见
// wgmma.commit_group 提交异步 matmul
// wgmma.wait_group 等待完成
asm volatile(
    "wgmma.fence.sync.aligned;\n"
    "{\n"
    "    .reg .b64 tma_desc;\n"
    "    // TMA 描述符: 从 global→SMEM 加载 K 和 V 分块\n"
    "    // wgmma 自动从 SMEM 读取数据\n"
    "    wgmma.mma_async.sync.aligned.m64n16k16.f16.f16\n"
    "    {%0,%1,%2,%3,%4,%5,%6,%7},\n"
    "    %8, %9, %10, 1, 1, 0, 0;\n"
    "}"
    : "=r"(o0),"=r"(o1), ...
    : "r"(desc));
```

**LeetCUDA 参考:** `kernels/hgemm/wgmma/` — Hopper wgmma HGEMM 实现

## 7. Blackwell tcgen05 / Tensor Memory (SM100)

### 7.1 为什么 Blackwell 要引入 Tensor Memory

```
Hopper 的限制：
  - wgmma 数据仍从 SMEM 读 → 占用了 SMEM 带宽
  - SMEM 在 Hopper 是共享资源（同时服务加载和计算）
  
Blackwell 的 Tensor Memory (TMEM):
  - 专用 SRAM（仅 Tensor Core 可访问）
  - 不占 SMEM 带宽
  - 更大容量、更高吞吐
```

### 7.2 tcgen05 指令

```
tcgen05 = Tensor Core Generation 05（Blackwell 的第五代 Tensor Core 指令集）

关键特性：
  - accumulator 放在 Tensor Memory 而不是寄存器
    （操作数 A/B 仍可从 SMEM 或 TMEM 读）→ 释放寄存器给其他工作
  - 更大的 tile 粒度（如 128×256）、单条指令由专门硬件调度（不占用 warp 发射槽）
  - 完全异步：发射后 warp 继续干别的，靠 mbarrier 同步
  - 原生 FP4/FP6 支持 (NVFP4)

对比：
  - wgmma  (Hopper):   操作数 SMEM/RF，accumulator 占用大量寄存器
  - tcgen05 (Blackwell): accumulator 进 TMEM，寄存器压力骤降，
    epilogue 由 CUDA core 从 TMEM 读出处理
```

### 7.3 tcgen05 使用模式

```cuda
// tcgen05 伪代码 (概念层面):
// 1. TMA descriptor 配置: global → Tensor Memory
// 2. tcgen05_async_mma: 在 Tensor Memory 上做异步 matmul
// 3. tcgen05_wait: 等待完成
// 4. 结果直接被后续 Tensor Core 指令消费

// 核心优势：TMEM 到 Tensor Core 的路径是 SMEM 到 Register 的 ~2× 带宽
// 且 TMEM 访问不竞争 SMEM 端口 → 可以同时做 SMEM 相关操作
```

**Blackwell FA4 选择 tcgen05 + CuTe 的原因：**
- CuTe 的 layout 抽象让 TMEM 上的数据布局管理更简洁
- tcgen05 的异步 MMA 需要复杂的 pipeline 管理，CuTe 的 TiledMMA 可以表达

## 8. Tensor Core 各代对比总结

| 架构 | 代 | 新增格式 | API | Tile 粒度 | 关键特性 |
|------|:---:|------|------|:---:|------|
| Volta | SM70 | FP16 | WMMA | 16×16×16 | 第一代 |
| Turing | SM75 | INT8/INT4 | WMMA | 16×16×16 | 整数量化 |
| Ampere | SM80 | TF32/BF16/FP64 | WMMA+MMA PTX | 16×8×16 | cp.async |
| Hopper | SM90 | FP8 | MMA PTX+wgmma | 64×16×16 | TMA, Warp Specialization |
| Blackwell | SM100 | NVFP4 | tcgen05 | 64×64 | Tensor Memory, FP4 |

## 9. LeetCUDA Tensor Core 源码索引

| LeetCUDA 目录 | Tensor Core 类型 |
|------|------|
| `hgemm/wmma/` | WMMA (SM70+) — 最基础 |
| `hgemm/mma/` | MMA PTX (SM80+) — 精确控制 |
| `hgemm/wgmma/` | wgmma (SM90 Hopper) |
| `ws-hgemm/` | Warp Specialization wgmma |
| `sgemm/sgemm_wmma_tf32_stage.cu` | WMMA TF32 + Multi-Stage |
| `flash-attn/mma/` | MMA-based FlashAttention |
| `cutlass/cute_dsl/` | CuTe DSL 实现 |

## 10. 学习检查清单（补充）

- [ ] 理解 wgmma 和 WMMA 的区别（warp vs warpgroup）
- [ ] 理解 Tensor Memory 和 Shared Memory 的区别
- [ ] 能讲清 Hopper TMA + wgmma 的 pipeline 架构
- [ ] 能讲清 Blackwell tcgen05 为什么比 wgmma 更高效
- [ ] 能在 LeetCUDA 中找到对应 Tensor Core 版本的源码
