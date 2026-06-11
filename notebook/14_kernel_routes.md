# Kernel 实现路线：CUDA / CUTLASS / Triton / Compiler

> 对象: 算子岗位（必达 L3）
> 前置: 06_roofline_and_flops.md, 01_gpu_hardware_architecture.md
> 目标: 面试能针对具体场景论证选什么工具链、解释 CuTe layout 和 epilogue fusion
> 参考 LeetCUDA: `cutlass/`, `openai-triton/`, `sgemm/`, `hgemm/`, `flash-attn/`

---

## 1. 工具链选择轴

**核心 trade-off：控制力/性能上限 ↔ 开发效率/可维护性**

```
手写 CUDA  PTX         ← 极致控制、性能上限最高
        ↑
CUTLASS / CuTe          ← 模板化 GEMM + layout 抽象
        ↑
Triton (DSL)            ← tile 级编程、开发效率高
        ↑
TVM / MLIR / IREE       ← 编译器可移植
torch.compile / TRT     ← 自动优化
```

**面试高频题："给你一个 attention kernel，用哪个工具链？"**
答案不是一个名字，是**沿着这条轴论证**：

| 场景 | 推荐 | 理由 |
|------|------|------|
| 通用 flash attention（如 FlashInfer） | CUTLASS/CuTe | 性能近手写、可维护、社区活跃 |
| 实验性 fused kernel 验证 | Triton | 开发快、autotune 省事 |
| 极致性能（LLM 核心 kernel） | 手写 CUDA/PTX | Full control、warp specialization |
| 量产部署的 engine 构建 | TRT-LLM / torch.compile | 自动优化 + 支持多硬件 |
| 跨平台部署 | TVM / IREE / MLIR | Compiler 层的可移植 |

---

## 2. 手写 CUDA

**LeetCUDA 的全面覆盖：**

| LeetCUDA 目录 | 类型 | 优化层次 |
|------|------|------|
| `sgemm/` | FP32 GEMM | naive → tiled → vectorized → WMMA TF32 |
| `hgemm/` | FP16/BF16 GEMM | WMMA → MMA PTX → Multi-Stage → wgmma |
| `ws-hgemm/` | Hopper Warp Specialization | Producer/Consumer wgmma |
| `elementwise/` | Elementwise | naive → vectorized → 128-bit |
| `flash-attn/` | FlashAttention | MMA PTX based |
| `rope/` | RoPE | f32 → f32x4 vectorized |
| `gelu/` | GeLU | f32 → f16 → f16x2 → f16x8 → pack |
| `layer-norm/` | LayerNorm | f32 → f32x4 → f16x8 pack |

**典型手写 CUDA 优化路径（以 SGEMM 为例）：**
```
sgemm.cu:
  1. Naive → 每个 thread 1 个元素, 无分块
  2. Tiled → shared memory 分块, 减少 global 访存
  3. Vectorized → float4 load/store, 减少指令数
  4. Bank conflict 消解 → padding

sgemm_wmma_tf32_stage.cu:
  5. WMMA Tensor Core → TF32 mma_sync
  6. Multi-stage pipeline → cp.async
```

---

## 3. CUTLASS / CuTe

### 3.1 CUTLASS 是什么

```
CUTLASS = CUDA Templates for Linear Algebra Subroutines
NVIDIA 官方维护的 GEMM / attention 模板库

核心构件：
  - Threadblock 级别的 tile 调度
  - Warp 级别的矩阵乘
  - Epilogue（GEMM 后融合 bias/activation/layernorm）
```

### 3.2 CuTe (CUTLASS 3.x 的 layout 抽象)

**CuTe 解决了什么？**
```
传统 GEMM 代码里：
  - 手动计算 A[tile.m][tile.k] 到 A[shared_idx] 的地址
  - 需要知道 row_major/col_major、stride、padding
  - 不同 layout 需要不同代码

CuTe 用 Layout 抽象：
  Layout<Shape, Stride> → 压缩的元组描述
  - Shape = (M, N, K) 或 (rows, cols)
  - Stride = (stride_row, stride_col)
  - Tensor<T, Layout> → 自动推导访存地址

示例：
  // col-major (M,N) 矩阵：行内 stride=1，跨列 stride=M
  auto col_major = make_layout(make_shape(M, N), make_stride(_1{}, M));
  // row-major: 跨行 stride=N，行内 stride=1
  auto row_major = make_layout(make_shape(M, N), make_stride(N, _1{}));
  Tensor t = make_tensor(ptr, col_major);   // t(i,j) 自动算地址
  // Layout 可组合/可分块：local_tile(t, tile_shape, coord) 直接切 tile，
  // swizzle 也表达为 layout 变换——这是"代数化"的地址计算
```

**CuTe 的核心概念：**
- `Layout`：Shape + Stride 的复合描述
- `Tensor`：数据 + Layout 的绑定
- `TiledCopy`：异步拷贝的 tile 描述
- `TiledMMA`：矩阵乘的 tile 描述

**FlashAttention-4 选用 CuTe 的原因：**
- CuTe 的 layout 抽象使 Blackwell 的 Tensor Memory/新布局支持更简洁
- 不在手写 PTX 中重复 layout 处理逻辑
- 可组合（不同 tile 大小/数据类型 swizzle 模式切换）

### 3.3 Epilogue Fusion（频繁面试）

```
标准 GEMM: C = A × B + bias → 写回 HBM → 读回 → 做 activation → 写回 HBM

Epilogue Fusion：GEMM 结果直接留在 SMEM/寄存器件，加 bias + activation 后再写

CUTLASS 的 epilogue 能做：
  - C = A × B + C        (矩阵乘加)
  - C = A × B + bias     (加 bias)
  - C = act(A × B + bias)(+ ReLU / GELU / SiLU)
  - C = A × B + C × scale (rescale)
  - C = dequant(A × B)   (反量化 + GEMM 融合)

为什么融合 = 减少 HBM 往返：
  一次 HBM 写入 ≈ ~200 cycles
  一次 HBM 读取 ≈ ~200 cycles
  融合后：所有操作 + 写回 = 1 次写入 vs un-fused 的额外 2+ 次 R/W
```

---

## 4. Triton (DSL)

### 4.1 核心概念

```
Triton 让你写 tile（block）级代码，不需要担心 thread/SMEM/barrier：

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                       # block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # tile 内地址
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)           # tile load
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask) # tile store
```

### 4.2 Triton vs 手写 CUDA 的取舍

| 方面 | Triton | 手写 CUDA |
|------|------|------|
| 开发效率 | 高（10 行完成 flash attention） | 低 |
| 性能上限 | ~80-90% of hand-tuned | 100%（可以有极致优化） |
| SMEM 控制 | 编译器自动分配 | 完全手动 |
| Pipeline 控制 | `num_stages` 粗粒度控制 | 完全控制 cp.async/TMA |
| Warp Specialization | 2025 起 Triton 3.x 在 Hopper/Blackwell 上支持自动 WS；更底层控制有 Gluon 方言 | 完全手动控制 |
| 精度/类型 | 支持各种 dtype | 完全控制 |
| Autotuning | 内置 autotune | 需手动调参 |

**典型结论：**
```
"Triton 帮你省了 thread 管理、SMEM 分片、warp 同步的体力活，
但你放弃了 warp specialization、精确的寄存器分配和 pipeline 深度控制。
对有清晰 tile 边界（attention, norm, elementwise）的 kernel，Triton 是首选。
对 warp specialization（FA v3）、复杂 pipeline，手写 CUDA 仍必要。"
```

### 4.3 LeetCUDA Triton 示例

| LeetCUDA 目录 | 内容 |
|------|------|
| `openai-triton/matrix-multiplication/` | Matmul kernel |
| `openai-triton/fused-attention/` | Flash attention |
| `openai-triton/fused-softmax/` | Safe softmax |
| `openai-triton/layer-norm/` | Fused layer norm |
| `openai-triton/merge-attn-states/` | Attention states 合并 |
| `openai-triton/vector-add/` | Vector add |

---

## 5. 编译器路线（了解）

```
TVM / Relax / MLIR / IREE / torch.compile / TensorRT

适用范围：
  - 跨平台部署（CPU / GPU / NPU / 手机）
  - 自动图融合（多个 kernel 合并为 1 个）
  - 算子自动调优

局限性：
  - 前沿优化（warp specialization 等）编译器不一定能自动发现
  - 手动调优 kernel 可能优于编译器自动产出
  - 调试困难
```

---

## 6. 算子融合（日常核心工作）

### 6.1 融合动机

```
例：add + relu（Elementwise Fusion），数组长度 N

不融合（精确数访存）:
  kernel 1: z = x + y      → 读 x, 读 y, 写 z   = 3N
  kernel 2: out = relu(z)  → 读 z, 写 out       = 2N
  合计 5N 字节 HBM 流量 + 2 次 kernel launch

融合:
  kernel 1: out = relu(x + y) → 读 x, 读 y, 写 out = 3N
  合计 3N 字节 + 1 次 launch

收益：HBM 流量 5N→3N（1.67× 提速上限），launch 减半。
被融掉的中间结果 z 全程活在寄存器里，从不落 HBM——这就是 fusion 的本质。
链越长收益越大：k 个 elementwise 串联，不融合 ≈ (2k+1)N，融合恒为 3N。
```

### 6.2 LLM 中的常见融合

| 融合模式 | 典型场景 | 收益 |
|------|------|:---:|
| bias + activation | GEMM→bias→ReLU/GELU | 减少 2 次 HBM R/W |
| RMSNorm + matmul | Pre-norm decoder | 减少 3 次 HBM R/W |
| RoPE fused into attention | Q 加载时直接旋转 | 减少 1 次 HBM W/R |
| Dequant + GEMM | 量化模型 | 避免反量化整张权重写回 HBM |

---

## 7. 学习检查清单

- [ ] 能画工具链选择轴，给定场景选推荐方案并论证
- [ ] 能解释 CuTe 的 layout 抽象解决了什么
- [ ] 能说清 epilogue fusion 是什么、能 fused 什么
- [ ] 能解释 Triton 帮你省了什么控制权、换来了什么
- [ ] 能说清算子融合减少了几次 HBM 往返（给具体例子）
- [ ] 能在 LeetCUDA 中找到对应的 CUDA/CUTLASS/Triton 源码

---

## 8. 自测 / 面试题

1. 为什么 elementwise 算子（如 add+relu）一定要融合？不融合多花在哪？
2. Triton 相比手写 CUDA，你放弃了什么控制权、换来了什么？
3. CUTLASS 的 epilogue 能做什么？为什么 FlashAttention-4 选 CuTe？
4. 给你一个 fused RMSNorm + matmul 的 kernel，你选 Triton 还是 CUDA 为什么？
5. 写一个 Triton fused add+relu kernel 伪代码。

---

## 9. 推荐阅读

| 资料 | 来源 |
|------|------|
| CUTLASS / CuTe 文档 | NVIDIA / GitHub |
| Triton Language Tutorial | OpenAI Triton Docs |
| OpenAI Triton: A DSL for Neural Networks (2021) | Tillet et al. |
| FlashInfer: Customizable Attention Engine (MLSys'25 最佳论文) | Zihao Ye et al., UW + CMU + NVIDIA |
| LLM 中的算子融合 | 各类技术博客 |
| LeetCUDA cutlass/ openai-triton/ | `/third_party/LeetCUDA/kernels/` |
