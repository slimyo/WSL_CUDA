# OpenAI Triton 完整教程 · 总览

> 为什么单开目录：ROUTE.md 把 Triton 定为 🔴L3（必须会写）——它是算子岗现场手写
> 概率最高的工具（白板写 CUDA 太慢，面试官普遍接受 Triton），也是 OpenAI/Meta/字节
> 等内部的生产工具、torch.compile 的 GPU 代码生成后端。
> 本教程目标：**2 周内做到不看资料写出 fused softmax / matmul / flash attention。**

---

## 0. Triton 是什么（一段话版本）

Python 风格的 GPU kernel DSL + 编译器：你以 **block(tile) 为单位**描述计算
（"这个 program 处理哪块数据、做什么运算"），编译器负责 CUDA 里最磨人的部分——
线程映射、shared memory 管理、访存合并、软件流水。写出 80-95% 手写性能的 kernel，
代码量是 CUDA 的 1/5。**注意与 NVIDIA Triton Inference Server 无任何关系**（19 章 §3）。

## 1. 环境安装（你的机器：RTX 2060 / WSL2 / CUDA 13.2）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # torch 自带配套 triton
python -c "import triton; print(triton.__version__)"   # 3.x 即可
python -c "import torch; print(torch.cuda.get_device_capability())"  # (7, 5)
```

**SM75 (Turing) 注意事项（贯穿全教程）：**
```
✓ fp16 的 tl.dot（tensor core）       ✗ bf16（SM80+）——所有示例用 torch.float16
✓ 全部 tl.* 核心算子                  ✗ num_stages>2 的收益（无 cp.async，保持 =2）
✓ TRITON_INTERPRET 调试               ✗ TMA/warp specialization 相关新特性（SM90）
```

## 2. 课程结构（每篇 1-3 天，配练习）

| 篇 | 内容 | 你将会写 | 对接 |
|----|------|------|------|
| [01 编程模型](01_programming_model.md) | program/grid、load/store/mask、和 CUDA 的对照 | vector add、复制类 kernel | 02 章(CUDA 模型) |
| [02 Fused Softmax 与归约](02_fused_softmax.md) | 行级归约、num_warps、融合 | fused softmax、RMSNorm | softmax_learning |
| [03 Matmul 与 Autotune](03_matmul_autotune.md) | tl.dot、分块、L2 swizzle、autotune | 达 ~90% cuBLAS 的 matmul | P2、hgemm_opt |
| [04 Flash Attention](04_flash_attention.md) | 在线 softmax 落地、causal、GQA | flash fwd（面试默写件） | P3、10 章 |
| [05 编译器内幕与调试](05_internals_debug.md) | TTIR/TTGIR/PTX 管线、调试/性能工具 | 会看中间产物、会定位慢因 | 19 章 §3、P1 |
| [06 练习题集](06_exercises.md) | 12 道阶梯练习（带验收标准） | RoPE/GeLU/dequant 等 | P5 |

**学习路径：01→02→03 是地基（必须顺序做）；04 是目标；05 随用随查；06 穿插练。**

## 3. 心智模型速查（学之前先种下，03 篇详细展开）

```
CUDA:   你管理 thread（threadIdx 算地址、__syncthreads、smem 手动分块）
Triton: 你管理 block（一个 "program" = 一个 tile 的全部逻辑），
        thread 不存在了——tl.arange 生成的"块内偏移"由编译器映射到线程

对应关系:
  CUDA blockIdx          ↔ tl.program_id(axis)
  CUDA threadIdx + 循环   ↔ tl.arange(0, BLOCK) 向量化表达
  CUDA __shared__ + sync ↔ 不写！编译器从数据流推导
  CUDA 边界 if           ↔ mask 参数
  CUDA 调 grid/block dim ↔ grid 你定，block 内部编译器定(num_warps 仅提示)
```

## 4. 全程通用的 benchmark / 验证模板

```python
import torch, triton

def verify(out, ref, name, atol=1e-2, rtol=1e-2):
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    print(f"[{name}] {'PASS' if ok else 'FAIL'} max_err={(out-ref).abs().max():.2e}")
    assert ok

ms = triton.testing.do_bench(lambda: my_kernel_call(...))   # 自带 warmup+中位数
gbps = total_bytes / ms * 1e-6      # 带宽利用率: /336 GB/s (RTX 2060)
tflops = total_flops / ms * 1e-9
```
铁律与 P0 一致：先 verify 再谈性能；报告带宽/算力利用率而非裸毫秒。

## 5. 参考资料指引

- 官方 tutorials（triton-lang.org）：01-vector-add / 02-fused-softmax /
  03-matrix-multiplication / 06-fused-attention —— 本教程的骨架来源，
  但**本教程做了 SM75 适配 + 面试导向重组，先学这里再回看官方**
- LeetCUDA: `kernels/openai-triton/`（vector-add/fused-softmax/layer-norm/
  matrix-multiplication/fused-attention/merge-attn-states 全套可对照）
- GPU MODE (前 CUDA MODE) lecture 14: Triton 实践经验
- 论文: Tillet et al., "Triton: An Intermediate Language and Compiler for
  Tiled Neural Network Computations" (MAPL 2019)
- notebook: 14 章（选型轴）、19 章 §3（生态定位）
