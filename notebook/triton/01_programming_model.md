# Triton 01 · 编程模型：从 thread 思维切换到 tile 思维

> 目标: 写出第一个 kernel，彻底理解 program/grid/mask 三件套
> 时长: 1-2 天 ｜ 前置: 02_cuda_programming_model.md（用 CUDA 概念做对照）

---

## 1. 核心概念：SPMD over tiles

Triton kernel 是 **SPMD 程序**：同一段代码被启动 N 个 **program** 实例，
每个 program 用 `tl.program_id()` 知道自己是谁、负责哪块数据。
program ≈ CUDA 的 thread block，但你**不再写 block 内部的线程逻辑**——
你写的是"这一整块 tile 怎么算"，向量化语义，编译器去分线程。

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr,
               n,                          # 运行时标量参数
               BLOCK: tl.constexpr):       # constexpr: 编译期常量(每个值编译一个版本)
    pid  = tl.program_id(axis=0)           # 我是几号 program
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # [BLOCK] 个偏移——这是一个"向量"
    mask = offs < n                            # 越界保护（替代 CUDA 的 if）
    x = tl.load(x_ptr + offs, mask=mask)       # 向量化 load（自动合并访存）
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)   # grid 可依赖编译参数
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out

x = torch.randn(10_000_00, device='cuda', dtype=torch.float16)
y = torch.randn_like(x)
torch.testing.assert_close(add(x, y), x + y)   # 第一关：先对
```

**逐行对照 CUDA：**

| Triton | CUDA 等价物 | 注意 |
|------|------|------|
| `tl.program_id(0)` | `blockIdx.x` | 最多 3 个 axis |
| `tl.arange(0, BLOCK)` | `threadIdx.x` + 网格跨步循环 | BLOCK 必须是 2 的幂 |
| `mask=offs<n` | `if (idx < n)` | load 的 `other=` 给 mask 外的填充值 |
| `tl.load(ptr+offs)` | 合并访存 + 向量化 ld | 编译器自动选 LDG.128 |
| 没有 `__syncthreads` | 手动同步 | 编译器从数据依赖推导插入 |
| `BLOCK: tl.constexpr` | template 参数/宏 | 改变它会触发重新编译（JIT 缓存按值区分） |

## 2. 必须想清楚的三个问题（每写一个 kernel 都问自己）

```
① 并行切分：grid 上每个 program 负责输出的哪一块？
   （elementwise: 1D 均分；矩阵: 2D tile；softmax: 每行一个 program）
② tile 内表达：这块计算如何用 tl 的向量/矩阵操作写出（避免标量循环）？
③ 边界与 mask：哪些维度可能不整除？mask 谁、填什么 other 值？
   （sum 填 0，max 填 -inf —— 填错是隐蔽 bug 之王）
```

## 3. 多维与 stride：处理矩阵的标准姿势

Triton 操作的是指针 + 偏移，**stride 显式传入**（支持非连续 tensor）：

```python
@triton.jit
def copy_2d(x_ptr, o_ptr, M, N, sxm, sxn, som, son,
            BM: tl.constexpr, BN: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)            # [BM]
    rn = pid_n * BN + tl.arange(0, BN)            # [BN]
    # 广播出 [BM, BN] 的二维偏移——Triton 的 numpy 式广播是核心表达力
    offs = rm[:, None] * sxm + rn[None, :] * sxn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tile = tl.load(x_ptr + offs, mask=mask)
    tl.store(o_ptr + rm[:, None] * som + rn[None, :] * son, tile, mask=mask)
```
`x.stride(0), x.stride(1)` 从 PyTorch 侧传进来。**记住模式：
`rm[:, None]*stride_m + rn[None, :]*stride_n` —— 后面 matmul/attention 全是它。**

## 4. 性能初体验：和 PyTorch 比带宽

```python
@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['n'], x_vals=[2**i for i in range(16, 26)],
    line_arg='provider', line_vals=['triton', 'torch'], line_names=['Triton','Torch'],
    ylabel='GB/s', plot_name='add-bw', args={}))
def bench(n, provider):
    x = torch.randn(n, device='cuda', dtype=torch.float16); y = torch.randn_like(x)
    fn = (lambda: add(x, y)) if provider == 'triton' else (lambda: x + y)
    ms = triton.testing.do_bench(fn)
    return 3 * n * 2 / ms * 1e-6     # 读x读y写out = 3n×2B → GB/s
bench.run(print_data=True)
```
**验收：大尺寸时两者都应逼近 ~300 GB/s（336 的 90%）。**没到就用 P1 的 SOP 查
（多半是尺寸太小 launch 开销主导——顺手观察小尺寸时 Triton 略输 torch，
因为 JIT kernel 的 Python 启动开销，这就是 05 篇 cache 话题的引子）。

## 5. 本篇练习（验收标准见 06 篇）

1. `fused_axpby`: out = a*x + b*y（标量 a,b 作为参数传入）
2. `cast_kernel`: fp32→fp16 转换，比较与 `.half()` 的带宽
3. `bias_gelu`: out = gelu(x + bias)，bias 沿最后一维广播——第一次用 2D 偏移
4. 把 BLOCK 从 64 扫到 4096，画"带宽 vs BLOCK"曲线，解释两端为什么差
   （小：program 太多+每个太轻；大：occupancy 受限——能讲清就真懂了）

## 6. 本篇要点回顾（面试快答）

- Triton 把 CUDA 的"线程级 SIMT"抬升为"tile 级 SPMD"，砍掉的是 thread/smem/sync
  的人工管理，保留的是分块策略和数据流设计（这才是性能的大头）
- mask + other 是边界处理的全部；归约场景 other 的取值要匹配归约恒等元
- stride 显式化使 kernel 天然支持转置/切片后的张量
