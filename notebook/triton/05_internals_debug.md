# Triton 05 · 编译器内幕、调试与性能排查

> 目标: 知道 Triton 把你的 Python 变成了什么、kernel 错了/慢了怎么查
> 时长: 1-2 天，之后当手册随用随查 ｜ 前置: 01-03 篇；19 章 §3（LLVM/MLIR 定位）

---

## 1. 编译管线：从 Python 到 SASS

```
@triton.jit Python AST
   ↓ 前端解析
TTIR   (Triton IR, MLIR 方言)        ← 与硬件无关的 tile 操作
   ↓ layout 推导/分块决策
TTGIR  (Triton GPU IR, MLIR 方言)    ← 关键一层! 决定 tile→warp/线程映射、
   ↓ lower                              smem 使用、流水 stage —— 你在 CUDA 里
LLVM IR                                 手写的东西都在这层"长出来"
   ↓ LLVM NVPTX 后端
PTX  → (driver JIT) →  SASS
```
这就是 19 章说"Triton 基于 LLVM/MLIR"的具体含义：TTIR/TTGIR 是两个自定义
MLIR dialect，优化 pass 跑在 MLIR 框架上，最后借 LLVM 出 PTX。

**查看每一层产物（面试演示级技能）：**
```python
h = matmul_kernel[grid](...)            # launch 返回 CompiledKernel
print(h.asm.keys())                     # dict: ttir / ttgir / llir / ptx / cubin
print(h.asm['ttgir'])                   # 看 smem 分配、流水安排
print(h.asm['ptx'])                     # 确认 mma.sync 指令在不在（tensor core 证据!）
print(h.n_regs, h.metadata.shared)      # 寄存器/共享内存用量 → occupancy 分析
```
缓存：编译产物在 `~/.triton/cache`（改 kernel 不生效先怀疑它，`rm -rf` 解决）。
环境变量：`TRITON_KERNEL_DUMP=1`（落盘全部 IR）、`MLIR_ENABLE_DUMP=1`（看 pass 过程）。

## 2. 正确性调试工具箱（按代价从小到大）

```
① tl.device_print("s=", s)         # kernel 内打印（小 grid 时用，输出巨大）
② TRITON_INTERPRET=1 python x.py   # ★解释器模式: kernel 在 CPU/numpy 上模拟执行
                                    #   可用 pdb 断点、print tile 的具体数值
                                    #   数值 bug 的首选武器（快查 mask/边界/rescale）
③ 二分缩小: 固定 BLOCK=16、N=17 这类畸形小形状 + 和 numpy 参考逐 tile 对比
④ tl.static_print / tl.static_assert  # 编译期检查 constexpr 逻辑
```
经验法则：**先 INTERPRET 调对，再回 GPU 调快**。两个模式数值有微小差异
（fp16 舍入路径不同），verify 用 1e-2 容差。

## 3. 性能排查决策树（接 P1 的 SOP）

```
慢 → 先 nsys: 时间真花在 kernel 上吗?
  ├─ Python/launch 开销大（小 kernel 高频调用）
  │    → 静态 grid + 缓存 kernel 句柄；考虑 CUDA Graph 包起来（17 章）
  ├─ autotune 没选好 → TRITON_PRINT_AUTOTUNING=1 查; 给该形状加专用 config
  └─ kernel 本身慢 → ncu（Triton kernel 名形如 matmul_kernel_0d1d...）:
       ├─ 没用上 tensor core: ptx 里搜 mma → 没有? 检查 tl.dot 输入 dtype/维度
       ├─ 寄存器溢出: h.n_regs > 128 或 ncu 报 local memory
       │    → 减小 BLOCK / 拆 kernel / 减少活跃中间量
       ├─ 访存不合并: 检查最快变化维的 stride 是不是 1; 加 tl.max_contiguous /
       │    tl.multiple_of 提示帮编译器向量化
       └─ occupancy 低: num_warps 调小（寄存器换并发）或 BLOCK 调小
```

**Triton 性能反模式清单（写代码时自查）：**
```
✗ 用广播乘+sum 代替 tl.dot           → CUDA core, 慢 10×
✗ 循环里 load 不变量（如 scale）      → 提到循环外
✗ BLOCK 维度过大导致单 program 太重   → SM 数 30, grid 至少 ≥ 60 才喂得满
✗ 频繁的 .to() 转换穿插在热循环       → 集中转换
✗ mask 计算放循环内但其实循环不变      → 提出去
```

## 4. 与 CUDA 互操作 / 接入 PyTorch

```
- Triton kernel 直接吃 torch tensor（传 data_ptr 由 jit 处理），零拷贝
- 注册为 PyTorch 自定义算子: torch.library.custom_op（让 torch.compile 不打断图）
- 反向: 用 torch.autograd.Function 包 fwd/bwd 两个 triton kernel
- 看 torch.compile 生成的 Triton 代码: TORCH_LOGS=output_code python train.py
  ——学完本教程你能读懂 Inductor 的产物，这是"理解 PyTorch 编译栈"的捷径
```

## 5. 能力边界：什么时候放弃 Triton 回 CUDA/CUTLASS

```
适合 Triton: elementwise/归约/norm/attention/常规 GEMM 变体/dequant —— 90% 日常
不适合:
  - 极小 GEMV（M<16, tl.dot 浪费）→ 手写 CUDA warp 级
  - 需要精确 warp 间编排（生产者-消费者、persistent kernel 高级用法）
  - 跨 block 细粒度同步/通信类算法
  - 追最后 10% 且形状固定 → CUTLASS/手写（14 章选型轴的右端）
2025 后的中间选项: Gluon（Triton 低层方言）、CuTe-DSL —— 见 18 章 §5.1
```

## 6. 本篇练习

1. 取 03 篇 matmul，打印 ttgir，找到 smem 分配和 stage 数的证据
2. 故意把 tl.dot 换成广播乘加，对比 ptx（mma 消失）和性能（慢一个量级）
3. 用 TRITON_INTERPRET 给 04 篇 flash attention 注入一个 rescale bug 并调出来
4. 用 `TORCH_LOGS=output_code` 看 `torch.compile(torch.nn.RMSNorm)` 生成的
   kernel，和你 02 篇手写版对比结构差异
