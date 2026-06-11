# Triton 03 · Matmul、tl.dot 与 Autotune

> 目标: 写出 ~90% cuBLAS 的 fp16 matmul；掌握 autotune 和 L2 友好的 program 排序
> 时长: 2-3 天 ｜ 前置: 01/02 篇、hgemm_optimization.md（优化概念）、P2 经验更佳

---

## 1. 全图：matmul kernel 的三层结构

```
grid 层   : 每个 program 算 C 的一个 [BM, BN] tile（2D 切分 → 1D pid 重排, §3）
循环层    : 沿 K 以 BK 步进，load A tile [BM,BK]、B tile [BK,BN]
计算层    : acc += tl.dot(a, b)   ← tensor core 在这里，acc 必须 fp32
```
对照 P2：你在 CUDA 里手写的 smem 分块、双缓冲、bank conflict 处理，
这里全部由编译器从 `tl.dot` + 循环结构推导生成——**你只保留分块策略的决策权。**

## 2. 完整实现（SM75 适配版）

```python
@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GROUP': 8}, num_warps=8, num_stages=2),
        triton.Config({'BM': 128, 'BN':  64, 'BK': 32, 'GROUP': 8}, num_warps=4, num_stages=2),
        triton.Config({'BM':  64, 'BN': 128, 'BK': 32, 'GROUP': 8}, num_warps=4, num_stages=2),
        triton.Config({'BM':  64, 'BN':  64, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=2),
    ],                                    # SM75: num_stages 固定 2（无 cp.async）
    key=['M', 'N', 'K'],                  # 形状变 → 重新挑选最优 config
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GROUP: tl.constexpr):
    # ---- §3 的 L2 友好重排：把 1D pid 映射成"列优先的行组" ----
    pid = tl.program_id(0)
    num_pid_m, num_pid_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    num_pid_in_group = GROUP * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(num_pid_m - first_pid_m, GROUP)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ---- 指针与累加 ----
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    A_ptrs = A + rm[:, None] * sam + rk[None, :] * sak
    B_ptrs = B + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)            # ★fp32 累加
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(A_ptrs, mask=rk[None, :] < K - k*BK, other=0.)
        b = tl.load(B_ptrs, mask=rk[:, None] < K - k*BK, other=0.)
        acc = tl.dot(a, b, acc)                           # ★tensor core
        A_ptrs += BK * sak
        B_ptrs += BK * sbk
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C + rm[:, None]*scm + rn[None, :]*scn, acc.to(tl.float16), mask=c_mask)

def matmul(a, b):
    M, K = a.shape; K2, N = b.shape
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BM']) * triton.cdiv(N, meta['BN']),)
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
```

## 3. 三个关键机制拆解

**① `tl.dot`（性能的全部来源）**
- 输入两个 2D tile（维度 ≥16），fp16 进、fp32 累加出 → 编译为 mma 指令
- 不用 tl.dot 而用广播乘加（`a[:,:,None]*b[None,:,:]` 求和）= 走 CUDA core，慢 10×+
  ——这是 Triton 初学者第一性能事故

**② Grouped ordering（L2 cache 的 swizzle）**
- 朴素行优先发射 program：算完一整行 C 才换行 → B 的列被反复从 HBM 拉
- GROUP 重排让相邻发射的 program 集中在 [GROUP × num_pid_n] 的窗口里
  → 它们共享的 A 行/B 列大概率还在 L2
- **等价于 P2 里的 block swizzle**，但在 Triton 里是 10 行 pid 算术
- 验收实验：GROUP=1 vs 8，4096³ 下测 L2 命中率（ncu `lts__t_sector_hit_rate`）

**③ Autotune**
- 对每个 `key` 形状第一次调用时，把 configs 全跑一遍记住最快的（有缓存）
- 为什么必须有：最优 tile 取决于形状×硬件，没有万能解（P2 你已体会）
- `TRITON_PRINT_AUTOTUNING=1` 看选了哪个；decode 形状（M=1~16）和方阵选的
  config 截然不同——把两者都打出来贴报告
- 坑：bench 时第一次调用含 tuning 时间，计时前先 warmup 一次

## 4. 验收

1. 正确性 vs `torch.matmul`（fp16, atol=1e-1 量级——fp16 大 K 累加误差正常）
2. 4096³ 性能 ≥ cuBLAS 的 85%（RTX 2060 fp16 约 20+ TFLOPS 即达标）
3. 解释 autotune 在 M=1（GEMV 形状）时选了什么、为什么仍打不过专用 GEMV
   （tl.dot 最小 M=16 → 15/16 算力浪费——P5 已讨论过的工业事实）

## 5. 本篇练习

1. epilogue fusion：在 store 前加 bias + gelu（对照 14 章 epilogue 概念，
   体会"在 Triton 里加 epilogue = 加两行"vs CUTLASS 模板的差距）
2. Batched matmul：grid 加一维 batch，stride 多传一组
3. AB 均转置布局时（列优先）性能掉多少？为什么？（访存合并方向）
4. 【进阶】split-K matmul：K 维也切给不同 program + atomic_add 合并——
   M、N 都小而 K 巨大时的方案（和 P4 split-KV 神似，体会"并行度不足就切归约维"）

## 6. 要点回顾

- Triton matmul 三件套：tl.dot(fp32 acc) + grouped ordering + autotune
- 你放弃的：smem 排布/流水细节；保留的：分块与发射顺序策略——14 章选型轴的实感
- decode 形状的 GEMM 是另一个世界（M<16），这是 Triton/tensor core 的共同盲区
