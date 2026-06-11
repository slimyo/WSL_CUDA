# P5 · W4A16 Fused Dequant-GEMM（Triton，1-1.5 周）

> 前置阅读: 15_quantization.md, 07_numerical_formats.md, 14_kernel_routes.md §6
> 前置项目: P1（验收）、P3 Step4（Triton 手感）
> 产出: ①INT4 权重打包/解包工具 ②fused dequant-GEMM Triton kernel
>       ③"decode 形状下 W4A16 ≈ 带宽换算力"的实测报告
> 简历叙事: "实现 W4A16 dequant-GEMM，decode 形状(M=1~16)下实测 x.x× 提速，
> 并能解释收益完全来自访存而非算力"——把 M6(kernel) 和 M7(量化) 一次打通。

---

## 1. 项目定义

模拟 LLM decode 的线性层：`Y[M,N] = X[M,K] @ W[K,N]`，
X 是 fp16 激活（M=1/4/16，decode 的真实形状），W 是 **per-group 对称量化的 INT4**
（group_size=128，每 group 一个 fp16 scale）。K=N=4096（一层 proj 的真实大小）。

理论预期（先算后测，09 章方法论）：
```
M=1 时该 GEMM 是 GEMV，memory-bound，时间 ≈ 权重字节 / 336GB/s
fp16 权重: 4096×4096×2B = 33.5MB → ~100μs
int4 权重: 8.4MB + scale 0.25MB → ~26μs   → 理论加速 ≈ 3.9×
你的目标：实测达到理论加速的 80% 以上。
```

## 2. 工程框架

```
src/projects/p5_w4a16/
├── quantize.py          # fp16 权重 → int4 打包 + scales（每 int32 装 8 个 int4）
├── kernels/
│   ├── gemm_fp16.py     # Triton fp16 基线（或直接调 torch.matmul 当基线）
│   ├── gemm_w4a16.py    # ★fused dequant-GEMM
│   └── dequant_naive.py # 反例：先整张 dequant 落显存再 matmul（必须实现！）
├── accuracy.py          # 量化误差评估: 逐层 MSE + cos 相似度
└── bench.py             # M ∈ {1,4,16,64,256,1024} 扫描 → 找 crossover 点
```

## 3. 分步任务

### Step 1（2 天）量化与打包

```python
# per-group 对称量化 (group 沿 K 维)
W_g    = W.reshape(K//G, G, N)                  # G=128
scale  = W_g.abs().amax(dim=1, keepdim=True) / 7   # int4 对称: [-7, 7] (-8 不用，对称)
W_q    = (W_g / scale).round().clamp(-7, 7).to(torch.int8)
# 打包: 8 个 int4 → 1 个 int32（注意符号位处理：先 +8 存无符号，kernel 里再 -8）
```
验收：`dequant(pack(W)) ≈ W`，逐元素误差 ≤ scale/2；
accuracy.py 报告 cos 相似度 >0.99（group=128 时正常应达到）。
**顺手做实验：group_size ∈ {32,128,1024,per-channel} 的误差曲线——
15 章 §1.2 "粒度 vs 精度" 的实测版，报告里一张图。**

### Step 2（4 天）fused kernel

```python
@triton.jit
def w4a16_gemm(X, Wq, Scales, Y, M, N, K,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr, GROUP: tl.constexpr):
    # 每个 program 算 Y 的 [BLOCK_M, BLOCK_N] tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        x   = tl.load(...)                       # [BLOCK_M, BLOCK_K] fp16
        w32 = tl.load(...)                       # [BLOCK_K//8, BLOCK_N] int32
        # ---- 解包: 位运算取出 8 路 int4 ----
        shifts = (tl.arange(0, 8) * 4)[:, None]
        w4  = (w32[:, None, :] >> shifts) & 0xF          # → [BLOCK_K, BLOCK_N]
        s   = tl.load(Scales + (k // GROUP) ...)         # 当前 K 段的 scale
        w   = (w4.to(tl.float16) - 8.0) * s              # dequant 在寄存器里!
        acc += tl.dot(x, w)
    tl.store(Y..., acc.to(tl.float16))
```
核心纪律：**解包+反量化全程发生在寄存器，W 的 fp16 形态从不落 HBM**——
这就是 14 章 §6 fusion 和 15 章 §6 的全部要义，其余都是位运算细节。
验收：与 `X @ dequant(W)` 的 torch 结果误差 <1e-2。

### Step 3（2 天）三方对比 + crossover 分析

bench 三条线：fp16 基线 / dequant_naive（先解包落显存再 matmul）/ fused。
扫 M ∈ {1...1024}，必出的结论（验收就看你能不能复现+解释）：
1. M 小（decode）：fused ≈ 3-4× 于 fp16 —— 纯访存收益
2. dequant_naive 永远最慢 —— 它比 fp16 还多读写一遍权重（反面教材的价值）
3. M 大（prefill 形状）：fused 加速比退化甚至跑输 fp16 ——
   GEMM 变 compute-bound 后，int4 不带来算力收益，解包反而占指令
   → **这就是 15 章 "W4A16 打 decode、W8A8 打 prefill" 的自产证据，
   报告里这张 crossover 图是全项目最值钱的产出。**
4. ncu 验证：fused 版 `dram__bytes_read` ≈ int4 权重字节（命中理论值）

### Step 4（1 天）报告

按 P0 模板写，重点：理论预测 vs 实测的偏差归因（scale 读取、解包指令、
L2 命中），以及"如果有 INT8 TC（本卡就有）怎么做 W8A8"的纸上设计。

## 4. 关键能力

1. **量化数值细节**：对称/非对称、group 粒度、打包格式、零点处理
2. **fusion 的本质判断**：中间结果（fp16 W）是否落 HBM 是唯一标准
3. **用 roofline 预测优化收益**：先算理论加速比再动手，误差 <20%
4. **crossover 思维**：同一优化在不同形状下收益反转——面试答"什么时候不用 X"的万能素材

## 5. 常见坑

- int4 符号处理：+8 偏移存储最省事，忘了减回去结果全错且不易察觉（误差检验救你）
- Triton 维度广播容易写出 shape 对但语义错的解包，先用 K=16,N=8 的小矩阵
  + numpy 模拟逐位验证打包/解包
- `tl.dot` 要求 BLOCK_M ≥ 16：M=1 时 padding 到 16（浪费但能用），
  这正是工业界 GEMV 不走 tensor core 的微观原因（顺手写进报告）
- scale 用 fp16 存但乘法转 fp32 做，避免小 scale 下溢

## 6. 扩展方向

- **AWQ 式保护**：用一个小校准集找激活大的通道，对其 scale 上调——
  对比量化误差改善（15 章 §2.3 的玩具复现，半天工作量，简历再加一句话）
- INT8 W8A8 + SM75 INT8 Tensor Core（`tl.dot` int8 路径）→ 验证 prefill 形状下反超
- 接到 P4：给 decode attention 的 KV 也做 INT8 quant（kernel A 内 dequant）
- 读 Marlin kernel 的 README（W4A16 标杆）：它比你的版本多了什么
  （异步流水、L2 友好的权重重排、warp 级解包）——纸上对比即可

## 7. 参考

- 工业实现: Marlin (github.com/IST-DASLab/marlin)、vLLM `csrc/quantization/awq/`、
  bitsandbytes、AutoAWQ 的 Triton kernel
- Triton 官方 tutorial 03-matrix-multiplication（GEMM 骨架）
- GPTQ/AWQ 论文（只看量化格式部分即可支撑本项目）
- notebook: 15 章 §2/§6、07 章 §2.2、14 章 §6
