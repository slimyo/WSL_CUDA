# 数值格式全家桶：FP32 / BF16 / FP16 / FP8 / INT8 / FP4

> 对象: CUDA / LLM 推理入门
> 前置: 01_gpu_hardware_architecture.md, 06_roofline_and_flops.md
> 目标: 面试能说清每种格式的 bits/exponent/mantissa 分配、range、precision、用途、硬件绑定关系、量化粒度
> 参考 LeetCUDA: `hgemm/`, `flash-attn/`, `sgemm/`, `layer-norm/`

---

## 1. 浮点格式解剖

### 1.1 核心概念：sign / exponent / mantissa

```
FP32 (IEEE 754 single-precision 32-bit floating-point):
       S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM  (1|8|23)
BF16:  S|EEEEEEEE|MMMMMMM                  (1|8|7)
FP16:  S|EEEEE|MMMMMMMMMM                   (1|5|10)
TF32:  S|EEEEEEEE|MMMMMMM... (truncated FP32 mantissa for TC internal)
FP8 E4M3: S|EEEE|MMM                         (1|4|3)
FP8 E5M2: S|EEEEE|MM                         (1|5|2)
FP4 E2M1: S|EE|M                             (1|2|1)
```

### 1.1.1 IEEE 754 特殊值与 Subnormal 数

```
IEEE 754 标准定义了以下分类。不只是 sign/exp/mantissa，exp 全 0 / 全 1 编码特殊值：

  exp=全 1, mant=0          -> Infinity（正负无穷，溢出/除零保护）
  exp=全 1, mant!=0          -> NaN (Not a Number，非法运算结果)
  exp=全 0, mant=0          -> 0（有符号零，+0 和 -0 在比较时相等）
  exp=全 0, mant!=0          -> subnormal（非规格化数，或称 denormal）
```

**Subnormal 数的意义：**

```
当数值渐近 underflow（下溢）时，正常浮点会从 min normal 直接跳 0，产生"精度悬崖"。
Subnormal 通过把 mantissa 前导位从 1 变成 0，让指数固定在 2^(1-bias)，
用 mantissa 位携带"慢慢消失"的精度，实现 gradual underflow（渐进下溢）。

FP32:  min normal ~1.2e-38，subnormal 最小到 ~1.4e-45（23-bit mantissa 全用上）
FP16:  min normal ~6.1e-5， subnormal 最小到 ~6.0e-8
BF16:  min normal ~1.2e-38，subnormal 最小到 ~9.2e-41
```

**GPU 上的 subnormal 处理（面试加分）：**

```
NVIDIA GPU 默认 flush denormals to zero (FTZ / DAZ, Flush-To-Zero / Denormals-Are-Zero)：
  - 输入 subnormal -> 直接当 0 处理
  - 输出 subnormal -> flush 到 0
原因：硬件处理 subnormal 需要 microcode assist（微码辅助），延迟代价极大（几十 cycle）。

这对训练有影响吗？
  - 通常没有。大部分训练值在 normal 范围内。
  - 但某些场景（如非常大的 batch norm、极小的学习率）可能产生 subnormal 梯度，
    flush 后会丢失信号。此时需要显式开启 --ftz=false（代价：性能下降）。
```

**Inf / NaN 传播链（调试 clue）：**

```
一个 Op 产生 NaN -> 所有后续 Op 都输出 NaN -> loss = NaN
常见根源：
  1. 梯度爆炸 -> overflow -> Inf -> 再做 Inf/Inf = NaN
  2. log(0) 或 sqrt(negative) -> NaN
  3. FP16 训练无 loss scaling -> underflow -> 0 -> 除以 0 -> NaN

NV Nsight Systems / Compute 中搜 'nan' 或 'inf' 能定位第一个出问题的 kernel。
也可用 __isnanf() / __isnan() 手动插 check kernel。
```

### 1.1.2 格式精度核心指标

```
理解精度的几个关键指标：

  Epsilon（机器精度）     = 2^(1-mantissa_bits)    - 两相邻 normal 数的最小间距
  Min normal / max normal = 区分可表示值的动态范围
  Unit in Last Place (ULP) = 在当前指数下，mantissa 最低位代表的数值
    相同浮点数值在不同格式下的 ULP 不同：
    FP16 值 1.0 的 ULP = 2^(-10) ~ 0.00098
    BF16 值 1.0 的 ULP = 2^(-7)  ~ 0.0078
    FP16 精度约为 BF16 的 8 倍（在 1.0 附近）
```

> 注意：Epsilon 是 1.0 附近的精度；估值接近 0 时精度更高（因为指数更负），
> 估值很大时精度更差（指数正，ULP 更大）。"相对精度保持稳定"是浮点的优点。

**关键 trade-off：exp 位多 -> 动态范围大；mantissa 位多 -> 精度高。**

### 1.2 格式对比

| 格式 | Total | Exp | Mantissa | Max Normal | Min Normal | Epsilon | 典型用途 |
|------|:---:|:---:|:---:|------|------|------|------|
| FP32 | 32 | 8 | 23 | ~3.4e38 | ~1.2e-38 | 1.19e-7 | 训练基准 / master weights |
| TF32 | 19 | 8 | 10 | ~3.4e38 | ~1.2e-38 | ~1/1024 | A100 Tensor Core 内部 |
| BF16 | 16 | 8 | 7 | ~3.4e38 | ~1.2e-38 | ~1/128 | **训练友好（范围同 FP32）** |
| FP16 (IEEE 754 half-precision 16-bit floating-point) | 16 | 5 | 10 | 65504 | ~6.1e-5 | ~1/1024 | 推理/部分训练（需 loss scaling） |
| FP8 E4M3 | 8 | 4 | 3 | 448 | ~1.6e-2 (2^-6) | 1/8 | **推理权重和激活** |
| FP8 E5M2 | 8 | 5 | 2 | 57344 | ~6.1e-5 (2^-14) | 1/4 | 训练梯度（需更大 range） |
| INT8 (8-bit Integer, 有符号 8 位整型) | 8 | -- | -- | 127 | -- | 1 | 量化推理权重 |
| INT4 (4-bit Integer, 有符号 4 位整型) | 4 | -- | -- | 7 | -- | 1 | 极低比特量化 (GPTQ/AWQ) |
| NVFP4 (NVIDIA FP4, E2M1) | 4 | 2 | 1 | 6 | 0.5 | 0.5 | Blackwell W4A4 推理 |

> **TF32 (Tensor Float 32)** 是 NVIDIA Ampere 架构引入的 Tensor Core 内部格式，
> 它不是开发者显式存储的格式，而是硬件在 GPU register 中读 FP32，在 Tensor Core 内部截断 mantissa 到 10 bit
> 做乘法计算，accumulate 到 FP32。对开发者完全透明——存的是 FP32 数据，但计算以 TF32 精度的 2x 吞吐运行。

### 1.3 深度对比：BF16 vs FP16（面试高频）

```
BF16 的 8-bit exponent（同 FP32）-> range 同 FP32 -> 训练不会 overflow
FP16 的 5-bit exponent -> range 仅 65504（最大可表示值）-> 训练容易 overflow（需 loss scaling）

但 FP16 的 mantissa 有 10 bit，BF16 仅 7 bit -> FP16 精度高于 BF16
精确对比：FP16 在值 1.0 附近的相对精度约 0.1%（ULP ~ 0.00098），
          BF16 在值 1.0 附近的相对精度约 0.8%（ULP ~ 0.0078）。
          FP16 精度约为 BF16 精度的 8 倍（在 1.0 附近）。

那什么时候 BF16 精度不够？
  - 大矩阵乘法累加：BF16 的 mantissa 只有 7 bit，但积累到 FP32 仍可接受；
    但在需要链式累积的场景（如 attention softmax 中 exp(-) 对极值的区分），
    BF16 的 7-bit mantissa 可能导致 exp 输入难以区分->softmax 输出接近 one-hot->精度损失。
  - 解决方式：accumulation（累积）始终用 FP32（Tensor Core 支持 fp32 accumulate，
    即 A=BxC 以 FP16 做乘法，结果以 FP32 累加），
    cuBLAS / CUTLASS 默认就这么做的。这也是为什么 flash attention 的 online softmax
    statistic m/l 始终用 FP32 维护的原因之一。

结论：训练首选 BF16（不需要 loss scaling、不怕 overflow）。
推理两者都常见：用 BF16 训练的模型通常直接用 BF16 推理（避免转换引入误差）；
FP16 的优势是逐元素精度更高、老硬件（V100/T4 无 BF16）也支持。
```

```
细节（面试加分）：

E4M3 (4-bit exponent, 3-bit mantissa) FP8 为了多挤出一点动态范围，不遵循标准 IEEE 754——
  它没有 Infinity（无穷大），且只保留一个 NaN 编码（S.1111.111），所以 max 是 448 而非 240。
  解释：若保留 Inf/NaN 编码，exp=1111 用作 Inf/NaN -> 最大 normal 由 exp=1110 给出。
  NVIDIA 的 E4M3 把 exp=1111 也用作 normal 数：
    max = 1.875 x 2^(15-7) = 1.875 x 256 = 448。
  E5M2 则是标准 IEEE 风格（有 Inf/NaN）。
  这也是为什么 E4M3 用于"值域可控"的权重/激活，E5M2 用于可能溢出的梯度。
```

```
Loss scaling（损失缩放）是什么？
  FP16 训练时，梯度值在深层网络中通常远小于 1（越靠近输入层越小）。
  若梯度 < 6.1e-5（FP16 min normal），直接 underflow（下溢）变 0 -> 该参数永远不再更新。
  Loss scaling = 把 loss 乘一个大常数（如 128.0）-> 梯度放大 -> 带入 FP16 可表示范围。
  backward 后在权重更新前 / scale 恢复。
  BF16 不需要 loss scaling -> 8-bit exp 够用。
```

```
扩展（面试加分）：Loss scaling 的 scale 怎么选？

  静态 loss scaling（固定缩放）：
    固定 scale（如 128.0、1024.0），简单但可能不够或过大。
    过大 scale -> 激活 overflow -> 梯度 Inf -> 训练崩溃。

  动态 loss scaling（AMP, Automatic Mixed Precision 自动混合精度 默认策略）：
    - 每 N 步检查梯度是否有 Inf/NaN
    - 若连续 N 步无 Inf/NaN -> scale x 2（逐步增大，试探上限）
    - 若发现 Inf/NaN -> 丢弃该步、scale / 2（快速恢复）
    典型配置：N=2000，scale 在 [1, 2^24] 之间调整。
    这也是 NVIDIA AMP 默认采用的策略（PyTorch GradScaler 实现）。

  AMP 全流程（训练侧）：
    fwd:  weights/activations 自动选 FP16/BF16（通过 autocast 自动类型转换）
          部分 ops 强制 FP32（softmax、layer-norm、reduce，这些对精度敏感）
    loss: 乘 scale（GradScaler 自动管理 scale 值）
    bwd:  梯度存储在 FP16/BF16（被 scale 放大后不会 underflow）
    grad: 检查 Inf/NaN -> 缩 scale 或跳步
    optim: 梯度 / scale -> 更新 master weights（FP32 副本）

  Master weights（主权重）是什么？
    即使 fwd/bwd 计算用 FP16，optimizer 阶段权重更新在 FP32 精度做。
    即内存里存两份：FP16 用于 forward/backward 计算；FP32 master copy 用于 optimizer update。
    Adam 的 m/v state（一阶/二阶动量）只用 FP32 维护（因为平方累积精度要求高）。
    原因是：FP16 的 ULP 在 1.0 附近是 0.001——如果学习率 1e-5，权重更新量可能远小于 1 ULP，
    round 后更新量为 0 -> 参数 stalling（停滞）。
    这个技巧叫 weight stalling 预防，是混合精度训练的核心工程细节。
```

```
延伸思考：如果模型用 BF16 训练，loss scaling 无意义（因为 range 已够大），
但这不代表 BF16 不需要 weight averaging 或 optimizer state 的 FP32 副本。
Adam 的 m/v state（动量/方差估计）用 FP32 维护（因为平方累积精度要求高），
weight 更新时也需要 FP32 master copy 来吸收小梯度增量。
```

---

## 2. 推理场景的格式选择

### 2.1 W8A8 (Weights 8-bit, Activations 8-bit, SmoothQuant)

```
W8A8：权重和激活都量化到 INT8（有符号 8 位整型）：
  权重: INT8 per-channel（每输出 channel 一个 scale/缩放因子）
  激活: INT8 per-tensor（整个 tensor 一个 scale，经 SmoothQuant 平滑 outlier）

优点：INT8 Tensor Core 加速（A100: 624 TOPS, 比 FP16 的 312 TFLOPS 高 2x）
缺点：INT8 range 窄（127），outlier（异常值）量化误差大
适用：prefill heavy / compute-bound 场景
```

**SmoothQuant 的关键 insight（核心洞察）：**

```
权重分布 easy（均匀，近似高斯），激活分布 hard（有 outlier 特征维度，某些 channel 的值远超其他）。
但激活的 outlier 总是出现在固定的几个 channel 上（不是随机的，由模型学到的模式决定）。
SmoothQuant 思路：把激活的难度"迁移"到权重：
  Y = (X . diag(s)) . (diag(s)^{-1} . W)

  对每个 channel i：
    X'[:,i] = X[:,i] / s[i]    - 激活除以 s[i] -> outlier 被压平到正常范围
    W'[i,:] = W[i,:] x s[i]    - 权重乘 s[i] -> weight 分担了部分难度

  s[i] 通过一个 smoothing factor (alpha) 调节迁移比例：
    s[i] = max(|X[:,i]|)^alpha / max(|W[i,:]|)^(1-alpha)
    alpha 通常取 0.5，控制难度负荷在 weight/activation 之间的分配。alpha 越大 -> 激活分到的难度越少。

迁移后：激活 INT8 量化几乎无额外损失，权重 INT8 因分布均匀也可接受。
```

### 2.2 W4A16 (Weights 4-bit, Activations 16-bit, GPTQ / AWQ)

```
权重 INT4（4-bit Integer），激活 FP16：
  权重: INT4 per-group（按组共享 scale，通常 group_size=128，即每 128 个元素共用一个 scale 和 zero point）
  激活: FP16

优点：权重访存砍 4x，decode memory-bound（带宽瓶颈）直接受益
缺点：激活仍 FP16，不会坍缩精度
适用：decode heavy / memory-bound 场景（最主流）

为什么 decode 阶段 W4A16 如此有效？从 roofline 模型分析：
  1. Decode 1 token = 1 step x batch_size 次 linear 运算
  2. 每个 linear 的计算量: MxKxN (M=1 或 small batch, K=hidden_dim, N=output_dim)
  3. 计算密度 (arithmetic intensity) = FLOPs / 访存量 = M*K*N / (M*K + K*N)
     当 M=1 时，这个值很小 -> memory-bound（带宽瓶颈）
  4. 权重从 FP16 (2 bytes/元素) -> INT4 (0.5 bytes/元素) -> 访存量降至原来的 1/4
  5. vLLM / SGLang 等框架 decode 阶段 >95% 时间花在 weight 访存 -> 4x 加速
```

### 2.2.1 GPTQ vs AWQ 核心区别

```
GPTQ（Generative Pre-Trained Transformer Quantization，基于 OPTQ 算法的后训练量化）：
  - 基于最优脑量化（Optimal Brain Quantizer）的贪心思想
  - 逐个 column 量化，用 Hessian 矩阵（二阶梯度信息）校准误差，并更新未量化列做补偿
  - 适合 batch 离线量化
  - 对 calibration dataset（校准数据集）较为敏感

AWQ（Activation-aware Weight Quantization，激活感知的权重量化）：
  - 观察激活值分布：权重中对应激活 outlier 的 channel 更重要
  - 对这些 salience channel（重要通道）的权重不做 INT4 -> 保留 FP16（仅 1% 左右）
  - 其余 99% 量化到 INT4
  - 因保留 FP16 channel，实际访存略高于纯 W4A16
  - 精度优于 GPTQ（尤其在低 bit 量化场景）

两者差异的核心哲学：
  GPTQ -> 让量化误差在整个矩阵间迁移抵消（"误差补偿"思路）
  AWQ -> 识别重要 channel 跳过量化，避免误差直接发生（"保护重要部分"思路）
  实践中两者都是 W4A16 方案的主流选择。
```

### 2.3 FP8 (H100+)

**硬件背景：**

```
FP8 Tensor Core 首次出现在 H100（Hopper 架构, 2023）：
  - 每个 SM（Streaming Multiprocessor）支持两种 FP8 input 组合（E4M3 + E4M3 或 E4M3 + E5M2）
  - FP8 Tensor Core 吞吐 ~ FP16 的 2x
  - H100 SXM: FP8 ~ 1979 TFLOPS, FP16 ~ 989 TFLOPS
  - B200 (Blackwell): FP8 ~ 4500 TFLOPS, FP4 ~ 9000 TFLOPS
```

**FP8 为何比 INT8 更适合激活量化：**

```
INT8 quantization（均匀量化）：
  - 最大 127，最小 -128
  - 量化步长固定: (max_range - min_range) / 256
  - 对 outlier（异常值）敏感：一个 200 的 outlier 把步长拉到 400/256 ~ 1.56
  - 小值区间（如 -5~5）只有约 6 个量化格点 -> 严重失真

FP8 E4M3:
  - 最大 448，最小 -448
  - 浮点分布：接近 0 时步长极小（2^-6 x 2^-3 = 2^-9 ~ 0.002，跟 FP16 的 ULP 相当）
  - 远离 0 时步长自然变大（指数变大，mantissa 步长等比缩放）
  - 量化误差自动匹配数值的统计分布（自适应的"好"性质）

结论（面试标准回答）：
  INT8 量化误差均匀分布在整个 range 上，
  FP8 的量化误差集中在靠近 0 的小值区间——而激活值大多集中在 0 附近。
  所以同样 8 bit，FP8 E4M3 的激活量化精度天然优于 INT8。
  这也是为什么 SmoothQuant 需要复杂校准来帮助 INT8，而 FP8 基本可以直接 fallback 的原因。
```

```
两种 FP8 格式：
  E4M3 (4-bit exponent, 3-bit mantissa)：值域 [-448, +448] -> 权重和激活的典型范围都在这个区间内
  E5M2 (5-bit exponent, 2-bit mantissa)：值域 [-57344, +57344] -> 训练梯度的范围可能更大，需要 5-bit exp

优势：
  - 动态范围比 INT8 大（448 vs 127）-> 校准简单，不需要 SmoothQuant 级别的复杂校准
  - 硬件原生支持 matmul（矩阵乘法），H100 FP8 Tensor Core 约 1979 TFLOPS（稠密）
  - FP8 KV cache（键值缓存）量化到 FP8 -> 同等显存下可支持更大的 batch 或更长的上下文
```

### 2.4 NVFP4 / MXFP4 / MX 格式家族 (Blackwell 2025-2026)

```
NVFP4 (NVIDIA FP4, E2M1, 2-bit exponent, 1-bit mantissa)：
  每个元素: 1 sign + 2 exp + 1 mantissa = 4 bits
  可表示值集: +/-{0, 0.5, 1, 1.5, 2, 3, 4, 6}
  一级 scaling: block-wise E8M0（每 block 32 元素共享一个 8-bit scale, E8M0 是 8-bit exponent-only 格式）
  二级 scaling: per-tensor FP32（全局 scale，可选，用于补偿全 tensor 的范围偏移）

  E8M0 scale 作用：把 E2M1 的 +2 exponent 范围放大到 FP8 水平的 256x。
  E8M0 是一个 8-bit exponent-only 格式（1 sign + 7 exp, no mantissa），
  表示 2^(e) 形式的 scale，值范围大约 2^(-127) ~ 2^(127)。
  等价于每个 block 的 FP4 数值乘以该 block scale。

MXFP4 (Microscaling FP4, OCP - Open Compute Project Microscaling Formats 标准, 2024)：
  定义在 OCP MX 规范中，与 NVFP4 核心差异：
  - block_size 也为 32（与 NVFP4 一致）
  - 共享 scale 格式统一为 E8M0（非 NVIDIA 私有编码）
  - 无二级 per-tensor scaling
  - 更强调跨硬件互操作性（AMD / Intel / NVIDIA 均可支持）

OCP MX 全家族：
  MXFP8:   element=E4M3/E5M2,  block_size=32, share E8M0 scale
  MXFP6:   element=1 sign + 2 exp + 3 mant (E2M3), block_size=32
  MXFP4:   element=E2M1,  block_size=32
  MXINT8:  block-wise INT8 with shared E8M0 scale

Block scaling 的关键优势：
  传统的 per-tensor/per-channel scaling 假设整个 range 上值分布是"一致的"，
  但激活和权重的 outlier 常常集中在局部区域（特定 token 位置、特定 channel）。
  Block scaling 让每个 block 的约 40 个元素有自己的独立 scale，outlier 影响被限制在 block 内部。

目标：W4A4（权重和激活都 4-bit）-> 吃满 Blackwell Tensor Core 吞吐（约 9000 TFLOPS）
坑：block_size=32 的共享 scale 虽缓解了 outlier，但 block 内部元素差异仍可能很大。
     若 block 内存在一个 outlier，其所在 block 的 scale 被 outlier 拉大，
     其余 31 个元素的有效精度被牺牲。
     这也是激活量化到 W4A4 比权重量化更困难的核心原因。
```

**FP4 可表示值的物理意义（面试中能实际画出来）：**

```
NVFP4 E2M1 的完整值集（1 sign + 2 exp + 1 mant）：

  exp=00 (subnormal/denorm):   0,  0.5
  exp=01:                      1,  1.5
  exp=10:                      2,  3
  exp=11（通常 Inf/NaN）:       4,  6

注意 E2M1 在 exp=11 时没有 Inf 或 NaN 概念，而是扩展了 max normal 值到 6。
这与 E4M3 的设计一致——为了挤出 max 值，牺牲 Inf 编码。
这种设计在推理中完全可以接受（推理没有除零或 NaN 传播路径）。
```

---

## 2.5 格式与 Tensor Core 的绑定关系（面试必知）


每种格式并非在所有 GPU 上都能运行，具体取决于 Tensor Core 的代际支持：

| 格式 | 支持的最小架构 | Tensor Core 代数 | 算力举例（TFLOPS/TOPS） |
|------|:----------:|:---:|------|
| FP16 | Volta V100  | 1st Gen (WMMA, Warp Matrix Multiply and Accumulate) | V100: 125 (TC) |
| TF32 | Ampere A100 | 3rd Gen (WGMMA, Warp Group MMA) | A100: 312       |
| BF16 | Ampere A100 | 3rd Gen          | A100: 312       |
| INT8 | Turing T4   | 2nd Gen (DP4A->TC)| A100: 624       |
| FP8  | Hopper H100 | 4th Gen          | H100: 1979      |
| FP4  | Blackwell B200 | 5th Gen (tcgen05) | B200: 9000    |

```
工程含义（面试标准回答）：
  - 如果部署集群以 A100 为主：推理主力 FP16/BF16 + INT8 W8A8（SmoothQuant）
  - 如果以 H100 为主：FP8 是性价比最优方案（吞吐 2x FP16，KV cache 量化为 FP8 节省显存）
  - Blackwell B200：FP4 W4A4 把 memory-bound decode 的 Wall-clock time 砍到极致（权重 4x 压缩）
```

### 格式演进路线（硬件决定软件）

```
NVIDIA GPU 数值格式支持的时间线——每个新架构都引入新格式，推动 AI 训练/推理效率上台阶：

  Volta (V100, 2017):    FP16 Tensor Core -> 混合精度训练走向主流
  Turing (T4, 2018):     INT8 + INT4 DP4A (Dot Product of 4 Elements and Accumulate) -> 推理量化发端
  Ampere (A100, 2020):   TF32（无需改代码的 2x 加速）+ BF16（大模型训练标配）-> LLM 训练标准
  Hopper (H100, 2023):   FP8 -> 8-bit 推理/训练的甜蜜点，FP8 Transformer Engine 标准
  Blackwell (B200, 2025): FP4 + MX (Microscaling) 格式 -> 4-bit 大模型推理

每个阶段的新格式都向下兼容之前的格式（老代码在新硬件上依然可运行，只是没有吞吐提升）。
```

---

## 3. 面试高频：格式选择推理

**Q: 为什么 W8A8（SmoothQuant）适合 prefill，W4A16 适合 decode？**

```
Prefill:
  - compute-bound（计算瓶颈）-> 算力是限制因素
  - INT8 Tensor Core 约 624 TOPS (A100) vs FP16 Tensor Core 约 312 TFLOPS -> 2x 吞吐
  - 激活量化到 INT8 后，利用 INT8 TC 的 2x 吞吐直接加速
  - SmoothQuant 在校准后精度损失很小（per-channel weight + per-tensor activation）

Decode:
  - memory-bound（带宽瓶颈）-> 访存是限制因素
  - 以 7B 模型为例，权重占约 12.9 GB (FP16) -> 占 decode 大部分访存
  - W4A16 把权重压缩到约 3.2 GB -> 访存减 3-4x
  - 激活只有一个 token 的量（约 KB 级别），量化收益可以忽略，所以保持 FP16
```

**Q: FP8 vs INT8 谁更好？**

```
FP8 优势：
  - 动态范围 ~448 vs INT8 ~127 -> outlier（异常值）处理更好
  - 校准简单（不需要 SmoothQuant 的复杂平滑迁移过程，直接 PTQ）
  - 硬件原生 matmul（矩阵乘法）+ 无额外 dequant（反量化）步骤
  - KV cache 也能量化到 FP8（同等显存可提升 batch size 或 context length）

FP8 vs INT8 的访存开销对比并不是 1:1——因为 INT8 需要先 dequant（反量化）到 FP16
才能做 accumulation（累积）。FP8 Tensor Core 直接吃 FP8 input + FP32 accumulate，
少一个 dequant 步骤，延迟更低、流水线更简洁。

INT8 优势：
  - 更成熟（工具链/库支持完善，社区生态好）
  - A100 就支持 INT8 Tensor Core，FP8 需要 H100+
  - per-channel scaling 更灵活（INT8 的量化/反量化计算简单，易于定制）
```

---

## 3.1 Stochastic Rounding（低精度训练的"秘密武器"）

```
标准 rounding（取整）：IEEE 754 round-to-nearest-even（向最近偶数舍入），确定性强，
但在低精度训练中会引入系统性偏差（systematic bias）。

举例：如果一列梯度值的分数部分全部略低于 quantized grid（量化网格）的中间值，
      标准 rounding 全部向下舍入 -> 长期会产生方向一致的统计偏差。

Stochastic rounding（随机舍入）：
  round(x) = floor(x)  with probability = 1 - (x - floor(x)) / bin_size
              floor(x) + bin_size  with probability = (x - floor(x)) / bin_size

  期望值是 E[round(x)] = x，即无偏估计（unbiased estimator）。

对训练的影响（面试加分）：
  - 标准 rounding 在 FP16 精度训练中引入的噪声 ~ 1 ULP
  - 多次 step 累积后可能形成一致的方向性偏差（不是零均值噪声）
  - Stochastic rounding 把这些噪声变成零均值随机噪声 -> 等效于有更高的有效精度
  - FP8 训练中 stochastic rounding 几乎是标配（NVIDIA Transformer Engine 内置实现）
  - 代价：随机数生成增加少量延迟，但影响远小于精度损失带来的收益
```

---

## 3.2 量化粒度精讲（面试必问）

```
量化时 scale/zero point（零值点）的粒度决定了精度与开销的权衡（粒度越细越准，但 scale 存储开销越大）：

  Per-tensor（单一层一个 scale）：
    - 最粗，开销最小（仅存 1 个 scale + 1 个 zero point）
    - 精度最差：outlier channel 把整个 tensor 的量化范围撑大
    - 例如某一 channel 的范围是 [-1, 200]，其他 channel 是 [-5, 8]，
      用 [-1, 200] 做整体 scale mapping -> 小值 channel 精度大幅损失

  Per-channel（每个 output channel 一个 scale）：
    - 开销适中（m 个 output channel 存 m 个 scale）
    - 精度显著提升：每个 channel 独立定 scale，outlier 不影响其他 channel
    - 权重量化常用（W4A16、W8A16 都用 per-channel）
    - 激活做 per-channel 的 compute 更复杂（dequant 路径需 channel 维度的 broadcast），常用 per-tensor

  Per-group（每 G 个元素一个 scale，如 group_size=128）：
    - GPTQ/AWQ 等权重量化方法的标准粒度
    - group_size 越小 -> 精度越高 -> scale 存储开销越大
    - 典型 trade-off：group_size=128 时 scale 占总存储 ~6.25%（以 W4A16 计）
    - group_size=32 可进一步提升精度，但 scale 开销上升到 25%

  Per-block（每 block 个元素共享一个 scale + exponent 偏移）：
    - MX (Microscaling) 系列的标准粒度
    - block_size=32，内含一个共享 E8M0 scale + 各元素的低精度数据
    - 比 per-group 多一个共享 exponent -> block 内部的数值可分布在不同的指数级上

总结（面试标准回答）：
  粒度越细 <-> 精度越高 <-> scale 存储开销越大
  权重: group=128 是精度/存储最优平衡点（GPTQ/AWQ 实践证明）
  激活: per-tensor 最常用（经 SmoothQuant 等方法校准后够用）
  Block 量化: MX 格式的核心，也是 2025-2026 的趋势
```

---

## 3.3 生产环境格式选型指南（面试终局问题）

```
选型不能被一张表决定，需要根据三个约束综合权衡：

维度 1：计算边界（Roofline 模型决定瓶颈）
  Prefill（compute-bound）-> 激活量化的收益高 -> FP8 / INT8 W8A8
  Decode（memory-bound）-> 权重量化的收益高 -> W4A16 / W4A4

维度 2：部署硬件
  A100 无 FP8 -> 只能 FP16/BF16 + INT8
  H100 有 FP8 -> FP8 W8A8 是 prefill 最佳选择
  B200 有 FP4 -> FP4 在 decode 上的加速比最极端

维度 3：精度要求
  大模型（70B+）量化更敏感 -> 倾向更细粒度（per-group / block scaling）
  小模型（7B-13B）量化鲁棒性更好 -> 可接受 W4A16 甚至 W4A4
  FP8 同 bit 精度优于 INT8 -> outlier 多的模型（如 Llama 系列）首选 FP8

常见生产方案矩阵（面试能画出来）：
                        Prefill              Decode
  A100 集群:        INT8 W8A8 (SmoothQuant)   W4A16 (GPTQ/AWQ)
  H100 集群:        FP8 E4M3 W8A8            W4A16 或 FP8 W8A8
  B200 (Blackwell): FP8 W8A8                 NVFP4 W4A4

注意：以上方案可以混合——同一模型的不同阶段用不同量化配置。
但调度器（scheduler）需要感知量化组件的切换开销（如 dequant kernel 的 load/unload 时间）。
```

---

## 4. LeetCUDA 格式相关源码

| LeetCUDA 目录 | 文件 | 格式覆盖 |
|------|------|------|
| `kernels/hgemm/` | `mma/`, `wmma/`, `wgmma/` | FP16/BF16/TF32 Tensor Core |
| `kernels/flash-attn/` | `flash_attn_mma.py` | FP16 flash attention |
| `kernels/gelu/` | `gelu.cu` | FP32/FP16 (f32/f16/f16x2/f16x8) |
| `kernels/sigmoid/` | `sigmoid.cu` | FP32/FP16 elementwise |
| `kernels/swish/` | `swish.cu` | FP32/FP16 elementwise |
| `kernels/elu/` | `elu.cu` | FP32/FP16 elementwise |
| `kernels/relu/` | `relu.cu` | FP32/FP16 elementwise |
| `kernels/layer-norm/` | `layer_norm.cu` | FP32/FP16 with FP32 accum |
| `kernels/rms-norm/` | `rms_norm.cu` | FP32/FP16 |
| `kernels/sgemm/` | `sgemm_wmma_tf32_stage.cu` | TF32 Tensor Core |
| `kernels/elementwise/` | `elementwise.cu` | FP32/FP16 broadcast |

> **LeetCUDA 格式演进：**
> ```
> f32 -> f32x4 (vectorized) -> f16 -> f16x2 (half2) -> f16x8 (128-bit)
> 每步: 更小的数据类型 -> 更宽的 load/store -> 更高的带宽利用率
> ```

---

## 5. 学习检查清单

- [ ] 能说出 BF16 vs FP16 的 exponent/mantissa 差异和为什么 BF16 训练更稳定
- [ ] 能解释 loss scaling 是什么、为什么 BF16 不需要
- [ ] 能说清 W8A8 vs W4A16 分别打哪个阶段、为什么
- [ ] 能对比 FP8 vs INT8 的优劣、适用场景
- [ ] 能解释 FP4 的两级 scaling 为什么需要、解决什么动态范围问题
- [ ] 能理解 FP16->FP32 累积精度的必要性（见 reduce/layernorm 源码）
- [ ] 能回答 subnormal 是什么、GPU 默认怎么处理（FTZ/DAZ）、对训练的影响
- [ ] 能解释 AMP 中 loss scaling + master weights 的全流程（从 fwd 到 optim 每一步）
- [ ] 能说明 per-tensor / per-channel / per-group / per-block 量化的差异和 trade-off
- [ ] 能说出每种格式对应哪个 GPU 架构的 Tensor Core 代际
- [ ] 能讲清 W4A16 为什么 decode 阶段收益最大（用 roofline 计算密度推导）
- [ ] 能理解 stochastic rounding 的数学动机（无偏估计）和适用场景
- [ ] 能给出一个"生产环境下量化方案选型"的完整理由（从计算边界、硬件、精度三个维度）

---

## 6. 自测 / 面试题

1. BF16 为什么可以取代 FP16 做训练？精度差了哪里？
2. FP8 的 E4M3 和 E5M2 分别用在什么场景？
3. W4A16 vs W8A8 在不同的推理阶段（prefill vs decode）分别怎么选？
4. SmoothQuant 解决了 INT8 量化的什么问题？
5. NVFP4 为什么要 per-block + per-tensor 两级 scale？E8M0 在其中起什么作用？
6. 如果你在 FP8 设备上跑推理，KV cache 量化到 FP8 的收益是什么？
7. AMP 动态 loss scaling 的 scale 增大/缩小策略是什么？为什么需要 master weights？
8. Per-group 量化中 group_size 不同（32 vs 128 vs 256）对精度和存储的影响？
9. Stochastic rounding 为什么能减少低精度训练的统计偏差？它的数学期望是什么？
10. 一台 B200 上跑 70B 模型推理，你会选哪个量化方案？为什么？
    提示：考虑 prefill 阶段（FP8 TC ~ 9000 TFLOPS）和 decode 阶段（FP4 TC ~ 9000 TFLOPS，
    权重从 140 GB（FP16）-> 35 GB（W4A4），显存是否可以容纳等实际约束。

---

## 7. 推荐阅读

| 资料 | 来源 | 笔记推荐度 |
|------|------|:------:|
| IEEE 754-2019 浮点标准 | IEEE | *** |
| Mixed Precision Training (arXiv:1710.03740) | Micikevicius et al. (2017) | ***** |
| FP8: A Mixed-Precision NN Architecture for HPC | Nvidia / arXiv | **** |
| SmoothQuant: Accurate and Efficient PTQ for LLMs | Xiao et al. (ICML'23) | ***** |
| AWQ: Activation-aware Weight Quantization for LLMs | Lin et al. (MLSys'24) | **** |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Frantar et al. (ICLR'23) | **** |
| OCP Microscaling Formats (MXFP4/MXFP8/MXFP6) Specification | OCP (Open Compute Project, 2024) | ***** |
| NVFP4 / Blackwell 白皮书 | NVIDIA (2025) | ***** |
| LeetCUDA hgemm / gelu / layernorm 源码 | `/third_party/LeetCUDA/kernels/` | **** |
| NVIDIA AMP / Transformer Engine 官方文档 | NVIDIA Docs (docs.nvidia.com) | *** |
