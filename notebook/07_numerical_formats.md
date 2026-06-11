# 数值格式全家桶：FP32 / BF16 / FP16 / FP8 / INT8 / FP4

> 对象: CUDA / LLM 推理入门
> 前置: 01_gpu_hardware_architecture.md, 06_roofline_and_flops.md
> 目标: 面试能说清每种格式的 bits/exponent/mantissa 分配、range、precision、用途
> 参考 LeetCUDA: `hgemm/`, `flash-attn/`, `sgemm/`

---

## 1. 浮点格式解剖

### 1.1 核心概念：sign / exponent / mantissa

```
FP32:  S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM  (1|8|23)
BF16:  S|EEEEEEEE|MMMMMMM                  (1|8|7)
FP16:  S|EEEEE|MMMMMMMMMM                   (1|5|10)
TF32:  S|EEEEEEEE|MMMMMMM... (truncated FP32 mantissa for TC internal)
FP8 E4M3: S|EEEE|MMM                         (1|4|3)
FP8 E5M2: S|EEEEE|MM                         (1|5|2)
FP4 E2M1: S|EE|M                             (1|2|1)
```

**关键 trade-off：exp 位多 → 动态范围大；mantissa 位多 → 精度高。**

### 1.2 格式对比

| 格式 | Total | Exp | Mantissa | Max Normal | Min Normal | Epsilon | 典型用途 |
|------|:---:|:---:|:---:|------|------|------|------|
| FP32 | 32 | 8 | 23 | ~3.4e38 | ~1.2e-38 | 1.19e-7 | 训练基准 / master weights |
| TF32 | 19 | 8 | 10 | ~3.4e38 | ~1.2e-38 | ~1/1024 | A100 Tensor Core 内部 |
| BF16 | 16 | 8 | 7 | ~3.4e38 | ~1.2e-38 | ~1/128 | **训练友好（范围同 FP32）** |
| FP16 | 16 | 5 | 10 | 65504 | ~6.1e-5 | ~1/1024 | 推理/部分训练（需 loss scaling） |
| FP8 E4M3 | 8 | 4 | 3 | 448 | ~1.6e-2 (2⁻⁶) | 1/8 | **推理权重和激活** |
| FP8 E5M2 | 8 | 5 | 2 | 57344 | ~6.1e-5 (2⁻¹⁴) | 1/4 | 训练梯度（需更大 range） |
| INT8 | 8 | — | — | 127 | — | 1 | 量化推理权重 |
| INT4 | 4 | — | — | 7 | — | 1 | 极低比特量化 (GPTQ/AWQ) |
| NVFP4 (E2M1) | 4 | 2 | 1 | 6 | 0.5 | 0.5 | Blackwell W4A4 推理 |

### 1.3 深度对比：BF16 vs FP16（面试高频）

```
BF16 的 8-bit exponent（同 FP32）→ range 同 FP32 → 训练不会 overflow
FP16 的 5-bit exponent → range 仅 65504 → 训练容易 overflow（需 loss scaling）

但 FP16 的 mantissa 有 10 bit，BF16 仅 7 bit → FP16 精度高于 BF16

结论：训练首选 BF16（不需要 loss scaling、不怕 overflow）。
推理两者都常见：用 BF16 训练的模型通常直接用 BF16 推理（避免转换引入误差）；
FP16 的优势是逐元素精度更高、老硬件（V100/T4 无 BF16）也支持。
```

```
细节（面试加分）：E4M3 为了多挤出一点动态范围，不遵循标准 IEEE 754——
  它没有 Infinity，且只保留一个 NaN 编码（S.1111.111），所以 max 是 448 而非 240。
  E5M2 则是标准 IEEE 风格（有 Inf/NaN）。
  这也是为什么 E4M3 用于"值域可控"的权重/激活，E5M2 用于可能溢出的梯度。
```

```
Loss scaling 是什么？
  FP16 训练时，梯度值通常远小于 1（尤其深层网络）。
  若梯度 < 6.1e-5（FP16 min normal），直接 underflow 变 0。
  Loss scaling = 把 loss 乘一个大常数（如 128.0）→ 梯度放大 → 在 FP16 范围内。
  backward 后在权重更新前 ÷ scale 恢复。
  BF16 不需要 loss scaling → 8-bit exp 够用。
```

---

## 2. 推理场景的格式选择

### 2.1 W8A8 (SmoothQuant)

```
权重和激活都 INT8：
  权重: INT8 per-channel
  激活: INT8 per-tensor （经过 SmoothQuant 平滑 outlier）

优点：INT8 Tensor Core 加速（A100: 624 TOPS）
缺点：INT8 range 窄（127），outlier 量化误差大
适用：prefill heavy / compute-bound 场景
```

### 2.2 W4A16 (GPTQ / AWQ)

```
权重 INT4，激活 FP16：
  权重: INT4 per-group（group_size=128）
  激活: FP16

优点：权重访存砍 4×，decode memory-bound 直接受益
缺点：激活仍 FP16，不会坍缩精度
适用：decode heavy / memory-bound 场景（最主流）
```

### 2.3 FP8 (H100+)

```
两种 FP8 格式：
  E4M3：值域 [−448, +448] → 权重和激活的典型范围够用
  E5M2：值域 [−57344, +57344] → 训练梯度够用

优势：
  - 动态范围比 INT8 大（448 vs 127）→ 校准简单
  - 硬件原生支持 matmul (H100 FP8 TC 约 4000 TFLOPS)
  - FP8 KV cache 可扩 batch/上下文
```

### 2.4 NVFP4 / MXFP4 (Blackwell 2025-2026)

```
NVFP4 (E2M1):
  元素: 1 sign + 2 exp + 1 mantissa = 4 bits
  值: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
  一级 scaling: per-block E4M3（每 block 32 元素，一个 8-bit scale）
  二级 scaling: per-tensor FP32（全局 scale）

MXFP4 (OCP):
  per-block scale（block=32，2 的幂）
  无二级 scaling
  更适合通用硬件

目标：W4A4（权重和激活都 4-bit）→ 吃满 Tensor Core 吞吐
坑：小 block 粒度会抵消传统 outlier 缓解手段
```

---

## 3. 面试高频：格式选择推理

**Q: 为什么 W8A8（SmoothQuant）适合 prefill，W4A16 适合 decode？**

```
Prefill:
  - compute-bound → 算力瓶颈
  - INT8 Tensor Core 约 624 TOPS (A100) vs FP16 约 312 TFLOPS
  - 激活量化到 INT8 能跑更快
  - SmoothQuant 精度损失很小

Decode:
  - memory-bound → 带宽瓶颈
  - 权重占 12.9 GB (FP16) → 占 decode 大部分访存
  - W4A16 把权重砍到 3.2 GB → 访存减 3-4×
  - 激活量化对 memory-bound 场景帮助不大
```

**Q: FP8 vs INT8 谁更好？**

```
FP8 优势：
  - 动态范围 ~448 vs INT8 ~127 → outlier 处理更好
  - 校准简单（不需要 SmoothQuant 的复杂平滑）
  - 硬件原生 matmul + 无额外 dequant
  - KV cache 也能量化到 FP8

INT8 优势：
  - 更成熟（工具链/库支持）
  - A100 就支持 INT8 TC，FP8 需要 H100+
  - per-channel scaling 更灵活
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
> f32 → f32x4 (vectorized) → f16 → f16x2 (half2) → f16x8 (128-bit)
> 每步: 更小的数据类型 → 更宽的 load/store → 更高的带宽利用率
> ```

---

## 5. 学习检查清单

- [ ] 能说出 BF16 vs FP16 的 exponent/mantissa 差异和为什么 BF16 训练更稳定
- [ ] 能解释 loss scaling 是什么、为什么 BF16 不需要
- [ ] 能说清 W8A8 vs W4A16 分别打哪个阶段、为什么
- [ ] 能对比 FP8 vs INT8 的优劣、适用场景
- [ ] 能解释 FP4 的两级 scaling 为什么需要、解决什么动态范围问题
- [ ] 能理解 FP16→FP32 累积精度的必要性（见 reduce/layernorm 源码）

---

## 6. 自测 / 面试题

1. BF16 为什么可以取代 FP16 做训练？精度差了哪里？
2. FP8 的 E4M3 和 E5M2 分别用在什么场景？
3. W4A16 vs W8A8 在不同的推理阶段（prefill vs decode）分别怎么选？
4. SmoothQuant 解决了 INT8 量化的什么问题？
5. NVFP4 为什么要 per-block + per-tensor 两级 scale？
6. 如果你在 FP8 设备上跑推理，KV cache 量化到 FP8 的收益是什么？

---

## 7. 推荐阅读

| 资料 | 来源 |
|------|------|
| FP8: A Mixed-Precision NN Architecture | Nvidia / arXiv |
| SmoothQuant: Accurate and Efficient PTQ | Xiao et al. (ICML'23) |
| NVFP4 / Blackwell 白皮书 | NVIDIA (2025) |
| LeetCUDA hgemm / gelu 源码 | `/third_party/LeetCUDA/kernels/` |
