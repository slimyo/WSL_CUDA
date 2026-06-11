# 量化策略：GPTQ / AWQ / SmoothQuant / FP8 / NVFP4

> 对象: LLM 推理 / 算子岗
> 前置: 07_numerical_formats.md, 06_roofline_and_flops.md, 14_kernel_routes.md
> 目标: 面试能讲 GPTQ vs AWQ 差异、W4A16 vs W8A8 选型、NVFP4 两级 scaling
> 参考 LeetCUDA: `sgemm/`, `hgemm/`, `flash-attn/`

---

## 1. 量化基础

### 1.1 W?A? 记法

```
W4A16: 权重 INT4，激活 FP16 → 主流推理
W8A8:  权重 INT8，激活 INT8 → 算力优先
W8A16: 权重 INT8，激活 FP16 → 精度优先
W4A4:  权重 INT4，激活 INT4 → Blackwell FP4 支持
```

### 1.2 量化参数

```
对称 vs 非对称：
  对称: scale = max(|x|) / max_int, zero_point = 0
  非对称: scale = (max - min) / (max_int - min_int), zero_point 非零

Per-tensor vs Per-channel vs Per-group:
  越细粒度 → 精度越高 → 反量化开销越大
  典型：group_size = 128（W4A16 标准）

静态 vs 动态量化：
  静态: scale/zero_point 由校准集预计算 → 推理无额外开销
  动态: 每次推理算 scale → 精度更高但开销大
```

---

## 2. W4A16：GPTQ 和 AWQ

### 2.1 decode 为什么选 W4A16 而不是 W8A8

```
decode (batch=1, 含 KV 2.1GB 访存):
  FP16:  总访存 ≈ 15 GB    → I ≈ 0.87 (见 09 章)
  W8A8:  权重减半 → 6.45+2.1 GB → I ≈ 1.5,  单步 ≈ 4.3ms
  W4A16: 权重砍 4× → 3.2+2.1 GB → I ≈ 2.4,  单步 ≈ 2.7ms (≈2.8× 提速)

W4A16 更直接地减少 memory-bound 的瓶颈
  
但 prefill (compute-bound) 场景：
  W8A8 可以利用 INT8 Tensor Core (624 TOPS vs FP16 312 TFLOPS)
  → prefill 提速 2× 可能
```

### 2.2 GPTQ (Hessian-Based, 2023)

**思想：用二阶信息（Hessian）逐层量化。**

```
目标：逐层最小化 ||WX - W_q X||²（量化后该层输出尽量不变）

做法（OBQ/OBS 思路的工程化）:
  1. 用校准集激活算 Hessian: H = 2XX^T（衡量每个权重方向对输出误差的敏感度）
  2. 逐列量化权重
  3. 每量化一列，把产生的误差按 H^{-1} 比例摊给"还没量化"的列去补偿

伪代码:
  for col in range(N):
    q[col] = round(w[col] / scale)             # 量化当前列
    err = (w[col] - deq(q[col])) / H_inv[col, col]
    w[col+1:] -= err * H_inv[col, col+1:]      # 用逆 Hessian 补偿剩余列

优点：精度高（量化误差不是各自为政，而是全层协同补偿）
缺点：要跑校准集 forward 并维护逆 Hessian（分块+Cholesky 技巧使其可行），
     量化一个大模型需要数小时 GPU 时间
```

### 2.3 AWQ (Activation-Aware, 2023)

**思想：看激活值的分布而非权重，保护大激活值对应的权重通道。**

```
观察：激活的某些通道值特别大（outlier）
→ 这些通道的权重如果被量化，误差会被放大（因为激活值大）

解法：
  1. 跑校准集得到激活分布
  2. 找出"显著权重通道"（激活幅度大的输入通道）
  3. per-channel scaling: 把大通道的权重 = 权重 × s（保护）
    反过来输入 activation = x / s（补偿）
  4. s 通过搜索得到（不固定，看哪个 s 使损失最小）

优点：
  - 实现简单（不需要 Hessian）
  - 低 bit 下常优于 GPTQ
  - 不需要反向传播（只跑 forward）
```

**GPTQ vs AWQ 对比：**

| | GPTQ | AWQ |
|---|---|---|
| 核心 | 二阶 Hessian、误差补偿 | 激活幅度感知、per-channel scaling |
| 校准 | 需算 H = 2XX^T | 只需 forward 激活分布 |
| 精度 | 极好 | 低 bit 下略好于 GPTQ |
| 速度 | 慢（大模型算 H 占资源） | 快 |
| 社区工具 | AutoGPTQ | AutoAWQ |

---

## 3. W8A8：SmoothQuant

### 3.1 问题

```
INT8 量化激活更难（因为激活的 outlier 范围更大）

激活 INT8 量化的问题：
  - 某些通道的激活值大到 100+
  - INT8 max = 127 → 其他通道的精度被严重压缩
  - 不能像权重那样 per-channel 量化（激活是 batch 敏感的）
```

### 3.2 SmoothQuant 解法

**把激活的难量化"平滑"迁移到权重上。**

```
数学:
  对激活 X 和权重 W:
    Y = X @ W   (无量化)

  SmoothQuant 引入 per-channel 迁移因子 s:
    Y = (X × diag(s)^{-1}) @ (diag(s) × W)
      = X_smooth  @ W_absorbed
    
    s_j = max(|X_j|)^α / max(|W_j|)^{1-α}

  取 α=0.5: s = sqrt(max(X) / max(W))

效果：
  - X_smooth 的各个通道值域更均匀 → INT8 量化容易
  - W_absorbed 的某些通道值被放大 → 可以在权重量化中容忍
  - 数学上严格等价（不分精度损失）
```

### 3.3 SmoothQuant 的正确评价

```
面试澄清：SmoothQuant 不是"精度损失大的量化"

W8A8 的精度损失来自：
  1. SmoothQuant 迁移过程本身是数学上精确的（没有精度损失）
  2. 损失来自后续的 INT8 量化（权重和激活各 127 级）
  3. 但由于 smooth 后激活更均匀 → INT8 量化精度高

对比 FP8 W8A8：FP8 因为动态范围大，不需要 SmoothQuant
→ 但不是因为 SmoothQuant"精度损失大"，而是 FP8 动态范围更友好
```

---

## 4. FP8 量化 (H100+)

### 4.1 FP8 的优势

```
- 动态范围~448 (E4M3) vs INT8~127 → outlier 处理更好
- 校准简单（不需要 SmoothQuant 的复杂平滑）
- 硬件原生 matmul (H100 FP8 TC: ~4000 TFLOPS)
- 延迟 scaling（per-tensor scaling 延迟计算，减少量化噪声）
```

### 4.2 FP8 KV Cache

```
KV cache 占 decode 显存大头
FP8 KV cache → 每元素 1 字节 → KV cache 减半

挑战：
  - KV cache 的值域分布：softmax 后的 V 值范围可控 → FP8 E4M3 友好
  - dequant 需要融合进 attention kernel
```

---

## 5. NVFP4 / MXFP4 (Blackwell, 2025-2026)

### 5.1 结构

```
NVFP4 元素格式 (E2M1):
  S|EE|M  (1 sign + 2 exp + 1 mantissa = 4 bits)
  可表示的值: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

两级 scaling:
  第一级 (per-block): 每 block=16 元素共享一个 FP8 E4M3 scale
                      （E4M3 是"连续"浮点 scale，比 MXFP4 的 2 的幂 scale 精细）
  第二级 (per-tensor): 全局 FP32 scale（把 block scale 们拉回合适的动态范围）

为什么要两级：E4M3 scale 自己的动态范围有限（max 448），
单级时遇到整 tensor 的极端 outlier 会让所有 block scale 挤到一头；
先用 per-tensor FP32 归一，再让 per-block E4M3 精细分配，两头都顾住。
```

### 5.2 和 MXFP4 的区别

```
MXFP4 (OCP Microscaling Format):
  - Block = 32 元素
  - 每个 block 一个 2 的幂 scale
  - 无二级 scaling

NVFP4 (NVIDIA Blackwell):
  - Block = 16 元素
  - 第一级 scale: per-block FP8 E4M3
  - 第二级 scale: per-tensor FP32
  - 更细的 block + 连续浮点 scale → 更高的精度
```

### 5.3 挑战

```
FP4 的落地挑战：
  1. 小 group (16) 的 scale 开销高（占存储比例大）
  2. 传统 outlier 缓解方法（如 SmoothQuant 的 channel scaling）
    被 group 粒度限制
  3. 需要专门校准算法（ModelOpt / MR-GPTQ）
  4. 仅 Blackwell 可跑（B200/B300/RTX 5090/PRO 6000）
```

---

## 6. Dequant-GEMM Fusion（跨 M6+M7）

```
不 fusion:
  weight INT4 → dequant 整张到 FP16 → 写 HBM → 读 HBM → GEMM

Dequant-GEMM Fusion:
  SMEM 中 load INT4 block → SMEM 中 dequant → 立即做 matmul
  不写回 HBM，不额外读

Triton 伪代码:
  @triton.jit
  def dequant_gemm_kernel(A_ptr, B_ptr, C_ptr, ...):
      A = tl.load(A_ptr + offsets, mask=mask)    # INT4
      scale = tl.load(scale_ptr + ...)            # scale
      A_fp16 = (A.to(tl.float16) * scale)        # dequant in register
      C = tl.dot(A_fp16, B)                      # matmul directly
```

> **源码映射：LeetCUDA 没有量化 kernel。**工业实现读这些：
> - vLLM `csrc/quantization/`（AWQ/GPTQ/FP8 的 dequant-GEMM CUDA kernel）
> - Marlin（IST Austria DAS lab）：W4A16 GEMM 标杆 kernel，vLLM 的 GPTQ 默认 backend
> - TensorRT-LLM / CUTLASS mixed-input GEMM；llm-compressor（量化工具链）

---

## 7. 学习检查清单

- [ ] 能说清 GPTQ vs AWQ 的机理差异
- [ ] 能论证 W4A16 vs W8A8 分别适用哪个阶段
- [ ] 能解释 SmoothQuant 做了"平滑"转移
- [ ] 能对比 FP8 vs INT8 的取舍
- [ ] 能讲 NVFP4 的两级 scaling 为什么需要
- [ ] 能解释 dequant-GEMM fusion 为什么重要

---

## 8. 自测 / 面试题

1. 为什么 decode 场景普遍 W4A16 而不是 W8A8？反过来呢？
2. SmoothQuant 的"平滑"在数学上做了什么？为什么把难量化迁到权重就 OK？
3. FP8 相比 INT8 最大的好处是什么？
4. NVFP4 为什么要 per-block + per-tensor 两级 scale？
5. AWQ 和 GPTQ 的核心思路差异是什么？哪种低 bit 更好？

---

## 9. 推荐阅读

| 资料 | 来源 |
|------|------|
| GPTQ: Accurate Post-Training Quantization (2023) | Frantar et al. |
| AWQ: Activation-Aware Weight Quantization (2023) | Lin et al. |
| SmoothQuant: Accurate & Efficient PTQ (ICML'23) | Xiao et al. |
| FP8: A Mixed-Precision NN Architecture | Nvidia / arXiv |
| NVFP4 Blackwell 白皮书 | NVIDIA (2025) |
| NVIDIA ModelOpt | GitHub |
| AutoGPTQ / AutoAWQ | GitHub |
| LeetCUDA hgemm / sgemm (融合参考) | `/third_party/LeetCUDA/` |
