# 实战项目总览：6-8 周从"会写 kernel"到"会优化、会证明"

> 目标：补齐 ROUTE.md 能力分级里的 🔴L3（动手实现），产出 3-5 个可写进简历、
> 能扛住面试追问的项目。每个项目都强制走 **"实现 → profile → 解释 → 优化 → 再 profile"** 闭环。
> 配套理论：每个项目文件头部标注了对应的 notebook 章节，先读再做。

---

## 0. 你的环境与硬约束（所有项目通用，先读！）

```
GPU:    RTX 2060 (Turing, SM75, 6GB VRAM, 30 SMs)
CUDA:   13.2，nsys / ncu 已装在 /usr/local/cuda/bin
系统:   WSL2
```

**SM75 能用什么 / 不能用什么（写代码前背下来）：**

| 能用 ✓ | 不能用 ✗（面试要会讲，但本机写不了） |
|------|------|
| FP16 WMMA/MMA Tensor Core (m16n8k8 / 16×16×16) | BF16 / TF32（SM80+） |
| INT8/INT4 Tensor Core | `cp.async` 异步拷贝（SM80+）→ 用寄存器双缓冲替代 |
| `__shfl_sync` 全家、LDG.128 (float4) | TMA / wgmma / setmaxnreg（SM90） |
| Triton（支持 Turing 的 `tl.dot` fp16） | tcgen05 / TMEM（SM100） |
| CUTLASS 2.x 风格 GEMM + CuTe layout 概念 | CUTLASS 3.x SM90 collective mainloop |

**编译统一用 `-arch=sm_75 -O3 -lineinfo`**（`-lineinfo` 让 ncu 能把指标定位到源码行）。

**理论峰值（roofline 计算用，背下来）：**
```
FP32:           ~6.5 TFLOPS          (1920 core × 2 × ~1.7GHz)
FP16 TC (fp32 累加): ~26 TFLOPS      (GeForce Turing 半速 FP32 累加)
FP16 TC (fp16 累加): ~52 TFLOPS
HBM(GDDR6) 带宽: 336 GB/s
→ ridge point: FP32 ≈ 19 FLOPs/B, FP16TC ≈ 77~155 FLOPs/B
```

**WSL2 下 ncu 的权限坑（第一次必踩）：**
ncu 读 GPU performance counter 需要 Windows 侧授权——
NVIDIA 控制面板 → 桌面 → 开发者设置（或 NVIDIA app → System）→
"允许所有用户访问 GPU 性能计数器"，改完重启。报 `ERR_NVGPUCTRPERM` 就是这个问题。
6GB 显存：所有实验的矩阵/序列规模都给了上限建议，OOM 自己减半。

---

## 1. 项目地图与时间线

```
周 1-2   P1 Profiling 训练馆        ← 先把"测量与归因"练成肌肉记忆
              │ (nsys/ncu 方法论贯穿后面所有项目)
周 2-4   P2 HGEMM: WMMA→优化→CUTLASS   ← Tensor Core + CUTLASS/CuTe
              │
周 4-6   P3 FlashAttention forward     ← 算子岗最高频手写题
              │
周 6-7   P4 FlashDecoding + Paged KV   ← decode 并行策略 + 非连续访存
              │
周 7-8   P5 W4A16 Dequant-GEMM (Triton) ← 量化 × fusion 一次打通

可选加餐（不占 GPU，适合穿插）:
  P6: 极简 continuous batching + chunked prefill 模拟器（纯 Python mock
      延迟模型，跑出 TPOT 分布对比图；见 ROUTE.md 项目清单第 4 条）
```

依赖关系：P1 的方法论是后面所有项目的"验收工具"；P2 的 WMMA tile 直接复用进
P3 的 S=QKᵀ 和 PV；P4 复用 P3 的 online softmax 合并公式；P5 独立。
**时间不够的优先级：P1 > P3 > P2 > P4 > P5**（P3 是面试现场最可能写的）。

---

## 2. 统一工程规范（每个项目照此执行）

```
cpp/
├── src/projects/
│   ├── p1_profiling/        每个项目一个目录
│   ├── p2_hgemm/
│   ├── ...
│   └── common/              共享：计时器、检验、cuBLAS 对照
│       ├── timer.cuh        # cudaEvent 计时 + warmup + 中位数
│       ├── verify.cuh       # 与参考实现比对 (相对误差 < 1e-2 for fp16)
│       └── bench.py         # 跑参数扫描、出 CSV/图
└── notebook/projects/
    └── P?_*.md              每个项目一份【实验报告】（见 §3 模板）
```

**铁律：**
1. 每个 kernel 必须先有 **正确性检验**（对照 cuBLAS/torch/CPU 朴素实现）再谈性能
2. 计时一律 cudaEvent + ≥10 次 warmup + 100 次取中位数（理由见 17 章 §5）
3. 每个优化 step 单独留存（`kernel_v1.cuh, v2, v3...`），report 里逐版本记录
   ncu 关键指标——**面试官要看的是"演进证据链"，不是最终版本**

---

## 3. 实验报告模板（每项目交一份，就是简历素材）

```markdown
# <项目名> 实验报告
## 1. 动机（用 roofline/二分语言，2-3 句）
## 2. 实现版本演进表
| 版本 | 改动 | 耗时(ms) | 带宽/算力利用率 | ncu 关键证据 |
## 3. 瓶颈分析（贴 ncu 截图/数据，说明每次优化"为什么有效/为什么没效"）
## 4. 与上限的距离（终版 vs cuBLAS/理论带宽，差距在哪、还能怎么榨）
## 5. 踩坑记录（面试聊起来最出彩的部分）
```

---

## 4. 简历叙事模板（项目做完往里填）

> "手写 ___ kernel 并用 Nsight Compute 做逐版本归因：从 naive 的 __% 带宽利用率，
> 通过 ___、___、___ 优化到 __%（达 cuBLAS/理论上限的 __%）；
> 能解释每一步的瓶颈迁移（memory→latency→compute）。"

面试官的追问必然是：**"你怎么知道瓶颈在哪？" → 用 P1 的方法论回答；
"为什么这一步有效？" → 用报告里的 ncu 指标回答。** 这就是闭环的意义。

---

## 5. 项目文件索引

| 文件 | 项目 | 核心能力 | 对应章节 |
|------|------|------|------|
| [P1_profiling_gym.md](P1_profiling_gym.md) | Profiling 训练馆 | nsys/ncu/瓶颈归因方法论 | 17, 06, 03 |
| [P2_hgemm_cutlass.md](P2_hgemm_cutlass.md) | HGEMM 三连 | Tensor Core / 流水线 / CUTLASS+CuTe | tensor_cores, hgemm_opt, 14 |
| [P3_flashattention.md](P3_flashattention.md) | FlashAttention fwd | online softmax / tiling / Triton | 10, softmax, flash_attn |
| [P4_flashdecoding_paged_kv.md](P4_flashdecoding_paged_kv.md) | Decode 专题 | split-KV / 非连续 gather | 10§6, 12 |
| [P5_w4a16_dequant_gemm.md](P5_w4a16_dequant_gemm.md) | 量化 GEMM | dequant fusion / 带宽换算力 | 15, 07, 14 |
