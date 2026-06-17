# FlashInfer 源码精读

> 仓库：[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)，默认分支 `main`。
> 两层结构：`include/flashinfer/`（C++/CUDA kernel 本体，header-only 风格）+
> `flashinfer/`（Python 封装 + JIT 编译入口）。
> fetch 时间：2026-06-16。

为什么排第三：vLLM 和 SGLang 的 attention/MLA/MoE kernel **后端之一就是 FlashInfer**
（见 [01_vllm](../01_vllm/README.md) 的 `vllm/v1/attention/selector.py`、
[02_sglang](../02_sglang/README.md) 的 quantization 里大量 `*_flashinfer_*` 文件名）。
读完两个 serving 引擎的"调度怎么调用 kernel"之后，FlashInfer 是看"kernel 本身怎么写"的自然下一站，
直接对应 [10](../../10_flashattention_deep_dive.md)/[14](../../14_kernel_routes.md) 从"用 kernel"到"写 kernel"的过渡。

---

## 1. 代码地图：概念 → 目录 → 关键文件

| 概念（notebook 章节） | 目录 | 关键文件 |
|------|------|------|
| FlashAttention 系 kernel（[10](../../10_flashattention_deep_dive.md)） | `include/flashinfer/attention/` | `prefill.cuh`（prefill 路径）、`decode.cuh`（decode 路径，对应 FlashDecoding 的单 query 场景）、`hopper.cuh`/`hopper/`（Hopper warp-specialized 实现，对应 [10](../../10_flashattention_deep_dive.md) §4 v3）、`persistent.cuh`/`persistent_template.cuh`（persistent kernel 模式） |
| MLA 专用 kernel（[11](../../11_attention_variants.md) §3） | `include/flashinfer/attention/` | `mla.cuh`、`mla_hopper.cuh`、`mla_params.cuh`、`cutlass_mla.cuh`（CUTLASS 实现版）、`decode_mla_cute_sm80.cuh`（Ampere 上用 CuTe 写的 MLA decode，**SM80 即可用，对你 RTX 2060 系列以上理解 CuTe 实战很有参考性**） |
| Blackwell 新架构（[18](../../18_frontier_2025_2026.md)） | `include/flashinfer/attention/blackwell/` | tcgen05/Tensor Memory 相关实现 |
| PagedAttention 的内存视图（[12](../../12_kv_cache_management.md)） | `include/flashinfer/` | `page.cuh`（非连续 paged KV 的取数抽象，这是 [12](../../12_kv_cache_management.md) §2 "attention kernel 必须改成从非连续 block gather" 的具体代码） |
| 稀疏 attention（[18](../../18_frontier_2025_2026.md) §2.2 NSA/MoBA） | `include/flashinfer/` | `attention/sparse_mla_sm120/`（说明稀疏 MLA 已经在往新硬件上落地） |
| GEMM / MoE / 量化（[15](../../15_quantization.md) [23](../../23_moe_inference.md)） | `include/flashinfer/gemm/`、`flashinfer/fused_moe/` | grouped GEMM 与 MoE 路由融合实现 |
| FP4/FP8 量化 kernel | `flashinfer/` | `fp4_quantization.py`、`fp8_quantization.py` |
| Python 侧统一入口 | `flashinfer/` | `decode.py`/`prefill.py`（对应上面 C++ 的高层封装）、`mla/`、`page.py`（block table 的 Python 接口）、`jit/`（按 shape/dtype 现场 JIT 编译特化 kernel，这是 FlashInfer 比"静态编译一份 kernel"更激进的工程选择） |
| 投机解码辅助（[22](../../22_sampling_decoding.md)） | `flashinfer/` | `sampling.py`（top-k/top-p 等采样 kernel）、`topk.py` |

---

## 2. 精读路径

1. **先看 Python 入口理解调用形态**：`flashinfer/decode.py` 的某个 `*_decode` 函数签名，
   理解它接收的 paged KV 表示（block table、page indices）是什么形状——
   这能直接回答 [12](../../12_kv_cache_management.md) "paged 之后 kernel 接口要改什么"。
2. **再下沉到 C++ kernel**：`include/flashinfer/page.cuh` 看非连续 block 怎么被 gather 进 SMEM，
   对照你自己写过的（或要写的）`projects/P4_flashdecoding_paged_kv.md` 玩具实现，看真实工业实现复杂度差在哪。
3. **MLA 专用路径**：`include/flashinfer/attention/mla.cuh`，对照 [11](../../11_attention_variants.md) §3.2b
   "MLA 推理为什么要矩阵吸收"，在代码里找 up-projection 矩阵被吸收进权重的证据。
4. **JIT 机制**：`flashinfer/jit/` 看它如何按 head_dim/dtype/causal 等参数现场生成并缓存编译产物——
   这是 [14](../../14_kernel_routes.md) "Triton autotune" 思路在手写 CUDA 库里的另一种实现方式，可以对比着记。

---

## 3. 常见面试追问 → 这份代码怎么答

- **"FlashInfer 和 FlashAttention 官方库（Dao-AILab/flash-attention）什么关系？"**
  → FlashInfer 不是 FA 的替代而是 serving 场景的扩展：专门优化 **decode/paged KV/MLA/批量变长序列**
  这些 serving 引擎实际需要、但 FA 官方库以训练场景为主没有重点覆盖的路径，
  这也是为什么 vLLM/SGLang 把它作为 attention backend 之一而不是唯一选择。
- **"为什么需要 JIT 而不是把所有 shape 组合提前编译好？"** → kernel 模板参数组合
  （head_dim × dtype × causal × MLA/非MLA × 硬件代际）太多，提前全编译二进制体积/编译时间爆炸，
  JIT 按实际遇到的 shape 现场编译+缓存是工程上的折中，这正是 [14](../../14_kernel_routes.md) 讲编译器化路线时
  "可移植性 vs 极致优化"那条轴的一个具体案例。
- **"MLA 的 kernel 为什么不能直接套用标准 MHA flash kernel？"** → 因为 KV 的物理布局是
  压缩后的 latent 向量 + 解耦的 RoPE 维度（[11](../../11_attention_variants.md) §3.3），
  `mla.cuh`/`mla_params.cuh` 里能看到这两部分是分开传参、分开处理的，标准 flash kernel 的输入假设不成立。

---

## 4. 备注

- 这是三个项目里**唯一以 CUDA/C++ 为主**的仓库，适合配合你自己的 P2/P3/P4 项目代码风格对比读。
- 头文件较多用 C++ template 重度泛型，第一遍读不需要逐个模板参数搞懂，先抓控制流和数据布局。
