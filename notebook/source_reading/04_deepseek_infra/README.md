# DeepSeek 推理三件套：FlashMLA / DeepEP / DeepGEMM

> 三个独立仓库，DeepSeek 在 2025 年 "Open Source Week" 一起放出，工业级、窄而深，
> 分别精确对应 MLA / 大规模专家并行(EP) / FP8 GEMM 三个面试高频点。
> fetch 时间：2026-06-16。
> 本地（均为 submodule 浅克隆，2026-06-17 核对路径全部存在）：
> `third_party/FlashMLA/`（pin `9241ae3`）、`third_party/DeepEP/`（pin `af9a040`）、`third_party/DeepGEMM/`（pin `88965b0`）。

为什么排第四：通用性不如前三个（不是 serving 引擎，是给 serving 引擎/训练用的专精算子库），
但**深度无可替代**——面试问到 MLA、大 EP、FP8 训练推理一致性时，
"我读过 DeepSeek 官方实现"是比"我看过论文"分量重得多的回答。

---

## FlashMLA

仓库：[deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)。MLA decode/prefill 的专用 kernel。

| 概念 | 文件 |
|------|------|
| 按 SM 架构分代实现（[01](../../01_gpu_hardware_architecture.md)） | `csrc/sm90/`（Hopper）、`csrc/sm100/`（Blackwell）、`csrc/smxx/`（通用兜底） |
| decode / prefill 分两条路径（[10](../../10_flashattention_deep_dive.md) §6 FlashDecoding 的 MLA 版） | `csrc/sm90/decode/`、`csrc/sm90/prefill/` |
| CUTLASS 依赖 | `csrc/cutlass`（子模块） |

**精读重点**：对照 [11](../../11_attention_variants.md) §3.3 decoupled RoPE，在 `decode/` 子目录里找
"latent 向量上投影还原" 和 "RoPE 维度单独处理" 这两步是否真的是分开两个 kernel 调用还是融合在一个 kernel 里——
这是判断"工业实现把数学上的两步融合到什么程度"的具体证据。

**面试落点**：FlashMLA 按 SM90/SM100 分别给实现，直接说明 [18](../../18_frontier_2025_2026.md) 提到的
"不对称硬件缩放" 不是空话——同一个数学问题在不同代际硬件上需要完全不同的 kernel 代码。

---

## DeepEP

仓库：[deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)。大规模 MoE 的 **all-to-all 专家通信库**。

| 概念（[16](../../16_distributed.md) [23](../../23_moe_inference.md)） | 文件 |
|------|------|
| EP 通信内核 | `csrc/kernels/`（具体 all-to-all dispatch/combine kernel） |
| 弹性 EP（[18](../../18_frontier_2025_2026.md) "大 EP 成旗舰标配"） | `csrc/elastic/`、`csrc/kernels/elastic/`（专家数/卡数动态变化时的处理） |
| Python API | `csrc/python_api.cpp`（pybind 入口） |

**精读重点**：DeepEP 解决的核心矛盾是 [23](../../23_moe_inference.md) §1-2 提到的 MoE
"token 路由到不同专家所在的卡" 这一步通信量大、延迟敏感；通读 `csrc/kernels/` 下 dispatch
（token 发给对应专家）和 combine（专家算完结果发回）两类 kernel，对照你笔记里 grouped GEMM 之前的
那一步路由通信，理解"EP 的瓶颈通常不在算而在传"。

**面试落点**：能讲清 "为什么 MoE 大模型(如 DeepSeek-V3) 需要专门的通信库而不是直接拿 NCCL all-to-all"——
答案是 token 路由是 **不均衡、动态稀疏** 的通信模式，需要针对性优化（如 NVLink/RDMA 混合、low-latency 模式），
对照 `csrc/kernels/backend/` 看它对不同互联拓扑的适配。

---

## DeepGEMM

仓库：[deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。**FP8 细粒度 scaling 的 GEMM 库**，
DeepSeek-V3 训练用的同一套 FP8 思路在推理侧的延伸。

| 概念（[07](../../07_numerical_formats.md) [15](../../15_quantization.md)） | 文件 |
|------|------|
| 核心 GEMM 实现 | `deep_gemm/include/`（CUDA/CUTLASS 风格头文件） |
| 历史/兼容实现 | `deep_gemm/legacy/` |
| 新一代实现（命名暗示在迭代架构） | `deep_gemm/mega/` |
| 工具/测试 | `deep_gemm/utils/`、`deep_gemm/testing/` |

**精读重点**：DeepGEMM 的卖点是 **细粒度（per-block/per-tile）FP8 scaling**，
不是简单 per-tensor scale——这是 DeepSeek-V3 技术报告里"FP8 训练不掉精度"的关键之一。
对照 [15](../../15_quantization.md) §4 FP8 小节，在 `deep_gemm/include/` 里找 scale 因子是
按多大粒度（per-128×128 block 之类）施加的，这比记住"FP8 训练"这个结论本身有区分度。

**面试落点**："DeepSeek 怎么做到 FP8 训练不崩？" 不要只答 E4M3/E5M2 格式选择，
要能讲到 **DeepGEMM 这种细粒度 scaling 的工程实现** 才是让 FP8 训练实际可用的关键一环，
呼应 [18](../../18_frontier_2025_2026.md) "FP8 训练(DeepSeek-V3) → FP4 推理(NVFP4/MXFP4)" 这条主线。

---

## 三者关系一句话总结

**FlashMLA 省 attention 的 KV 访存，DeepEP 省 MoE 的跨卡通信，DeepGEMM 省 GEMM 的算力/带宽（FP8）**——
三个刀口对准 DeepSeek 架构（MLA + 细粒度 MoE）里三个最贵的环节，凡是面试问"DeepSeek 推理为什么快"，
这三个名字 + 各自解决什么，是比泛泛而谈"工程优化好"更有说服力的回答。

---

## 本地核对补充（2026-06-17）

逐一核对三件套的本地目录，上文路径全部存在，并补充几条对照时有用的细节：

- **FlashMLA**：`csrc/` 下确实是 `sm90/`、`sm100/`、`smxx/`（兜底）三套按架构分代的实现 + `cutlass/`（子模块）；
  而且 `csrc/sm90/` 内部**真的分成了 `decode/` 和 `prefill/` 两个子目录**（外加 `helpers.h`），印证上文"decode/prefill 走两条独立 kernel 路径"。
  另有 `csrc/api/`（对外接口）、`csrc/kerutils/`、`params.h`/`defines.h`/`utils.h`。读 MLA 时从 `csrc/sm90/decode/` 入手最贴近 decode 主场景。
- **DeepEP**：`csrc/kernels/` 下分 `backend/`（不同互联拓扑：NVLink/RDMA 适配，对应面试落点）、`elastic/`（弹性 EP 的 dispatch/combine）、`legacy/`；
  仓库还有顶层 `csrc/jit/`（运行时编译通信 kernel）、`csrc/indexing/`、`csrc/elastic/`、`csrc/python_api.cpp`（pybind 入口）。
  与 vLLM `distributed/elastic_ep/`、SGLang `srt/eplb/` 三处对照，能讲清"弹性大 EP"是这一代 MoE 推理的共同主题。
- **DeepGEMM**：`deep_gemm/` 下的 `include/`、`legacy/`、`mega/`（新一代）、`utils/`、`testing/` 都在；
  注意仓库**新增了顶层 `csrc/` 目录**，里面有 `jit_kernels/`、`jit/`、`apis/`、`indexing/`——说明 DeepGEMM 也走了**JIT 生成 kernel**的路线
  （和 FlashInfer 的 `jit/` 同思路，见 [03_flashinfer](../03_flashinfer/README.md)）。找"细粒度 FP8 scaling"的证据时，从 `deep_gemm/include/` 里搜 scale/tile 粒度相关定义。
- **交叉印证**：vLLM 的 `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py`（见 [01_vllm](../01_vllm/README.md) 补充）说明 **DeepGEMM 已被上游 serving 引擎直接集成**，
  不是孤立 demo——这正是"窄而深的专精算子库 → 被通用框架吸收"的真实路径，面试时是有力的细节。
