# 开源项目源码精读 · Source Reading

> 目的：把 ROUTE.md / 00-23 章学到的概念，落到**真实代码的具体文件/函数**上。
> 面试官问"vLLM 的 PagedAttention 怎么实现的"，要求你能答出大概在哪个文件、关键结构是什么，
> 而不是只会背 SOSP'23 论文的图。
>
> 本目录每个项目一个文件夹，记录时间：**2026-06-16**（仓库结构会持续变化，
> 笔记里标注的文件路径请定期用 `git log -1` 或 GitHub 网页核对是否还在）。

---

## 优先级排序（投入产出比）

| 优先级 | 项目 | 仓库 | 为什么优先 | 对应 notebook 章节 |
|:---:|------|------|------|------|
| 🥇 P0 | **vLLM** | [vllm-project/vllm](https://github.com/vllm-project/vllm) | 行业事实标准、面试被提及概率最高；V1 引擎一个仓库覆盖 调度/PagedKV/量化/分布式/投机解码 几乎全部模块 | [12](../12_kv_cache_management.md) [13](../13_scheduling.md) [15](../15_quantization.md) [16](../16_distributed.md) [22](../22_sampling_decoding.md) [23](../23_moe_inference.md) |
| 🥈 P1 | **SGLang** | [sgl-project/sglang](https://github.com/sgl-project/sglang) | RadixAttention/前缀复用的原产地；调度器与 vLLM 对比是高频追问；P/D 分离生产实现 | [12](../12_kv_cache_management.md) [13](../13_scheduling.md) |
| 🥉 P2 | **FlashInfer** | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | vLLM/SGLang 的 attention/MLA/MoE kernel 底层供应商，是连接"会用框架"和"会写 kernel"的桥 | [10](../10_flashattention_deep_dive.md) [11](../11_attention_variants.md) [14](../14_kernel_routes.md) [23](../23_moe_inference.md) |
| 4 | **DeepSeek 推理三件套**（FlashMLA / DeepEP / DeepGEMM） | [deepseek-ai](https://github.com/deepseek-ai) | MLA/大规模 EP/FP8 GEMM 的工业级参考实现，2024-2025 面试高频提及 | [11](../11_attention_variants.md) §3 [16](../16_distributed.md) [23](../23_moe_inference.md) [18](../18_frontier_2025_2026.md) |

> 排序逻辑：先吃**通用 serving 引擎**（vLLM 覆盖面最广，且是其余项目的对照基准）→
> 再吃**调度/前缀缓存的另一种实现**（SGLang，用来对比出设计取舍）→
> 再下沉到**两者共用的 kernel 层**（FlashInfer）→
> 最后看**专精的工业组件**（DeepSeek 三件套，窄但深，呼应 MLA/MoE 章节）。

---

## 怎么用这套笔记

每个项目文件夹下的 `README.md` 结构固定：

1. **代码地图**：仓库顶层目录 → 笔记里学过的概念 → 关键文件，三栏对照表。
2. **精读路径**：按"先看接口/数据结构，再看调度逻辑，最后看 kernel"的顺序给文件清单。
3. **与 notebook 章节的映射**：本仓库哪块代码对应你已经学过的哪一节，复习时双向跳转。
4. **常见面试追问 + 这份代码怎么答**。
5. **版本说明**：记录 fetch 时间和分支，防止"代码已经改了笔记没改"的错觉。

读的时候建议开两个窗口：左边这份笔记，右边 GitHub 按路径直接跳转
（`https://github.com/<org>/<repo>/blob/main/<path>`），不需要本地 clone 整个仓库——
这些仓库都是数十万行级别，clone 下来也不会逐行看，按需在线读 + 笔记里记路径更高效。
如果要本地调试/打断点，再针对单个模块 `git clone --depth 1 --filter=blob:none --sparse` 拉子目录。

| 文件夹 | 项目 |
|------|------|
| [01_vllm/](01_vllm/README.md) | vLLM |
| [02_sglang/](02_sglang/README.md) | SGLang |
| [03_flashinfer/](03_flashinfer/README.md) | FlashInfer |
| [04_deepseek_infra/](04_deepseek_infra/README.md) | FlashMLA / DeepEP / DeepGEMM |
