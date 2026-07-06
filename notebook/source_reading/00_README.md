# 开源项目源码精读 · Source Reading

> 目的：把 ROUTE.md / 00-23 章学到的概念，落到**真实代码的具体文件/函数**上。
> 面试官问"vLLM 的 PagedAttention 怎么实现的"，要求你能答出大概在哪个文件、关键结构是什么，
> 而不是只会背 SOSP'23 论文的图。
>
> 本目录每个项目一个文件夹，记录时间：**2026-06-16**，源码 submodule 克隆并核对路径：**2026-06-17**
> （仓库结构会持续变化，笔记里标注的文件路径请定期用 `git submodule status` 看 pin 的 commit，
> 或 `git -C third_party/<repo> log -1` 核对是否还在）。
>
> **2026-06-17 pin 的 commit**：vllm `3d20275`、sglang `3fb65eb`、flashinfer `2ec0c9b`、
> FlashMLA `9241ae3`、DeepEP `af9a040`、DeepGEMM `88965b0`。本目录所有路径已对照这些 commit 逐一核实存在。

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

读的时候建议开两个窗口：左边这份笔记，右边对应源码按路径跳转。

### 源码已作为 git submodule 克隆到本地 `third_party/`

这些仓库已经像 `third_party/LeetCUDA` 一样，以**浅克隆（`--depth 1`）submodule** 的方式落到本地，
方便用 `grep`/IDE 直接跳转、断点调试，不必每次去 GitHub 翻：

| 文件夹 | 项目 | 本地路径 | 源码主目录 |
|------|------|------|------|
| [01_vllm/](01_vllm/README.md) | vLLM | `third_party/vllm/` | `vllm/v1/` |
| [02_sglang/](02_sglang/README.md) | SGLang | `third_party/sglang/` | `python/sglang/srt/` |
| [03_flashinfer/](03_flashinfer/README.md) | FlashInfer | `third_party/flashinfer/` | `include/flashinfer/` + `flashinfer/` |
| [04_deepseek_infra/](04_deepseek_infra/README.md) | FlashMLA / DeepEP / DeepGEMM | `third_party/{FlashMLA,DeepEP,DeepGEMM}/` | 各自 `csrc/` 或 `deep_gemm/` |

```bash
# 别人首次 clone 本仓库后，拉取所有 submodule（含上述源码）：
git submodule update --init --depth 1

# 只把某个仓库更新到最新（浅克隆，省时间/空间）：
git -C third_party/vllm pull --depth 1 origin main

# 查看当前 pin 的 commit（笔记里标的文件路径以此 commit 为准）：
git submodule status
```

> **为什么用 `--depth 1`**：这些仓库都是数十万行级别、git 历史巨大，但我们只读当前代码、不看历史，
> 浅克隆能把 vLLM/SGLang 从 GB 级压到 ~100MB。`.gitmodules` 里已对它们标了 `shallow = true`。
> 各项目 README 顶部记录了**克隆时 pin 的 commit hash**；仓库结构会持续漂移，
> 笔记里的文件路径若对不上，先 `git submodule status` 看本地 commit，再 `git -C third_party/<repo> pull --depth 1` 更新。
>
> 在线阅读仍可用 `https://github.com/<org>/<repo>/blob/main/<path>`，与本地路径一一对应。

### 用 understand-anything 给 submodule 生成代码知识图谱

已安装 Claude Code 插件 [`understand-anything`](https://github.com/Lum1104/Understand-Anything)（marketplace `Lum1104/Understand-Anything`），
其 `/understand` 技能能对一个目录跑 7 阶段流水线，产出 `.understand-anything/knowledge-graph.json`（节点=文件/类/函数，
边=imports/contains/inherits，外加架构分层 + 导览），再用 `/understand-dashboard` 可视化浏览。

```bash
# 对某个子目录生成中文知识图谱（在 Claude Code 里执行）：
/understand third_party/vllm/vllm/v1/core --language zh
# 看图：
/understand-dashboard third_party/vllm/vllm/v1/core
```

已跑过的产物快照存在 [`understand_graphs/`](understand_graphs/) 下（如 [`vllm_v1_core/`](understand_graphs/vllm_v1_core/README.md)，2026-06-17）。

> **⚠️ 关键陷阱：分析大仓库的"子目录"时，`projectRoot` 必须是"包根"，否则内部依赖边全部丢失。**
> 这些项目里的文件用的是**绝对包导入**（`from vllm.v1.core.kv_cache_utils import ...`，而非相对 `from .kv_cache_utils`）。
> 若直接 `/understand third_party/vllm/vllm/v1/core`，导入解析器会把包路径当作相对该子目录解析 → 找不到文件 → `imports` 边解析出 **0 条**。
> 解决：把分析根设到**能让 `vllm.x.y` 解析成真实文件**的那一层（即 `third_party/vllm/`），或接受 importMap 退化为按文件数分批
> （后续 LLM 读源码阶段仍能补回部分关系，但结构边会缺）。给 SGLang（`from sglang.srt...`）、FlashInfer 等同理。
>
> 另外：`.understand-anything/` 会写进对应 **submodule 的工作区**（未跟踪文件，不影响主仓库提交）；
> 不想污染 submodule 可把它加进 `third_party/<repo>/.git/info/exclude`，或只保留 `understand_graphs/` 下的快照副本。
