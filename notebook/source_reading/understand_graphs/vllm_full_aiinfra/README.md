# understand-anything 知识图谱 · vLLM(AI-infra 范围)

> 用 `/understand` 半自动管线对 `third_party/vllm`（pin `3d20275`）的 **6 大 AI-infra 模块** 跑出的代码知识图谱快照。
> 生成时间：2026-06-17。配套 [01_vllm](../../01_vllm/README.md) 精读用。原始产物在 `third_party/vllm/.understand-anything/`。

## 规模

- **411 个 Python 文件**，**1353 节点**（411 file / 731 class / 211 工厂函数）、**2437 边**（contains 942 / inherits 348 / imports 1146 + tested_by 1）。
- 校验：core 的 `validateGraph` + zod schema 均通过，0 问题、0 孤立节点。

## 7 个分层（= 6 大模块，core 拆成调度 + KV）

| 层 | 目录 | 节点数 |
|---|---|---|
| 调度层 | `vllm/v1/core/sched/` | 20 |
| KV 缓存管理 | `vllm/v1/core/`(非 sched) | 53 |
| 注意力后端 | `vllm/v1/attention/` | 255 |
| 投机解码 | `vllm/v1/spec_decode/` | 36 |
| 分布式 / EP / KV 传输 | `vllm/distributed/` | 448 |
| 量化 | `vllm/model_executor/layers/quantization/` | 255 |
| MoE 融合 | `vllm/model_executor/layers/fused_moe/` | 286 |

8 步导览（`.tour`）按面试主线串：调度器主循环 → KV 缓存 → 注意力后端选择 → MLA/混合架构 → 投机解码 → 量化 → MoE 融合+DeepGEMM → 分布式 TP/EP/KV 传输。

## 摘要质量说明（重要）

- **头部关键文件**（笔记点名的 16 个：eagle/flashinfer/fp8/fused_moe/parallel_state…）是**手工中文精修**。
- **其余长尾节点**摘要是**半自动生成**：`[中文角色标签]` + 从源码 **docstring 首句**自动提炼；无 docstring 的回退为 `[标签] 名称`（如 `[调度] Scheduler 类。`）。所以长尾节点的摘要简略、且 docstring 原文多为英文——这是 411 文件不逐个手写的取舍，结构（类/继承/导入）是准确的。

## 怎么看

```bash
/understand-dashboard third_party/vllm        # 指向原始产物目录（不是这个快照目录）
```

> 这个快照目录文件是平铺的、没有 `.understand-anything/` 子层，dashboard 读不了；它是离线留存 + 可直接 `jq .layers/.tour/.nodes` 当文本地图。

## 范围说明

未纳入 `vllm/v1/worker/`(92f) 等执行层；如需扩范围，重跑时把目录加进 scope 文件列表即可（注意仍要用包根 `third_party/vllm` 做 projectRoot 解析导入，见 [00_README](../../00_README.md)）。
