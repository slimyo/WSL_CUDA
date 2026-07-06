# understand-anything 知识图谱 · vllm/v1/core

> 用 Claude Code 插件 [`understand-anything`](https://github.com/Lum1104/Understand-Anything) 的 `/understand` 对
> `third_party/vllm/vllm/v1/core`（pin commit `3d20275`）跑出的代码知识图谱产物，**复制**到这里留存。
> 生成时间：2026-06-17。这是给 [01_vllm](../../01_vllm/README.md) 精读时配套用的「可视化代码地图」。

## 文件说明

| 文件 | 作用 |
|------|------|
| `knowledge-graph.json` | 核心产物：65 节点(15 file / 35 class / 15 function) + 84 边(contains 50 / inherits 12 / imports 22) + 4 个架构分层 + 6 步导览。dashboard 渲染它。 |
| `meta.json` | 元数据：分析时的 commit hash、时间、文件数（增量更新时用来判断是否变更）。 |
| `fingerprints.json` | 各源文件的结构指纹基线，供 `/understand` 增量更新时比对哪些文件结构变了。 |
| `config.json` | `{"outputLanguage":"zh"}`，记住输出语言偏好。 |

## 怎么看（dashboard）

dashboard 默认从「被分析目录」里的 `.understand-anything/` 读图。原始产物仍在
`third_party/vllm/vllm/v1/core/.understand-anything/`（**没删**，删了 dashboard 就找不到），直接：

```bash
# 在 Claude Code 里：
/understand-dashboard
# 或对指定目录：/understand-dashboard third_party/vllm/vllm/v1/core
```

这里这份是**离线快照**：仓库结构漂移、或重跑 `/understand` 覆盖了原件后，还能回看 2026-06-17 当时的图。
也可以直接 `jq` 翻 `knowledge-graph.json`（`.layers[]`、`.tour[]`、`.nodes[]`）当文本地图读。

## 图里的四层（对照 [01_vllm](../../01_vllm/README.md) 的 v1/core 部分）

1. **KV 缓存基础 (Foundation)** — `kv_cache_utils.py`(块对象/空闲块 LRU 链表/块哈希/配置算法) + `kv_cache_metrics.py`。
2. **KV 缓存管理层** — `block_pool.py` → `single_type_kv_cache_manager.py`(按注意力类型分裂) → `kv_cache_coordinator.py`(多组协调) → `kv_cache_manager.py`(门面)。
3. **调度层 (Scheduling)** — `sched/scheduler.py`(continuous batching 主循环) + `interface.py`/`output.py`/`request_queue.py`/`async_scheduler.py`/`utils.py`。
4. **多模态编码缓存** — `encoder_cache_manager.py`。

> 6 步导览(`.tour`)就是按「地基 → 块池 → 单类型管理器 → 协调器 → 门面 → 调度器主循环」的依赖顺序排的，
> 跟着走一遍能快速建立 v1/core 的全局心智模型，再回 `01_vllm/README.md` 逐文件深读。
