# P4 · Decode 专题：FlashDecoding (split-KV) + 玩具版 Paged KV（1.5 周）

> 前置阅读: 10_flashattention_deep_dive.md §6, 12_kv_cache_management.md, 09_inference_workload.md
> 前置项目: P3（直接改它的 kernel）
> 产出: ①decode attention 三版本（naive / split-KV / paged）②SM 利用率对比报告
>       ③一个 ~200 行的玩具 BlockManager（Python 或 C++）
> 简历叙事: "实现过 PagedAttention 的取数路径和 FlashDecoding 的 split-KV 并行"——
> 这两个点直接对应 vLLM 的核心，远胜于复述论文。

---

## 1. 项目定义

场景切到 decode：`Q=[batch, heads, 1, d]`（单 token query）打一条长 KV
（seq_len 512~16K，fp16, heads=8, d=64）。三个递进实现：

| 版本 | 并行策略 | 解决什么 |
|------|------|------|
| v1 naive decode | grid = batch×heads | 基线；SM 喂不满（30 个 SM vs batch×heads 个 block） |
| v2 split-KV | grid = batch×heads×**num_splits** | KV 维并行 + 二阶段 merge |
| v3 paged | v2 + block table 间接寻址 | 非连续 KV block 的 gather |

## 2. 工程框架

```
src/projects/p4_decode/
├── decode_v1_naive.cu       # 一个 block 串行扫整条 KV（用 P3 的循环体，Br=1）
├── decode_v2_splitkv.cu     # kernel A: 每 split 输出 (O_part, m, l)
│                            # kernel B: merge —— 用 10 章 §1.3 的合并公式
├── decode_v3_paged.cu       # K/V 物理布局 [num_pages, heads, PAGE=16, d] + block_table
├── block_manager.py         # 玩具版分配器: alloc/free/fork(COW)/碎片统计
├── bench.py                 # 扫 (batch, seq_len) 网格, 出热力图
└── verify.py
```

## 3. 分步任务

### Step 1（1 天）v1 + 病情确认

P3 的 flash kernel 把 Q block 退化为 1 行即是 v1。然后用 ncu 抓证据：
`sm__cycles_active` 的 SM 间分布 / `Compute Workload Analysis` ——
batch=1, heads=8 时只有 8 个 block，**30 个 SM 里 22 个在睡觉**。
这一张截图就是 split-KV 的全部动机（10 章 §6.1 的实测版）。

### Step 2（4 天）v2 split-KV——本项目的技术核心

```
kernel A (partial attention):
  grid = (batch*heads, num_splits)
  每个 block 处理 KV[split_start : split_end]，跑 P3 的内循环
  输出到 workspace: O_part[bh, split, d], m[bh, split], l[bh, split]

kernel B (merge)：对每个 (bh) 把 num_splits 份部分结果合并:
  m* = max_s m[s]
  l* = Σ_s l[s]·exp(m[s] - m*)
  O  = Σ_s O_part[s]·l[s]·exp(m[s]-m*) / l*     ← 注意 O_part 是否已归一化，
                                                   和 kernel A 的输出约定要一致!
```
num_splits 的选择策略（启发式）：`min(ceil(seq/256), 足够喂满 SM 的数量)`。
**验收：**
1. 正确（vs v1 输出一致）
2. batch=1, seq=8K 时 v2 比 v1 快 **3× 以上**；
3. 画出"加速比 vs (batch, seq_len)"热力图——**找到 split-KV 开始没收益的边界**
   （batch×heads 已喂满 SM 时 merge 纯属开销），报告里解释这条边界。
   这条边界就是 FlashInfer 里启发式选 num_splits 的原因。

对照阅读：LeetCUDA `kernels/openai-triton/merge-attn-states/` —— 它就是 kernel B
的 Triton 工业版，读懂后可以直接拿来替换你的 merge 验证正确性。

### Step 3（3 天）v3 paged + BlockManager

**kernel 侧**（v2 上改动其实很小——这是面试要点："paged 对 kernel 的侵入度"）：
```cuda
// 连续版:   K_ptr = K + bh*seq*d + pos*d
// paged 版: 
int page_id   = block_table[seq_idx][pos / PAGE_SIZE];   // 多一次查表
half* K_ptr   = K_pool + ((page_id*heads + h)*PAGE_SIZE + pos%PAGE_SIZE)*d;
```
**管理侧** `block_manager.py`（纯 Python 即可，逻辑才是重点）：
- `allocate(seq_id, n_tokens)` / `append_token(seq_id)` / `free(seq_id)`
- `fork(parent, child)`：共享全部 page + 引用计数；写时 `copy_on_write(page)`
- `stats()`：输出物理页利用率、内部碎片
写一个模拟脚本：随机到达/结束的 128 个序列，对比"连续预分配 max_len=4K"
vs paged 的显存占用曲线（matplotlib 一张图，vLLM 论文 Figure 的复刻）。

**验收：**
1. v3 与 v2 输出一致；测出 paged 的性能税（预期 <10%，PAGE=16；
   把 PAGE 调成 1 再测——TokenAttention 的代价立刻可见，12 章 §4 实证）
2. 碎片对比图：paged 利用率 >90% vs 连续分配 <40%
3. fork+COW 在"并行采样 4 路"场景下显存只多 ~1/4 而非 4×

### Step 4（1 天）报告 + 面试题自测

报告必答三问（全是高频面试题）：
- 为什么 decode 必须换并行维？证据是哪个 ncu 指标？
- num_splits 怎么选？什么时候 split 反而变慢？
- paged 之后 kernel 改了哪几行？block_size 的 trade-off 实测数据？

## 4. 关键能力

1. **并行策略与硬件占用的匹配**：会用"block 数 vs SM 数"做第一性分析
2. **两阶段 reduce 的数值正确性**：online softmax 合并公式在分布式场景的应用
3. **间接寻址 kernel 改造**：gather 访存的代价评估（查表+非连续的 L2 行为）
4. **存储管理的系统思维**：分配器/引用计数/COW，算子岗少有的系统加分项

## 5. 常见坑

- kernel A 输出"已归一化"还是"未归一化"的 O_part 必须想清楚，merge 公式相应变化
  （两种都对，混着用必错——本项目最常见 bug）
- workspace 大小 = batch×heads×num_splits×(d+2)×4B，记得按最大配置预分配
- split 边界与 page 边界不对齐时的 off-by-one；用 seq_len=17 这类畸形值测
- 6GB 显存：seq=16K, batch=8 时 KV ≈ 8×2×8×64×16K×2B = 2.1GB，留意余量

## 6. 扩展方向

- GQA decode（KV head 广播），衔接 P3 扩展
- KV cache FP8/INT8 量化版 v4：dequant 融进 kernel A（衔接 P5 与 12 章 §5）
- 把 BlockManager 升级成带 prefix hash 的 APC 玩具版（12 章 §3.3）
- 读 vLLM `csrc/attention/paged_attention_v2.cu`（v2 后缀正是 split-KV！）
  写一篇"我的玩具 vs vLLM 工业版差异"笔记——简历叙事的最后一块拼图

## 7. 参考

- LeetCUDA: `kernels/openai-triton/merge-attn-states/`（merge 的工业实现）
- vLLM: `csrc/attention/paged_attention_v1/v2.cu`、`vllm/core/block_manager.py`
- FlashDecoding 博客 (Tri Dao, pytorch.org)、FlashInfer 论文 §split-kv
- notebook: 10 章 §6、12 章 §2-4、09 章 §1.4
