# 优化笔记：KV Cache 内存管理深度指南
> **对象**: LLM 推理工程师 / CUDA 算子岗
> **核心目标**: 手画 PagedAttention 内存映射图；讲清 block_size 对碎片与 Kernel 性能的 Trade-off；对比 vLLM 与 SGLang 的缓存策略。
> **参考**: LeetCUDA `flash-attn/` (理解 TMBA (Tile-Matrix Multiply Accumulate) 是理解 PagedAttention 的基石)
---
## 1. 痛点：为什么 KV Cache 是显存杀手？
**核心公式（显存占用）**：
$$ M_{kv} = 2 \times L \times N_{kv} \times d_{head} \times \text{seq\_len} \times \text{batch} \times \text{dtype\_size} $$
**案例计算（7B GQA vs MHA）**：
*   **参数**: L=32, Batch=64, Seq=4096
*   **GQA (8 KV Heads)**: $32 \times 2 \times 8 \times 128 \times 4096 \times 64 \times 2B \approx \mathbf{34 \text{ GB}}$
    *   (模型权重 FP16 ~13GB，**KV 比权重还大**)
*   **MHA (32 KV Heads)**: $\approx 137 \text{ GB}$ (单卡炸裂，这也是 GQA/MLA 存在的意义)
### 1.1 连续分配的“三座大山”
传统方案（如 HF Transformers）为每个序列预分配 `max_seq_len` 的连续内存块：
| 碎片类型 | 成因 | 后果 |
| :--- | :--- | :--- |
| **内部碎片** | 预留 4K，实际只用 100 tokens | 显存预留但未使用，生成提前结束时浪费严重 |
| **外部碎片** | 显存频繁分配/释放，空洞化 | 剩余总量够，但因不连续无法分配给新请求 |
| **管理开销** | 每个序列一个巨型指针 | 难以实现高效的显存复用 |
---
## 2. PagedAttention (vLLM, SOSP'23) —— 核心架构
**核心思想**：将 OS 虚拟内存分页机制引入 GPU 显存管理。**逻辑空间**与**物理空间**解耦。
### 2.1 架构图解（面试必画）
面试时建议画出以下两部分：
```text
      [逻辑视角 - 用户看到的]              [物理视角 - GPU HBM 实际存储]
      
      Sequence A (生成中)                 Block Table (映射表)
      ┌──────────────┐                    ┌─────┬─────┬─────┐
      │ Block 0 (满) │ ──────映射────────→│  7  │  3  │ 15  │ ...
      │ Block 1 (满) │ ──────映射────────→└─────┴─────┴─────┴───┘
      │ Block 2 (写) │ ──────映射────────→      ↓
      └──────────────┘                    物理显存
                                          ┌───┐ ┌───┐ ┌───┐ ┌───┐
                                          │Blk│ │Blk│ │Blk│ │Blk│ ...
                                          │ 7 │ │ 3 │ │15 │ │ 2 │
                                          └───┘ └───┘ └───┘ └───┘
                                            ↑
      Sequence B (刚启动)                   │
      ┌──────────────┐                     │
      │ Block 0 (满) │ ──────映射──────────┘ (共享 Block 7)
      └──────────────┘
```
**讲解口诀**：
1.  **逻辑连续，物理离散**：Token 0-15 可能在 Page 7，Token 16-31 在 Page 3。
2.  **按需分配**：生成 100 个 token 只占 7 个 block (若 block_size=16)，不再预留 max_seq_len。
3.  **Copy-on-Write (CoW)**：Beam Search 或多轮对话时，多个序列可指向同一个物理 Page，直到需要修改才申请新 Page 复制。
### 2.2 Kernel 视角的改动（算子岗重点）
**面试题：PagedAttention Kernel 与标准 Attention 有何不同？**
标准 Attention（如 FlashAttention）假设 KV 内存是连续的：
```cuda
// 标准 FlashAttention: 指针直接偏移
float* k_ptr = k_base + seq_idx * stride_seq;
```
PagedAttention 必须处理 **非连续访存**：
```cuda
// PagedAttention: 需要二级寻址
// 1. 查 Block Table 获取物理 Block ID
int physical_block_id = block_table[block_idx];
// 2. 计算实际物理地址
float* k_block_ptr = k_cache_base + physical_block_id * BLOCK_SIZE * HEAD_DIM;
// 3. Gather 到 SRAM/Registers 进行计算
```
**代价**：
*   **寻址开销**：增加了全局内存读取次数（查表）。
*   **访存效率**：物理上不连续，破坏了 Spatial Locality，难以像 FlashAttention 那样完美合并内存访问。
*   **优化点**：vLLM 将 Block Table 加载到 SRAM/Registers，减少查表延迟；并利用 `ld.global.cg` (Cached Global Load) 缓解非连续访问惩罚。
---
## 3. Block Size 的 Trade-off (高频考点)
Block Size 是碎片率与计算效率的博弈。
| Block Size | 内部碎片率 | Block Table 大小 | Kernel 开销 | 适用场景 |
| :---: | :---: | :---: | :---: | :--- |
| **16** (vLLM 默认) | ~3-4% (平均浪费 8 tokens) | 中等 (seq_len/16) | 中等 | **通用推荐**，平衡点 |
| **1** (TokenAttention) | **0%** | **极大** (seq_len) | **极高** (查表开销等同计算) | 极致显存敏感，计算换显存 |
| **64 / 128** | <1% | 极小 | 低 (更像 FlashAttn) | 超长上下文、大 Batch |
**面试回答模板**：
> "Block Size 是一个超参数。太小（如 1）虽然消除了内部碎片，但导致 Block Table 巨大且 Kernel 频繁查表，计算效率极低；太大则退化为连续分配，产生大量内部碎片。vLLM 默认 16 是基于实测的平衡点，将碎片率控制在 4% 以下，同时保持合理的 Kernel 性能。"
---
## 4. 进阶：RadixAttention vs Automatic Prefix Caching (APC)
**场景**：多轮对话、Few-shot Prompt 存在大量重复前缀。
### 4.1 机制对比
| 特性 | **RadixAttention (SGLang)** | **APC (vLLM)** |
| :--- | :--- | :--- |
| **数据结构** | **基数树** | **哈希表 + LRU** |
| **复用粒度** | 路径复用 (自动识别公共前缀) | 块级 Hash 匹配 |
| **调度策略** | **Cache-Aware Scheduling**<br>(主动调度相似前缀请求到同一 Worker) | 被动命中<br>(请求来时查表) |
| **驱逐策略** | 引用计数 + 叶子节点优先驱逐 | 全局 LRU 驱逐 |
| **优势** | 复用率高，适合 Chat/Few-shot | 实现简单，通用性强 |
### 4.2 核心差异图解
```text
[SGLang Radix Tree]
Root -> "The capital of" -> " France is Paris"
                      └-> " Germany is Berlin"
      (显式维护前缀树，自动共享中间节点)
[vLLM APC]
Hash("The capital of") -> Block 7 (Hit!)
Hash("The capital of France") -> Block 8 (New)
      (基于 Block 内容哈希，被动匹配)
```
---
## 5. KV Cache 量化
**目的**：进一步降低显存带宽压力，支持更长上下文或更大 Batch。
*   **INT8**: 简单的对称量化，需处理 Outlier。
*   **FP8 (E4M3)**: H100+ 原生支持，计算与存储双赢。
*   **INT4**: 极致压缩，需反量化后计算，通常伴随精度损失。
**Kernel 实现**：
*   **On-the-fly Dequant**: 在加载 KV 到 SMEM/Registers 时反量化为 FP16/BF16，再进入 Tensor Core 计算。
*   **Fused Kernel**: 量化和 Attention 算子融合，避免中间结果写回显存。
---
## 6. 面试自测清单
1.  **画图题**：能画出 PagedAttention 的逻辑 Block 到物理 Page 的映射关系，并标出 Block Table 的位置。
2.  **计算题**：给定 7B 模型参数，估算 KV Cache 占比；计算 Block Size 从 16 改为 64 对显存碎片的影响。
3.  **原理题**：为什么 PagedAttention 会影响 GPU Kernel 性能？（答案：非连续访存破坏 Coalescing，增加寄存器压力存 Block Table）。
4.  **对比题**：SGLang 的 Radix Tree 相比 vLLM 的 Hash Table，在什么场景下优势明显？（答案：多轮对话、System Prompt 共享，Tree 可以自然地共享中间节点，Hash 需要完整匹配或者人工切片）。
5.  **场景题**：如果显存极度紧张，但 Batch 要求不高，Block Size 该调大还是调小？（答案：调小，牺牲计算效率换碎片率）。
---
## 7. 补充：参考资源
*   **Paper**: *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, SOSP'23)
*   **Paper**: *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention)
*   **Code**: `vllm/attention/ops/paged_attn.py` (核心 CUDA Kernel 实现)
*   **Related**: FlashAttention (理解标准 Attention 的 Memory Access Pattern 基准)
