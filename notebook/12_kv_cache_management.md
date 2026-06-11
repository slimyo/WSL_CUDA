# KV Cache 内存管理：PagedAttention / RadixAttention / APC

> 对象: LLM 推理工程 / 算子岗
> 前置: 11_attention_variants.md, 10_flashattention_deep_dive.md
> 目标: 面试能手画 PagedAttention 架构，讲清碎片管理、prefix caching、block_size 取舍
> 参考 LeetCUDA: `flash-attn/`

---

## 1. KV Cache 是推理的显存瓶颈

```
以一个 7B 级 GQA 模型为例（32 层, 8 KV heads, seq=4096, batch=64）:
  模型权重: ~13 GB (FP16)
  KV cache: 32 × 2 × 8 × 128 × 4096 × 64 × 2B ≈ 34 GB
  → KV cache 比权重还大
（若是 Llama-7B 原版 MHA（32 KV heads），同条件下 KV ≈ 137 GB，单卡放不下——
 这也是 11 章 GQA/MLA 存在的原因）

矛盾：
  显存是有限的 → 需要高效管理 KV cache
  服务高并发 → 需要容纳更多序列的 KV
```

### 1.1 碎片化的来源

```
连续分配（HF transformers 方式）:
  对每个序列按 max_seq_len 预分配连续显存
  ┌─────────────────────────────────────┐
  │ Seq 0: 预分配 4K (用 100 tok)         │ ← 内部碎片
  │ Seq 1: 预分配 4K (用 512 tok)         │ ← 内部碎片
  │ Seq 2: 预分配 4K (用 2K tok)           │ ← 部分碎片
  │ ...                                   │
  │ Seq N: 无法分配 (但总容量还有空余)      │ ← 外部碎片
  └─────────────────────────────────────┘

三种碎片：
  - 内部碎片：预分配但用不上的空间（序列提前结束）
  - 外部碎片：空闲块但不连续，无法分配给新序列
  - 管理开销：每个序列的连续分配管理
```

---

## 2. PagedAttention (vLLM, SOSP'23)

### 2.1 核心思想

**照搬 OS 虚拟内存分页：KV cache 切成固定大小的 block（page），通过 block table 做逻辑→物理映射。**

```
┌──────────────────┐    block table    ┌──────────────────┐
│ 逻辑 KV 序列       │                  │  物理显存 (HBM)   │
│                   │                  │                  │
│  Block 0 (tok 0-15) │ ────────→     │  Page 7 (Data)   │
│  Block 1 (tok 16-31)│ ────────→     │  Page 3 (Data)   │
│  Block 2 (tok 32-47)│ ────────→     │  Page 15 (Data)  │
│  Block 3 (tok 47-63)│ ────────→     │  Page 2 (Data)   │
│  ...               │                  │                  │
└──────────────────┘                  └──────────────────┘
```

### 2.2 优点

```
1. 只分配实际使用的空间
   序列生成了 100 token → 只需 ceil(100/16) = 7 个 page
   不再预留 max_seq_len

2. 只有最后一块有内部碎片：每序列平均浪费 block_size/2 = 8 个 token 的空间
   （浪费率 = 8/seq_len，序列越长越可忽略；vLLM 论文实测显存浪费 < 4%，
    对比连续预分配方案的 60-80% 浪费）

3. Copy-on-Write（写时复制）
   并行采样/beam search 共享前缀 block
   写时才复制（不需要每样本独立分配）

4. 支持内存复用（swap / recompute）
```

### 2.3 block_size 取舍（面试高频）

| block_size | 内部碎片 | block table 大小 | Kernel 寻址开销 | 适用场景 |
|:---:|:---:|:---:|:---:|------|
| 16（vLLM 默认） | ~3% | 中 | 中 | 通用 |
| 64 | ~0.8% | 小 | 低 | 长序列为主 |
| 1（TokenAttention） | ~0% | 极大 | 高 | 极限碎片化 |
| 256 | ~0.2% | 极小 | 很低 | 长序列/大 batch |

**规律：block 小 → 显存省（碎片少）但管理开销大（block table 大、kernel 寻址代价高）；block 大 → 反之。**

### 2.4 PagedAttention Kernel

**核心改动：attention kernel 从连续内存改为非连续 block gather。**

```cuda
// 标准连续 attention:
// K_cache: [num_layers, num_heads, max_seq_len, head_dim]
float *K_cached = K_base + h * max_seq_len * head_dim;
float key = K_cached[pos * head_dim + i];

// Paged attention (非连续):
// K_cache: [num_pages, num_heads, tokens_per_page, head_dim]
// block_table: [num_blocks] → 逻辑到物理映射
int num_blocks = seq_len / BLOCK_SIZE;
for (int b = 0; b < num_blocks; b++) {
    int phys_page = block_table[b];
    float *K_block = K_base + phys_page * BLOCK_SIZE * head_dim;
    // 在这个 block 内做部分 attention
    // 需要处理跨 block 的地址跳跃
}
```

**LeetCUDA 参考：LeetCUDA 没有直接实现 PagedAttention，但 `flash-attn/mma/` 展示了如何在 SMEM 中 tiling attention，这就是 PagedAttention kernel 需要的 tile 能力。vLLM 的 paged attention kernel 在 `vllm/attention/ops/paged_attn.py`。**

---

## 3. RadixAttention (SGLang)

### 3.1 基数树管理

```
比 PagedAttention 更进一步：用基数树（Radix Tree）管理 KV 块。

请求 1: "What is the capital of France?"
请求 2: "What is the capital of Germany?"
请求 3: "What is the capital of France and its population?"

                    root
                     │
              "What is the "
             /            \
    "capital of France?"  "capital of Germany?"
            │
   " and its population?"

共享前缀的 KV block 复用 → 聊天场景效果显著。
```

### 3.2 Cache-Aware Scheduling

```
SGLang 的调度器不只是"来请求就处理"，它考虑 cache 亲和性：
  - 把共享相同前缀的请求路由到同一 worker
  - LRU 驱逐策略：最久未用的 KV block 被优先清理
  - 清理后，后续请求如果共享该前缀需要重新计算 prefill
```

### 3.3 RadixTree vs Hash-Based (vLLM APC)

| 特性 | RadixAttention (SGLang) | APC (vLLM) |
|------|:---:|:---:|
| 复用粒度 | 前缀（path 共享） | block hash 匹配 |
| 存储结构 | Radix Tree | Hash Table |
| 驱逐策略 | LRU（子树级别） | LRU（block 级别） |
| 调度影响 | 可路由请求使亲和 | 被动匹配 |
| 缓存命中 | 高（共享前缀场景） | 中（block 级别） |
| 实现复杂度 | 较高 | 较低 |

**面试回答：**
```
"RadixAttention 用树结构显式追踪前缀，可以做精确的 cache-aware 调度；
 APC 用 block hash 隐式匹配，实现更简单但不可控调度。两者目标相同，
 SGLang 适合 chat/few-shot（高前缀复用），vLLM 适合通用场景。"
```

---

## 4. TokenAttention (LMDeploy)

**本质：block_size=1 的 PagedAttention。**

```
优点：碎片为 0（每 token 1 block = 无内部碎片）
缺点：
  - block table 大小 ≈ seq_len 每个序列（vs vLLM: seq_len/16）
  - kernel 寻址开销大（每次 attention 做 4096 次 block 查询 vs vLLM 的 256 次）
  - GPU 的局部性更差（连续 token 不连续存储）
```

---

## 5. KV Cache 量化

```
KV cache 是 decode 显存大头 → 量化它直接扩 batch / 上下文

常见方案：
  - INT8 KV: 每元素从 2B 到 1B → KV cache 减半
    问题：outlier 导致精度损失
  - FP8 KV: H100+ 原生支持，E4M3 范围和 KV cache 值域匹配良好
  - INT4 KV: 更激进，per-channel/per-group 量化

量化后的 attention kernel 需要：
  输入时 dequant（在 SMEM/寄存器中）
  或者直接 INT4/FP8 matmul（需 Tensor Core 支持）
```

---

## 6. KV Offloading / 分层缓存

```
热 KV → HBM（GPU 显存）
温 KV → CPU DRAM（PCIe 传输）
冷 KV → NVMe SSD（最慢）

Mooncake (2024) 的全局 KV cache pool：
  - 跨节点共享 KV cache
  - P/D 分离中的 KV 传输层
  - RDMA 零拷贝传输
```

---

## 7. 学习检查清单

- [ ] 能手画 PagedAttention 的逻辑→物理 page 映射图
- [ ] 能算 block_size 对内部碎片率的影响
- [ ] 能解释为什么 paged attention kernel 需要改（非连续内存 gather）
- [ ] 能对比 RadixAttention vs APC 的实现差异
- [ ] 能解释 Copy-on-Write 在 beam search 中的应用
- [ ] 能说清 KV cache 量化做什么、怎么做

---

## 8. 自测 / 面试题

1. PagedAttention 把内部碎片降到多少？为什么仍有零头？
2. block_size=16 vs 1 的各自代价？
3. 两个请求共享 100 token 系统 prompt，RadixAttention 和 APC 分别怎么复用？
4. 为什么 paged 之后 attention kernel 不能再假设 KV 连续？kernel 要改什么？
5. KV cache 量化到 INT8 能省多少显存？对 attention 精度影响多大？
6. P/D 分离里 KV 怎么从 prefill 池传到 decode 池？

---

## 9. 推荐阅读

| 资料 | 来源 |
|------|------|
| Efficient Memory Management for LLM Serving (PagedAttention, SOSP'23) | vLLM paper |
| SGLang: Efficient Execution of Structured Language Model Programs | SGLang paper |
| vLLM Automatic Prefix Caching 文档 | vLLM docs |
| LMDeploy TokenAttention | LMDeploy GitHub |
| Mooncake: A KV Cache-Centric Disaggregated Architecture | arXiv |
| KV cache quantization 论文 | INT8/FP8 等工作 |
