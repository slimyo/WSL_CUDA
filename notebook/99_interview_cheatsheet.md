# 面试速查卡：数字 · 公式 · 一句话答案

> 用法: 面试前 1-2 天通刷；平时每学完一章回来核对"这章给速查卡贡献了什么"。
> 所有数字与正文章节一致（dense 算力、修正后的 FLOPs 口径）。

---

## 1. 数字阶梯（必须形成体感）

### 延迟阶梯（GPU 上"远近"的概念）
```
寄存器          ~0 cycle          L2              ~200 cycle
Shared Memory   ~20-30 cycle      HBM             ~400-600 cycle
L1              ~30 cycle         NVLink 跨卡      ~1-2 μs
kernel launch   ~3-10 μs          跨节点 RDMA      ~2-5 μs
                                  PCIe 往返        ~10 μs
```

### 带宽阶梯（GB/s）
```
寄存器/SMEM   ~20,000+ (聚合)   │  NVLink4 (H100)   900     │ PCIe Gen4 x16  32
A100 HBM      2,000             │  NVLink5 (GB200)  1,800   │ PCIe Gen5 x16  64
H100 HBM3     3,350             │  IB NDR / 卡      50      │ NVMe SSD       ~7
RTX 2060      336（你的实验机） │  CPU DDR5         ~100    │
```

### 算力与 ridge point（dense！宣传页 2× 的是稀疏算力）
| 卡 | FP32 | FP16/BF16 TC | FP8 TC | 带宽 | ridge(FP16TC) |
|----|:---:|:---:|:---:|:---:|:---:|
| A100 | 19.5 T | 312 T | —(INT8 624 TOPS) | 2.0 TB/s | 156 |
| H100 SXM | 67 T | 989 T | 1979 T | 3.35 TB/s | 295 |
| RTX 2060 | 6.5 T | ~26 T(fp32 acc) | — | 336 GB/s | ~77 |

### 模型体感数字（Llama-7B 口径，06/09 章推导）
```
权重 FP16: ~13 GB           decode 单步: ~13 GFLOPs, 访存 ~15 GB, I≈0.87
KV/token/层(MHA): 16 KB     prefill 4K: ~62 TFLOPs, I≈4100
KV 4K 上下文 32 层: 2.1 GB   A100 decode 下限: 15GB/2TBps ≈ 7.5 ms/token
DeepSeek-V3 MLA: (512+64)×2B ≈ 1.2 KB/token/层（对比 MHA 等效 64KB, ~57×）
```

## 2. 公式卡（白板必写对）

```
算术强度        I = FLOPs / HBM字节       ridge = P_peak / B_peak
                I > ridge → compute-bound, 否则 memory-bound

GEMM            FLOPs = 2MNK    访存 = (MK+KN+MN)×dtype    方阵 I ≈ N/3×... 用 2MNK/(3N²·b)
模型 forward    FLOPs ≈ 2 × 参数量 × token 数（attention 部分另加 4·L·N²·H_dim·heads 级）

KV cache        = 2 × layers × kv_heads × head_dim × seq × batch × bytes
                （MLA 例外: K/V 共享 latent，= layers × (d_c + d_rope) × seq × batch × bytes）

decode batch    I(B) = FLOPs·B / (权重 + B·KV)  → 渐近线 = FLOPs/KV字节, 摊薄的只有权重
                分组件: 线性层 I≈B（B≈ridge 时转 compute-bound）; attention 永远 memory-bound

Occupancy       驻留 block 数 = min(寄存器墙, smem墙, 线程墙)
                例: 256线程×40reg=10240/block, 65536/10240=6 block → 48/64 warp = 75%

TP all-reduce   每层 2 次, 传的是激活: msg = batch×seq×hidden×bytes（与权重无关!）
Ring all-reduce 每卡通信量 = 2(N-1)/N × 数据量

投机解码        E[接受数] = α+α²+...+α^k = α(1-α^k)/(1-α)   （前缀式接受!）
                拒绝采样: 接受概率 min(1, p/q); 拒绝后从 norm(max(0,p-q)) 重采 → 严格等于 p

碎片            paged 每序列平均浪费 block_size/2 个 token; 连续预分配浪费 60-80%
量化收益        W4A16 decode ≈ (权重/4 + KV) vs (权重+KV) → 7B 级约 2.8×
```

## 3. 一句话答案 Top 20（说出口的版本）

1. **decode 为什么慢**：每生成 1 个 token 要把全部权重+KV 从 HBM 过一遍，算术强度 <1，带宽墙。
2. **batching 为什么"免费"**：权重搬运被 batch 共享，只有 KV 部分线性涨。
3. **prefill/decode 为什么分离**：一个吃算力一个吃带宽，混跑互相干扰且无法独立扩缩。
4. **FlashAttention 快在哪**：不快在计算（FLOPs 不变），快在不往 HBM 写 N² 中间矩阵。
5. **online softmax 核心**：max 和 sum 可以增量维护，旧结果乘 exp(m_old−m_new) 修正。
6. **FlashDecoding**：decode 没有 Q 维可切 → 沿 KV 切分并行再用合并公式归并。
7. **PagedAttention**：OS 分页思想管 KV，碎片从 60-80% 降到 <4%，代价是 kernel 查表 gather。
8. **GQA/MLA**：都是砍 KV——GQA 砍头数，MLA 低秩压缩+矩阵吸收（decode 形如大 head_dim 的 MQA）。
9. **W4A16 vs W8A8**：前者砍权重带宽救 decode，后者用 INT8 TC 算力救 prefill。
10. **SmoothQuant**：把激活的 outlier 难度按通道迁给权重，让 W8A8 可行，迁移本身数学等价。
11. **FP8 vs INT8**：非均匀间隔容纳 outlier，免复杂校准，但要 Hopper+。
12. **TP 为什么不出节点**：每层 2 次 all-reduce 激活，prefill 吃带宽 decode 吃延迟，只有 NVLink 扛得住。
13. **MoE 省什么不省什么**：省 FLOPs 和单 token 权重带宽，不省显存容量；decode 需要更大 batch 才摊得开。
14. **continuous batching**：iteration 级调度，每步动态进出请求，消灭队头阻塞。
15. **chunked prefill**：把 prefill 切块与 decode 同跑，用一点 TTFT 换 TPOT 平稳。
16. **投机解码为何无损**：大模型验证+拒绝采样，输出分布严格等于 target 分布。
17. **CUDA Graph**：录制整段 kernel DAG 一次提交，消 launch 开销，要求 shape 固定（vLLM 按 batch size 录多张）。
18. **Triton 取舍**：交出 thread/smem/同步的控制权，换 10× 开发效率，保留分块策略权。
19. **怎么定位瓶颈**：nsys 看时间去向 → ncu SoL 分类（SM% vs Mem%）→ full set 查根因，手算理论值对照。
20. **pinned memory**：DMA 需要物理地址不动；pageable 的"异步"拷贝会退化成同步+多一次 CPU 复制。

## 4. 高频陷阱提醒（答题时别踩）

```
□ 报算力分清 dense / 2:4 sparse（宣传页是后者）
□ "FLOPs" vs "FLOPS"（量 vs 速率）；TOPS 是整数
□ FlashAttention 不减少计算量、不是近似算法
□ MLA 的 KV 公式不能套 2×（K/V 共享 latent）
□ TP 通信的是激活不是权重；ZeRO 分片的是权重/优化器状态（计算时 gather 回来）
□ 投机解码期望接受数是等比数列不是 k×α
□ fp16 大 K 归约必须 fp32 累加；softmax 在 fp32 logits 上做
□ ncu 下的绝对时间不可信（锁频+回放），性能数字用 cudaEvent
```

## 5. 面试前最后一晚清单

- [ ] 默写: online softmax 递推 + 分块合并公式（10 章 §1）
- [ ] 默写: Triton flash attention（triton/04，30 分钟内）
- [ ] 白板: Llama-7B decode 的 FLOPs/访存/AI 推导（06 章 §3）
- [ ] 白板: Megatron TP 切法图 + all-reduce 位置（16 章 §2）
- [ ] 白板: P/D 分离架构图 + KV 传输链路（13 章 §3）
- [ ] 口述: 一个 token 的生命周期（19 章 §6）
- [ ] 口述: 自己项目的"演进证据链"（P0 §3 报告里的数字）
- [ ] 过一遍本卡 §3 §4
