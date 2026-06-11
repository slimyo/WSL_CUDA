# 采样与解码：logits → token 的最后一公里

> 对象: 推理岗（模型 forward 之后、token 返回之前的全部环节——此前体系的空白带）
> 前置: 08_transformer_architecture.md, 09_inference_workload.md, 13_scheduling.md §4
> 目标: 面试能写 top-k/top-p 的流程与复杂度、证明投机解码数学无损、
>       讲清 constrained decoding 怎么实现

---

## 1. 采样管线全景

```
hidden[B, H] ─LM Head GEMM→ logits[B, V]   (V = 32K~256K, 词表越来越大!)
   ↓ ① logits 处理器链（顺序敏感）
   repetition / presence / frequency penalty   （按已生成 token 修改 logits）
   temperature:  logits /= T                   （T→0 趋近 greedy, T>1 更随机）
   top-k:        只保留最大的 k 个             （需要 [B,V] 的 top-k —— 不便宜!）
   top-p(核采样): 按概率降序累加到 p 截断       （需要排序 —— 更不便宜!）
   min-p:        保留 prob ≥ p·max_prob 的     （2024+ 流行，只需 max，便宜）
   ↓ ② softmax（fp32! logits 上做，别在 fp16 上 softmax）
   ↓ ③ 抽样: multinomial / gumbel-max（greedy 则直接 argmax）
token id → detokenize → 流式返回
```

**系统视角三个事实：**
1. 这段是 **decode 每步都要跑** 的，B 大时 [B, V] 的 sort/topk 开销可观
   （V=128K, B=256 → 每步处理 32M 个 logit）——所以工业引擎全部 fuse 成专用 kernel
2. 不同请求的采样参数各不相同（batch 内异构）→ kernel 要按行带参数
3. 它在 CUDA Graph 里（20 章）→ 随机数状态、参数都要做成 tensor 输入而非 Python 分支

## 2. Kernel 视角：top-p 怎么做才不爆炸

```
朴素: 全词表 sort（V·logV）→ 前缀和 → 截断 → 再归一化 → multinomial
FlashInfer 的拒绝采样法（工业 SOTA，思想优美值得记）:
  不排序! 直接从完整分布采一个 token，检查它是否落在 top-p 集合内
  （用一个 pivot 概率判断），不在则调整 pivot 重采——期望几轮内命中
  → 把 O(V logV) 的 sort 换成 O(V) 的几轮扫描，GPU 友好（无全局排序）
Gumbel-max 技巧: argmax(logits + Gumbel noise) ≡ 按 softmax 概率抽样
  → 抽样变成一次 reduce-argmax，天然可并行/可融合
```
对照源码：vLLM `csrc/sampler.cu` / FlashInfer `sampling.cuh`；
练手：06 篇练习风格——写一个 fused temperature+min-p+gumbel 采样 Triton kernel。

## 3. 投机解码的数学：为什么严格无损（13 章 §4.3 的补全）

```
draft 分布 q(x)，target 分布 p(x)。对 draft 抽出的 token x:
  以 min(1, p(x)/q(x)) 的概率【接受】
  若拒绝 → 从修正分布 norm(max(0, p−q)) 中重采一个 token

证明一行版: P(输出=x) = q(x)·min(1, p/q) + P(拒绝)·norm(max(0,p−q))(x)
            两项相加恰好 = p(x)   ∀x      （分两种情况 p≥q / p<q 验证即可）

推论（面试常追问）:
  - 接受率 = Σ_x min(p(x), q(x))，即两个分布的重叠度 → draft 越像 target 越快
  - greedy（T=0）时退化为 "draft token == target argmax 才接受"
  - 这就是"无损"的含义: 输出分布 = 纯 target 采样的分布，不是"近似没差"
```

## 4. Constrained / Structured Decoding（agent 时代的必备件）

```
需求: 强制输出合法 JSON / 符合 grammar / 只能选工具名 —— 18 章 §1.2 的工作负载
原理: 每步采样前，用一个【状态机】算出"当前语法状态下合法的 token 集合"，
      把非法 token 的 logit 置 −inf（bitmask），然后正常采样
        JSON schema → 正则/CFG → 编译成 FSM（token 级, 不是字符级——难点!
        一个 token 可能跨多个语法单元, 需要预编译 token→状态转移表）
工业实现: outlines / xgrammar（SGLang 内置）/ llguidance；
          vLLM/SGLang 都支持 response_format=json_schema
系统问题: FSM 状态推进在 CPU、apply bitmask 在 GPU → 又一个 overlap 点（20 章）;
          掩码后大量 −inf → softmax 的数值处理（max 还是要先减）
```

## 5. 其他解码策略速览

| 策略 | 一句话 | 现状 |
|------|------|------|
| beam search | 维护 top-B 条路径取整体最优 | LLM 时代基本退役（开放生成质量差、KV 要 fork——但它是 PagedAttention COW 的设计动机之一, 12 章） |
| best-of-n | 采 n 条完整回答选最好 | RLHF/推理时扩展常用，吃吞吐 |
| parallel sampling | 一个 prompt 采 n 条（共享 prefill KV） | 引擎用 COW 高效支持 |
| logprobs 返回 | 每步返回 top-k 概率 | API 标配，注意它强制了一次 topk |

## 6. 学习检查清单

- [ ] 能默写采样管线顺序，说清每个处理器改 logits 还是改 prob
- [ ] 能解释为什么 top-p 比 min-p 贵、工业 kernel 怎么避免全排序
- [ ] 能在白板证明投机解码的拒绝采样恒等式
- [ ] 能讲 constrained decoding 的 FSM+bitmask 机制和 token 级编译难点
- [ ] 知道采样和 CUDA Graph / batch 异构参数的工程纠缠

## 7. 自测 / 面试题

1. temperature→0 和 →∞ 分别极限成什么？top-k=1 等于什么？
2. V=128K、B=128，朴素 top-p 每个 decode step 的开销量级？怎么优化？
3. 证明: 拒绝采样后输出分布恰为 p(x)。接受率的表达式？
4. 为什么 softmax 要在 fp32 logits 上做？fp16 会出什么问题？（溢出 + 长尾下溢）
5. JSON mode 下生成为什么可能变慢？（mask 计算、FSM 串行性、token 化不对齐）
6. 同一 batch 里 A 要 greedy、B 要 top-p=0.9，kernel 怎么设计？

## 8. 参考

- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding"（§3 数学的原始出处, 附录有完整证明）
- FlashInfer sampling 文档/源码（拒绝采样 top-p 的实现）
- xgrammar / outlines 论文与 README（constrained decoding 两条路线）
- vLLM: `vllm/v1/sample/` 目录（logits processor 链的工业形态）
- notebook: 13 章 §4（系统视角）、18 章 §1.2（agent 负载）、20 章（graph 纠缠）
