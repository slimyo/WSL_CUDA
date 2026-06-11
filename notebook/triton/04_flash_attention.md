# Triton 04 · Flash Attention：把 01-03 拼成面试默写件

> 目标: 不看资料 30 分钟写出 causal flash attention forward
> 时长: 3-4 天 ｜ 前置: 02 篇 §3 (online 循环)、03 篇 (tl.dot)、10 章 §1-2
> 与 P3 的关系: P3 是完整项目（含 CUDA 版+实验报告），本篇是其中 Triton 部分的
> 精讲+默写训练。先读本篇再做 P3 Step4，或做完 P3 回来做默写训练，皆可。

---

## 1. 组装视角：你已经会了所有零件

```
flash attention = 03 篇的 matmul 外壳
                + 02 篇 §3 的 online softmax 循环体
                + 一个 [BLOCK_M] 维的 (m, l) 状态（从标量升级为按行向量）
新东西只有两个: ① O 累加器的 rescale ② causal mask 的块级处理
```

## 2. 参考实现（先理解，后默写）

```python
@triton.jit
def flash_fwd(Q, K, V, O, sm_scale,
              sqh, sqm, skh, skn, svh, svn, soh, som,   # 各 tensor 的 (head, seq) stride
              H, N, D: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)            # Q 块编号   ← FA2 的 seq 维并行(10章§3.1)
    pid_h = tl.program_id(1)            # batch*head

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + pid_h*sqh + offs_m[:, None]*sqm + offs_d[None, :],
                mask=offs_m[:, None] < N, other=0.)            # [BM, D] 常驻寄存器

    m_i = tl.full((BM,), -float('inf'), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)

    hi = (pid_m + 1) * BM if CAUSAL else N        # causal: 只扫到对角块为止 ①
    for start_n in range(0, hi, BN):
        k = tl.load(K + pid_h*skh + (start_n+offs_n)[:, None]*skn + offs_d[None, :],
                    mask=(start_n+offs_n)[:, None] < N, other=0.)     # [BN, D]
        s = tl.dot(q, tl.trans(k)) * sm_scale                          # [BM, BN]
        if CAUSAL:                                                     # ②对角块内细粒度 mask
            s = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], s, -float('inf'))
        s = tl.where((start_n + offs_n)[None, :] < N, s, -float('inf'))  # 边界

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p     = tl.exp(s - m_new[:, None])                 # [BM, BN]
        alpha = tl.exp(m_i - m_new)                        # 旧状态的 rescale 因子
        l_i   = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + pid_h*svh + (start_n+offs_n)[:, None]*svn + offs_d[None, :],
                    mask=(start_n+offs_n)[:, None] < N, other=0.)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)   # ③
        m_i = m_new

    acc = acc / l_i[:, None]                               # ④归一化只在最后
    tl.store(O + pid_h*soh + offs_m[:, None]*som + offs_d[None, :],
             acc.to(tl.float16), mask=offs_m[:, None] < N)
```

启动：`grid = (cdiv(N, BM), B*H)`；SM75 建议 BM=64, BN=64, D=64, num_warps=4。

## 3. 四个标注点 = 四个必考细节

```
① causal 的块级跳过: hi = (pid_m+1)*BM —— 整块在上三角的直接不进循环，
   耗时近减半。只做②不做①是"对而不快"的典型。
② 对角块内仍要逐元素 mask（块内一半位置非法）。
③ p 转 fp16 才能进 tl.dot（tensor core 吃 fp16），但 acc/l/m 全程 fp32
   ——精度纪律（02 篇 §4 的延续）。
④ 循环内维护的是"未归一化"acc，除 l_i 只做一次（10 章 §2.1 修正版伪代码）。
```

## 4. 默写训练法（面试的真实形态）

```
第 1 遍: 对照本文敲一遍，跑通 verify
第 2 遍: 只看 §1 的"组装视角"三行提示，重写。卡住的地方就是你的薄弱点
第 3 遍: 空白文件 + 计时 30 分钟。验收 = verify 通过
高频卡点排行（亲测）: rescale 漏乘 alpha / mask 的 other 填错 /
   stride 算错（用 §2 的命名习惯可避免）/ causal 边界 off-by-one（用 N=BM+1 测!）
```

## 5. 验收与对照

1. 正确性：vs `torch.nn.functional.scaled_dot_product_attention`（causal/非 causal 各测）
2. 性能：seq=2048, H=8, D=64 时 ≥ torch SDPA(flash 后端) 的 70%
3. ncu：HBM 读写量级 = O(N·D)（对照 P3 naive 版的 O(N²) 数据）
4. 对照 LeetCUDA `kernels/openai-triton/fused-attention/` 和官方 tutorial 06——
   找出它们比你多做的优化（如 stage 间 K/V 预取、`tl.multiple_of` 提示）

## 6. 扩展（按面试价值排序）

1. **GQA**：`kv_head = pid_h % H_KV` 一行映射——11 章 §2.3 的实现版，5 分钟改完，
   但能在简历写"支持 GQA"
2. **decode 模式**：BM=1 时本 kernel 退化为 P4 的 v1；接 P4 的 split-KV 改造
3. 返回 logsumexp（`m_i + tl.log(l_i)`）——backward 和 split 合并都需要它
4. sliding window attention：把 lo 也抬起来（`lo = max(0, (pid_m+1)*BM - W)`）
   ——Mistral/gpt-oss 风格（18 章 §2.3 的 hybrid 中常见）

## 7. 要点回顾

- flash attention 在 Triton 里 = matmul 外壳 + online softmax 内核，没有新魔法
- 四个细节（块级 causal、对角 mask、fp16/fp32 分工、最后归一化）决定对不对
- 这个 kernel 写熟后，你拥有了：面试手写题、P3/P4 项目核心、读 FlashInfer
  源码的入场券
