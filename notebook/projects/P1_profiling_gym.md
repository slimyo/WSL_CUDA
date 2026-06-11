# P1 · Profiling 训练馆：把 nsys/ncu 练成肌肉记忆（1.5-2 周）

> 前置阅读: 17_profiling.md, 06_roofline_and_flops.md, 03_gpu_memory_hierarchy.md
> 产出: ①一套自己的 profiling SOP（命令清单+判定决策树）②4 份 kernel 瓶颈分析报告
> 为什么第一个做它: 后面所有项目的"优化是否有效"都靠这套工具回答。
> 没有它，优化 = 玄学；有了它，每一步都有证据。

---

## 1. 项目定义

**不写新 kernel。** 拿 4 个你已有/已学的 kernel 当"病人"，练"诊断"：

| 病人 | 来源 | 预期病情（先猜后验，训练预判） |
|------|------|------|
| elementwise add (f32 vs f32x4) | LeetCUDA `kernels/elementwise/` | 纯 memory-bound，看带宽利用率天花板 |
| block reduce | 你的 `src/Puzzle/reduce_wrap.cu` | 带宽 + 同步开销，对比 shuffle/smem 版本 |
| naive softmax vs safe vs online | LeetCUDA `kernels/softmax/softmax.cu` | 多次 pass 的冗余 HBM 读 |
| naive sgemm vs tiled | LeetCUDA `kernels/sgemm/sgemm.cu` | 从 memory-bound 优化成 compute-bound 的全过程 |

---

## 2. 工程框架

```
src/projects/p1_profiling/
├── runner.cu            # 统一入口: ./runner <kernel> <size> 方便 ncu 指定目标
├── kernels/             # 4 个病人的各版本拷贝（从 LeetCUDA / Puzzle 抄来改造）
├── scripts/
│   ├── profile_all.sh   # 批量 nsys + ncu 命令（见 §3）
│   └── roofline.py      # 读 ncu csv，画 roofline 散点图
└── reports/             # 4 份诊断报告（用 P0 §3 模板）
```

要点：runner 用命令行参数选 kernel 和规模，这样 `ncu -k <kernel名>` 能精确抓单个
kernel，避免 profile 一堆无关的。每个 kernel 跑前先 `cudaDeviceSynchronize()` 干净隔离。

---

## 3. SOP：三层诊断流程（这是本项目要练出的核心资产）

### 第一层 nsys——"时间花在哪了"（看全局，不看 kernel 内部）

```bash
nsys profile -t cuda,nvtx,osrt -o rep --force-overwrite true ./runner softmax 4096
nsys stats rep.nsys-rep --report cuda_gpu_kern_sum   # kernel 耗时排序
nsys stats rep.nsys-rep --report cuda_gpu_trace      # 时间线明细
```
回答三个问题：
1. GPU 在算的时间占比多少？（gap 多 → launch/CPU bound → CUDA Graph/合并 kernel）
2. 哪个 kernel 是大头？（永远先优化 top-1，see Amdahl）
3. H2D/D2H 拷贝有没有和计算 overlap？
用 NVTX 标注代码段（`nvtxRangePushA("phase1")`）让时间线可读——你之前的
`nvtx.nsys-rep` 实验就是这个，正式用起来。

### 第二层 ncu Speed-of-Light——"这个 kernel 卡在哪类资源"

```bash
ncu --set basic -k regex:softmax -o sol ./runner softmax 4096
# 重点看 GPU Speed Of Light Throughput 区:
#   SM %  vs  Memory %
```
判定决策树（背下来，面试原题）：
```
Memory% 高(>70) SM% 低     → memory-bound → 减访存/提复用 (fusion/tiling/vectorize)
SM% 高(>70) Memory% 低     → compute-bound → 提算力效率 (TC/减冗余计算)
两个都低(<40)              → latency-bound → 看 occupancy / stall 原因（第三层）
两个都低 + kernel 极短(<10μs) → launch-bound → 合并 kernel / CUDA Graph
```

### 第三层 ncu 细查——"为什么"

```bash
ncu --set full -k regex:softmax -o full ./runner softmax 4096
```
按病情查这些 section（GUI 里看更直观，`ncu-ui full.ncu-rep`）：

| 看什么 | Section/指标 | 典型结论 |
|------|------|------|
| 访存效率 | Memory Workload Analysis: `sectors/req` | 32B sector 浪费 → 未合并访问 |
| 带宽实际值 | `dram__bytes.sum.per_second` | 对照 336 GB/s 算利用率 |
| bank conflict | `l1tex__data_bank_conflicts_pipe_lsu` | >0 → smem 布局要改 |
| occupancy 受限原因 | Occupancy: limiter (reg/smem/blocks) | 寄存器超了→`__launch_bounds__` |
| stall 原因 | Warp State Statistics: `stall_long_scoreboard` 等 | long_scoreboard=等 global 数据 |
| 源码级定位 | Source Counters（需 `-lineinfo`） | 哪一行产生了未合并访问 |
| roofline | `ncu --set roofline` | 算术强度 + 离屋顶距离 |

---

## 4. 分步任务（每步有验收）

**Step 1（2 天）打通工具链**
elementwise f32 → 跑通三层 SOP。验收：算出它的带宽利用率（应 >80%，
若不到，找出原因——这是"理论上最简单的 kernel"，到不了上限说明环境/测法有问题，
正好排雷：WSL 计数器权限、boost 频率波动（`nvidia-smi -lgc` 锁频）、规模太小）。

**Step 2（2 天）量化"向量化"的收益**
f32 vs f32x4 vs f16x8 版本对比。验收报告回答：向量化在 ncu 里改变了哪个指标？
（提示：指令数下降、`sectors/req` 不变、带宽利用率上升——理解"省指令发射"而非"省流量"）

**Step 3（3 天）softmax 三版本的 HBM 流量审计**
naive(3 pass)/safe(3 pass)/online(2 pass) + fused 一个 block 内完成（1 pass）。
验收：用 `dram__bytes_read/write.sum` 实测每版本字节数，和手算的理论值对上
（误差 <15%）。**"手算访存量 → ncu 验证"是算子岗核心基本功。**

**Step 4（3 天）sgemm naive→tiled 的瓶颈迁移**
验收报告回答：naive 版的瓶颈是什么（DRAM）？tiled 之后迁移到了哪
（L1/smem 或 compute）？roofline 上两个点各落在哪？

**Step 5（1 天）沉淀 SOP**
把 §3 决策树 + 你的实测经验整理成一页 `profiling_SOP.md`。这一页就是你面试答
"怎么定位 kernel 瓶颈"的标准答案，而且全是自己跑出来的数字。

---

## 5. 关键能力（= 面试考点）

1. **三层归因方法论**：nsys(全局) → ncu SoL(分类) → ncu full(根因)，不跳层不猜
2. **手算与实测互验**：访存量、FLOPs、利用率，先算后测
3. **指标→优化的映射**：每个坏指标都能立刻说出 2 个对应优化手段
4. 知道工具的盲区：ncu 会锁频+串行回放（绝对时间失真，看比率指标）、
   nsys 采样开销、短 kernel 测量噪声

## 6. 扩展方向

- PyTorch profiler + holistic trace：profile 一次 transformers 推理，找出 decode
  step 里的 launch gap（衔接 17 章 CUDA Graph 动机）
- 写一个 `roofline.py` 把所有项目的 kernel 自动画到同一张 roofline 图（后续项目通用）
- Nsight Systems 的 GPU metrics sampling（采 SM 利用率时间线，不用 ncu 也能看粗粒度）

## 7. 参考

- LeetCUDA: `kernels/nvidia-nsight/README.md`、`bank_conflicts.md`（现成的 ncu 实验）
- NVIDIA: Nsight Compute Kernel Profiling Guide（指标定义的唯一权威）
- 博客: "Nsight Compute 里每个 section 怎么读"——CUDA Mode (GPU MODE) lecture 1/16
- notebook: 17 章决策树、06 章 roofline、bank_conflict_learning.md
