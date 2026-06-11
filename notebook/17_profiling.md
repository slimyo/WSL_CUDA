# Profiling 与工程横切：Nsight Systems / Nsight Compute / CUDA Graph

> 对象: 算子岗位（必达）
> 前置: 06_roofline_and_flops.md, 04_warp_execution_model.md
> 目标: 面试能定位 kernel 瓶颈并给出优化方向
> 参考 LeetCUDA: `nvidia-nsight/`

---

## 1. 为什么 profiling 是算子岗的必备技能

**面试常问："你的 kernel 怎么优化？瓶颈在哪？"**
不能回答"我猜是 memory-bound"，要用 profiler 数据和 roofline 来证明。

---

## 2. Nsight Systems (nsys) — 全局视角

### 2.1 作用

```
看全局：kernel launch 间隔、CPU-GPU 同步、通信重叠、显存操作

适合回答：
  - "我的 kernel 慢在哪一阶段？"
  - "有没有做无用的同步？"
  - "CPU 有没有成为瓶颈？"
  - "通信（all-reduce）是不是 overlap 了计算？"
```

### 2.2 常用命令

```bash
nsys profile -o output_file -t cuda,nvtx ./my_program
nsys stats output_file.nsys-rep   # 汇总统计
```

### 2.3 关键看什么

```
1. Kernel launch gap（核函数启动的间隔）
   如果 launch gap 占大部分时间 → CPU 调度/launch overhead 瓶颈

2. GPU 空闲区间
   计算单元长时间不工作 → 带宽受限或 launch bound

3. CUDA memcpy 和 kernel 的重叠
   memcpy 如果和 kernel 串行 → 没 overlap → 优化空间

4. CUDA 同步
   cudaDeviceSynchronize 是否频繁
```

---

## 3. Nsight Compute (ncu) — 单 kernel 分析

### 3.1 作用

```
看单 kernel 内部：occupancy、访存吞吐、计算吞吐、roofline

常用命令：
  ncu --set full -o kernel_report ./my_program
  ncu --print-summary per-kernel ./my_program
  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
      --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./my_program
```

### 3.2 关键指标解读

| 指标 | 含义 | 好 | 差 |
|------|------|:---:|:---:|
| sm__throughput.avg.pct_of_peak | SM 计算吞吐（% of peak） | > 60% | < 20% |
| dram__throughput.avg.pct_of_peak | 显存带宽利用率（% of peak） | > 80% | < 30% |
| sm__occupancy | 每 SM 驻留 warp 数 / max | > 50% | < 25% |
| l1tex__throughput.avg.pct_of_peak | L1/SMEM 吞吐 | > 60% | < 30% |
| sm__warps_active.avg.pct_of_peak | 活跃 warp 比例 | 高 | 低 → latency bound |
| launch__registers_per_thread | 每线程寄存器数 | ≤ 32 | > 64 → spilling |
| l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate | L1 cache hit rate | > 80% | < 50% |

**瓶颈判定逻辑：**
```
if dram_throughput > 70% and sm_throughput < 40%:
    瓶颈 = memory-bound（带宽吃满，计算闲）
    → HBM 带宽是瓶颈

if sm_throughput > 60% and dram_throughput < 50%:
    瓶颈 = compute-bound（计算吃满，带宽够用）
    → 算力是瓶颈

if dram < 40% and sm < 30% and warps_active < 30%:
    瓶颈 = latency-bound（occupancy 低、stall 多）
    → 减少寄存器使用、增加指令级并行

if kernel launch 密集且 kernel 很轻:
    瓶颈 = launch-bound（kernel launch overhead > kernel 运行时间）
    → CUDA Graph 或 kernel fusion
```

### 3.3 Occupancy 分析

```
occupancy = active_warps / max_warps_per_sm

影响因素：
  - 每线程寄存器数（最多 255/thread）
  - shared memory 用量
  - block size（1024 max threads/block）

A100:
  max_warps_per_sm = 64
  max_threads_per_sm = 2048
  每 SM 最多 64 warp → 算一下 block_size=256 时能驻留几个 block

面试必算：
  block_size=256, 每 thread 用 40 个 reg → 256×40 = 10240 寄存器/block
  A100 每 SM 65536 regs → 65536/10240 = 6.4 → 最多 6 个 block 驻留
  （同时检查线程上限：2048/256 = 8 block，寄存器先卡住 → 取 6）
  → 6 × 8 warps = 48 warps → occupancy = 48/64 = 75%
```

### 3.4 Nsight Roofline 分析

```
ncu 内置 Roofline section：
  ncu --set roofline -o report ./my_program     # 或 --set full（包含 roofline）
  在 GUI 的 "GPU Speed Of Light Roofline" 图上标注 kernel 位置

直接读出：arithmetic intensity、离 roofline 的距离、bound 类型
```

---

## 4. CUDA Graph — 消除 Launch Overhead

### 4.1 问题

```
decode 每步有几十个小 kernel（RMSNorm × 2 + attention + FFN + ...）
每个 kernel launch 都有 ~5μs 的 CPU 开销

100 个小 kernel × 5μs = 500μs → 如果每步只有几 ms 的话，这是很大比例
```

### 4.2 解法：CUDA Graph

```
一次构建（capture）多步 kernel launch → 一次 replay

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  kernel1<<<grid, block, 0, stream>>>(...);
  kernel2<<<grid, block, 0, stream>>>(...);
  // ...
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, 0);   // 编译成可执行 graph（一次）

// 之后每步只需 replay（一次 launch 执行全部 kernel）
cudaGraphLaunch(graphExec, stream);
```

**收益：**
- 减少 CPU-GPU 同步次数
- kernel launch 从 N × 5μs → N × 0.1μs
- decode 场景实测 1.2-1.5× 加速

### 4.3 局限

```
- Dynamic shape：graph 需要固定 shape（控制流/数据依赖需要条件）
- 构建慢：不适用于 kernel 参数经常变动的场景
- 适合 decode（shape 稳定，小 kernel 多），不适合 prefill（param 变化大）
```

---

## 5. Benchmark 方法论

```
正确的 benchmark 做法：
  - Warmup：避免第一次 launch 的初始化开销
  - 锁频：CPU/GPU 频率不稳定的影响
  - 固定 batch/seq_len：避免动态变量
  - 区分测量：TTFT / TPOT / throughput 分开测
  - 多次跑取中位数：避免噪声影响
  - 用 nsys 排除无关时间段

错误的 benchmark：
  - 只跑一次 / 不 warmup
  - 混在一起算延迟
  - 直接用 time.time() 包住 kernel 调用——CUDA kernel 是异步的，
    不先 torch.cuda.synchronize() 测到的只是 launch 时间
  - 正确做法：cuda Event 计时（event.record + elapsed_time）
    或 triton.testing.do_bench
```

---

## 6. LeetCUDA Nsight 参考

| LeetCUDA 目录 | 内容 |
|------|------|
| `nvidia-nsight/README.md` | Nsight 使用说明 |
| `nvidia-nsight/bank_conflicts.md` | Bank conflict profiling |
| `nvidia-nsight/elementwise.cu` | elementwise 带宽测试 |
| `nvidia-nsight/relu.cu` | ReLU profiling 示例 |

---

## 7. 学习检查清单

- [ ] 能用 Nsight Systems 查看全局 timeline
- [ ] 能用 Nsight Compute 分析单 kernel 的瓶颈
- [ ] 能根据 ncu metrics 判断 memory/compute/latency/launch-bound
- [ ] 能讲出 CUDA Graph 解决了什么、decode 场景的收益
- [ ] 能讲正确的 benchmark 方法论
- [ ] 能在 LeetCUDA 中找到 profiling 示例

---

## 8. 自测 / 面试题

1. 给一个慢的 kernel，你怎么用 Nsight 判断瓶颈、定优化方向？
2. decode 为什么适合 CUDA Graph？prefill 为什么不适合？
3. 一个 kernel occupancy=30%、dram_throughput=90%、sm_throughput=20%，瓶颈是什么？
4. 怎么测一个 kernel 是 compute-bound 还是 memory-bound？用 ncu 的哪个指标？
5. 你的 profile 显示 launch gap 占 30%，你怎么优化？

---

## 9. 推荐阅读

| 资料 | 来源 |
|------|------|
| Nsight Compute / Nsight Systems 教程 | NVIDIA Developer |
| CUDA Graph 文档 | NVIDIA CUDA Programming Guide |
| CUDA C++ Best Practices Guide | NVIDIA |
| PyTorch Profiler | PyTorch |
| LeetCUDA nvidia-nsight/ | `/third_party/LeetCUDA/kernels/nvidia-nsight/` |
