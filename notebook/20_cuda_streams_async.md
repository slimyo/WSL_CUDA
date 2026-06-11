# CUDA 并发与异步执行：Stream / Event / Pinned Memory / Overlap

> 对象: 算子与系统岗（01-05 讲的是"单 kernel 怎么快"，本章讲"多任务怎么并发"）
> 前置: 02_cuda_programming_model.md, 03_gpu_memory_hierarchy.md, 17_profiling.md
> 目标: 面试能画 H2D/compute/D2H 三路 overlap 的时间线、讲清 pinned memory 为什么必须
> 你的存货: 你已做过 `compute-io-overlap.nsys-rep` / `pinned.nsys-rep` 实验，本章是其理论化

---

## 1. 心智模型：GPU 是一台"多队列异步设备"

```
CPU 发命令（launch/memcpy）≠ GPU 立刻执行。命令进入 stream（FIFO 队列）：
  同一 stream 内: 严格按提交顺序执行（天然依赖）
  不同 stream 间: 无顺序保证 → 硬件资源允许时【并行】

GPU 上可并行的三类引擎（Turing 起均独立）:
  Compute Engine     跑 kernel（SM 资源够时多 kernel 并发）
  Copy Engine (H2D)  PCIe 上行 DMA
  Copy Engine (D2H)  PCIe 下行 DMA
→ 理论上同一时刻可以: 算 chunk[i] + 上传 chunk[i+1] + 回传 chunk[i-1]
```

**默认流（default/null stream）的坑**：legacy 默认流和所有其他流互相隐式同步——
这是"为什么我开了多 stream 还是串行"的第一嫌疑。编译加
`--default-stream per-thread` 或显式创建非默认流。

## 2. Pinned Memory：异步拷贝的前提（高频考点）

```
为什么 cudaMemcpyAsync 必须搭配 pinned（页锁定）内存？
  GPU 的 DMA 引擎按【物理地址】搬数据。普通 pageable 内存随时可能被 OS 换页
  → 物理地址会变 → DMA 不敢直接读
  → pageable 拷贝实际流程: CPU 先把数据复制进驱动的 pinned staging buffer，再 DMA
     （多一次 CPU memcpy，且这步是同步的 → "Async"名不副实）
  → cudaMallocHost/cudaHostAlloc 申请的 pinned 内存: OS 承诺不换页 → DMA 直读

实测差异（你的 pinned.nsys-rep 应能看到）:
  pageable: ~2-6 GB/s 且占 CPU    pinned: 接近 PCIe 上限（Gen3 x16 ≈ 12 GB/s, 本机）
注意: pinned 内存是稀缺资源（占物理内存、加重 OS 管理负担），别无脑全 pinned。
```

## 3. 三路 Overlap 的标准范式（手写考题）

```cuda
// 把大任务切 N 个 chunk，轮转使用 K 个 stream（K=2~4 足够）
cudaStream_t s[K];
for (int i = 0; i < K; i++) cudaStreamCreate(&s[i]);

for (int c = 0; c < N; c++) {
    int k = c % K;
    cudaMemcpyAsync(d_in + off(c), h_in + off(c), bytes, H2D, s[k]);
    kernel<<<grid, block, 0, s[k]>>>(d_in + off(c), d_out + off(c));
    cudaMemcpyAsync(h_out + off(c), d_out + off(c), bytes, D2H, s[k]);
}
cudaDeviceSynchronize();
```
```
时间线（nsys 里应看到的阶梯）:
stream0:  H2D₀ K₀ D2H₀        H2D₂ K₂ D2H₂
stream1:       H2D₁ K₁ D2H₁        H2D₃ K₃ D2H₃
加速上限 = max(T_h2d, T_kernel, T_d2h) 取代三者之和（流水线原理）
```
**验收实验**（把你旧的 overlap 实验正规化）：chunk 数 1/2/4/8 扫描，
nsys 截图三条引擎轨道，验证总时间逼近 max() 而非 sum()。

## 4. Event：计时与跨流依赖

```cuda
// 用途 1: 精确计时（17 章 benchmark 方法论的实现）
cudaEventRecord(t0, s);  ...  cudaEventRecord(t1, s);
cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);

// 用途 2: 跨流依赖（DAG 编排的原语）——producer 流的结果给 consumer 流用
cudaEventRecord(evt, s_producer);
cudaStreamWaitEvent(s_consumer, evt, 0);   // s_consumer 后续命令等 evt，CPU 不阻塞
```
工业对应：推理引擎里"通信流 等 计算流"（NCCL overlap）、P/D 分离里 KV 传输
与下一层 prefill 的重叠，底层全是 `StreamWaitEvent` 这一个原语。

## 5. 现代 API 速览（面试加分项）

| API | 解决什么 | 一句话 |
|------|------|------|
| `cudaMallocAsync` / 内存池 | cudaMalloc 是全局同步点（毁 overlap） | stream-ordered 分配，PyTorch caching allocator 的官方版 |
| CUDA Graph（17 章§4 进阶） | launch 开销 + CPU 调度抖动 | 把整段多流 DAG 录制下来一次提交；vLLM 对每个 batch size 录一张 decode graph（所以要 padding 到固定 shape） |
| `cudaLaunchHostFunc` | 在流里插 CPU 回调 | 替代轮询，注意回调里不能再调 CUDA API |
| Stream priority | decode 流不被 prefill 流饿死 | `cudaStreamCreateWithPriority`，chunked prefill 的 kernel 级补充 |
| MPS | 多进程共享 GPU 时 SM 真并发 | 多模型混部/多租户场景 |

## 6. 与推理系统的连接（为什么本章重要）

```
vLLM/SGLang 的单步 decode 里并发着什么:
  计算流: 几十个 kernel（CUDA Graph 回放）
  通信流: TP 的 all-reduce（NCCL 有自己的流，靠 event 与计算流握手）
  拷贝流: 下一批请求的输入上传 / KV swap-in/out（抢占恢复, 13 章）
  CPU:    调度器在准备下一个 iteration 的 batch（overlap scheduler, 18 章§4.2）
所有"overlap"宣传语，拆开都是本章的 stream + event。
```

## 7. 学习检查清单

- [ ] 能解释 pinned vs pageable 的 DMA 原理差异，知道 Async 退化成同步的条件
- [ ] 能手写三路 overlap 的多 stream 循环，并画出理想时间线
- [ ] 能用 nsys 验证 overlap 是否真发生（三条引擎轨道）
- [ ] 能讲 cudaStreamWaitEvent 如何表达跨流 DAG
- [ ] 知道 cudaMalloc/默认流/同一 stream 串行这三个"隐式串行点"
- [ ] 能说清 vLLM 为什么按 batch size 录多张 CUDA Graph

## 8. 自测 / 面试题

1. cudaMemcpyAsync 传了 pageable 指针会发生什么？（不报错！但行为退化——说清退化成什么）
2. 两个 kernel 提交到两个 stream 却没并发，列三种可能原因（SM 占满/默认流/隐式同步 API）
3. 设计：1GB 数据 H2D 12GB/s、kernel 处理 10ms、D2H 12GB/s，单流总时间？4 流流水后？
4. CUDA Graph 为什么要求 capture 期间 shape/地址固定？vLLM 怎么绕开动态 batch 的矛盾？
5. NCCL all-reduce 怎么和计算 overlap？哪个原语保证正确性？

## 9. 参考

- CUDA C++ Programming Guide: Asynchronous Concurrent Execution 章节
- NVIDIA blog: "How to Overlap Data Transfers" / "CUDA Graphs" 两篇经典
- 你的旧实验: `compute-io-overlap.nsys-rep`, `pinned.nsys-rep`（重新打开对照本章）
- vLLM 源码: `vllm/worker/model_runner.py` 的 CUDA Graph capture 部分
