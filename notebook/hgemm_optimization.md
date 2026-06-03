# HGEMM 优化全流程

> TODO.md: 阶段4 — 从能跑到 98% cuBLAS
> 前置: `tensor_cores_intro.md`
> 参考: `third_party/LeetCUDA/kernels/hgemm/`, `kernels/swizzle/`

## 1. 优化路线图

```
WMMA baseline → Multi-Stage Pipeline → Register Double Buffer
→ SMEM Padding → Block Swizzle → 98% cuBLAS
```

每步用 ncu 测量, 记录 throughput 变化。

## 2. Multi-Stage Pipeline (cp.async)

**问题**: global→shared 拷贝期间 GPU 空等。
**解决**: producer-consumer pipeline, 用多个 shared memory stage。

```cuda
// 4-stage pipeline
__shared__ half smem_A[4][BLOCK_M][BLOCK_K];
__shared__ half smem_B[4][BLOCK_K][BLOCK_N];

// Producer: 异步预取 (cp.async)
for (int stage = 0; stage < 4; stage++) {
    cp_async(&smem_A[stage][...], &A[...]);
    cp_async(&smem_B[stage][...], &B[...]);
}
cp_async_commit_group();

// Consumer: 等 stage 0 就绪后开始计算
cp_async_wait_group<3>();
// ... mma_sync on smem[0] ...
__syncthreads();

// Pipeline: 加载 stage_{n+3} 的同时计算 stage_n
```

`cp.async` 是 SM80+ 的异步拷贝指令, 不占用计算单元。

## 3. Register Double Buffer

A/B fragment 各两份, 交错加载和计算:

```
Cycle 1: load A[0],B[0] → compute A[0]×B[0]
Cycle 2: load A[1],B[1] → compute A[1]×B[1]  (A[0]已用完)
Cycle 3: load A[2],B[2] → compute A[2]×B[2]  (A[1]已用完)
```

减少计算单元等数据的时间。

## 4. Shared Memory Padding

详见 `bank_conflict_learning.md`:
```cuda
__shared__ half smem_B[BLOCK_K][BLOCK_N + 1];  // +1 padding 消除 conflict
```

## 5. Block Swizzle

**问题**: block 顺序访问 global memory, 相邻 block 访问不重叠的数据 → L2 cache miss。
**解决**: 重排 block 计算顺序, 让相邻 block 的数据在 L2 中复用。

参考 `kernels/swizzle/` 的实现。

## 6. 性能测量

每个优化步骤后:
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./hgemm
```

关注三个指标:
- SM occupancy (越高越好)
- Compute throughput (% of peak)
- Memory throughput (% of peak)

瓶颈在哪就优化哪 (compute-bound → 减少冗余计算, memory-bound → 增加复用)。

## 7. 推荐阅读

1. `kernels/hgemm/` WMMA → MMA → MultiStage 渐进
2. `kernels/swizzle/` — swizzle layout
3. `kernels/hgemm/` Multi-Stage 版本 — 最完整优化

