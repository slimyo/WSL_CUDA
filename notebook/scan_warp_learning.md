# Warp-Level Prefix Sum 学习笔记

> 对应文件: `src/Puzzle/scan_wrap.cu`
> 前置知识: Puzzle 10, `reduce_warp_learning.md`

---

## 1. Scan vs Reduce — 关键区别

| | Reduce | Scan (Prefix Sum) |
|---|--------|-------------------|
| 输入 | [a, b, c, d] | [a, b, c, d] |
| 输出 | 1 个值: a+b+c+d | 4 个值: [a, a+b, a+b+c, a+b+c+d] |
| 数据流 | 聚合到 0 号线程 | **每个线程**都需要自己的前缀和 |

你的 puzzle10 `block_sum_kernel` 其实是 reduce:
```cuda
if (local_idx == 0) out[blockIdx.x] = cache[0];  // 只产出 1 个值
```

真正的 scan 产出 N 个值，每个位置都有结果。

---

## 2. Warp Inclusive Scan (Hillis-Steele)

算法: offset 从 1,2,4,8,16 逐步翻倍:

```
初始:           [a,    b,    c,    d   ]
off=1 (shfl_up): [a,   a+b,  b+c,  c+d ]
off=2 (shfl_up): [a,   a+b, a+b+c, a+b+c+d]
```

代码:
```cuda
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_inclusive_scan_f32(float val) {
  int lane = threadIdx.x % kWarpSize;
  #pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    float tmp = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset) val += tmp;
  }
  return val;
}
```

**为什么必须用 `tmp` 暂存?**
```cuda
// ✅ 正确 — 先读旧值，再累加
float tmp = __shfl_up_sync(0xffffffff, val, offset);
if (lane >= offset) val += tmp;

// ❌ 错误 — shuffle 时 val 已被前几轮修改
val += __shfl_up_sync(0xffffffff, val, offset);
```

## 3. Warp Exclusive Scan

```cuda
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_exclusive_scan_f32(float val) {
  int lane = threadIdx.x % kWarpSize;
  #pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    float tmp = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset) val += tmp;
  }
  float result = __shfl_up_sync(0xffffffff, val, 1);
  return (lane == 0) ? 0.0f : result;
}
```

---

## 4. Block-Level Scan — 三阶段

```
阶段 1: WARP-SCAN (寄存器, 0 sync)
阶段 2: CROSS-WARP (1 次 sync, warp0 scan warp_totals)
阶段 3: ADD OFFSET (warp_i += sum of warp_0..warp_{i-1})
```

图解 (2 warps × 4 lanes):
```
输入: [a,b,c,d, e,f,g,h]
阶段1: warp0=[a, a+b, a+b+c, T0], warp1=[e, e+f, e+f+g, T1]
阶段2: totals=[T0,T1] → excl scan → offsets=[0, T0]
阶段3: warp0+0, warp1+T0 → 完整的全局 prefix sum
```

完整代码:
```cuda
template <const int NUM_THREADS = 256>
__global__ void block_inclusive_scan_f32_kernel(
    const float *input, float *output, int N) {
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  __shared__ float warp_totals[NUM_WARPS];
  __shared__ float warp_offsets[NUM_WARPS];

  int tid = threadIdx.x, idx = blockIdx.x * NUM_THREADS + tid;
  int warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;

  // ── 阶段 1 ──
  float val = (idx < N) ? input[idx] : 0.0f;
  #pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    float tmp = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset) val += tmp;
  }

  // ── 阶段 2 ──
  float warp_total = __shfl_sync(0xffffffff, val, WARP_SIZE - 1);
  if (lane == 0) warp_totals[warp] = warp_total;
  __syncthreads();
  if (warp == 0) {
    float off = (lane < NUM_WARPS) ? warp_totals[lane] : 0.0f;
    #pragma unroll
    for (int o = 1; o < NUM_WARPS; o <<= 1) {
      float tmp = __shfl_up_sync(0xffffffff, off, o);
      if (lane >= o) off += tmp;
    }
    float excl = __shfl_up_sync(0xffffffff, off, 1);
    if (lane < NUM_WARPS) warp_offsets[lane] = (lane == 0) ? 0.0f : excl;
  }
  __syncthreads();

  // ── 阶段 3 ──
  val += warp_offsets[warp];
  if (idx < N) output[idx] = val;
}
```

---

## 5. 处理任意长度 N

当 N > 256 时需多 block: 分段 scan → block totals 递归 scan → 广播 offset。这是经典 multi-block scan 问题。

---

## 6. Hillis-Steele vs Brent-Kung

| 算法 | 轮数 | 工作量 | work-efficient? |
|------|:---:|:---:|:---:|
| Hillis-Steele | log2(N) | N×log2(N) | 否 |
| Brent-Kung | 2×log2(N) | 2N | **是** |

Warp 内 (≤32): Hillis-Steele 足够。Block 内 (256+): Brent-Kung 更优。

---

## 7. 学习检查清单

- [ ] 理解 inclusive vs exclusive scan
- [ ] 画 Hillis-Steele 3 步的 lane 数据流
- [ ] 知道 `__shfl_up_sync` 里为什么要用 `tmp`
- [ ] 理解三阶段 block scan
- [ ] 能写 exclusive scan 的两种实现
- [ ] 理解 multi-block scan 分解策略

