# Warp-Level Reduction 学习笔记

> 对应文件: `src/Puzzle/reduce_wrap.cu`
> 参考实现: `third_party/LeetCUDA/kernels/reduce/block_all_reduce.cu`
> 前置知识: Puzzle 8 (dot-product), Puzzle 11 (axis-sum)

---

## 1. 为什么升级: shared memory reduce → warp shuffle

### 你现有的写法 (Puzzle 8/11 风格)

```cuda
// 全部用 shared memory + for 循环 + __syncthreads
__shared__ float shares[256];
shares[threadIdx.x] = val;
__syncthreads();
for (int stride = 256/2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
        shares[threadIdx.x] += shares[threadIdx.x + stride];
    __syncthreads();
}
if (threadIdx.x == 0) atomicAdd(out, shares[0]);
```

**问题**: 每一步都需要 `__syncthreads()` + 读写 shared memory。256 线程 8 轮同步，每轮都是 shared memory 访问，延迟 ~30 cycles/sync。

### warp shuffle 方式

```cuda
// 等价于最后 5 轮 (32→16→8→4→2→1)
// 但全部在寄存器中完成，0 次 shared memory，0 次 __syncthreads()
for (int mask = 16; mask >= 1; mask >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, mask);
}
```

**shuffle 延迟 ~5 cycles/轮**，比 shared memory 快 ~6 倍，且不需要 barrier。

---

## 2. 三种 Warp Shuffle 原语

一个 warp = 32 个线程，shuffle 在 warp 内交换寄存器数据。

### 2.1 `__shfl_down_sync(mask, val, delta)` — 树形归约

```
lane i 收到 lane (i + delta) 的 val
i + delta >= 32 的 lane 保持自己的 val 不变

树形 reduce (width=32):
  delta=16: lane0 += lane16, lane1 += lane17, ...
  delta=8:  lane0 += lane8,  lane1 += lane9,  ...
  delta=4:  lane0 += lane4
  delta=2:  lane0 += lane2
  delta=1:  lane0 += lane1
  结果: lane0 持有 32 个值的总和
```

### 2.2 `__shfl_xor_sync(mask, val, laneMask)` — 蝴蝶归约 (LeetCUDA 用的)

```
lane i 收到 lane (i ^ laneMask) 的 val

laneMask=16: lane0 swaps with lane16, lane1 swaps with lane17, ...
laneMask=8:  lane0 swaps with lane8, ...
...
laneMask=1:  lane0 swaps with lane1

结果: 所有 32 个 lane 都持有最终 sum (自动 broadcast!)
```

**LeetCUDA `block_all_reduce.cu:19-21`:**
```cuda
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;  // 所有 lane 都有完整 sum
}
```

**为什么用 xor 而不是 down?**
- `shfl_down`: 只有 lane0 有最终结果，需要额外 broadcast
- `shfl_xor`: 所有 lane 自动得到结果，cross-warp 阶段更简洁

### 2.3 `__shfl_up_sync(mask, val, delta)` — 用于 Prefix Sum

```
lane i 收到 lane (i - delta) 的 val
i - delta < 0 的 lane 保持自己的 val

用于 inclusive scan:
  for (offset = 1; offset < 32; offset <<= 1) {
      float tmp = __shfl_up_sync(0xffffffff, val, offset);
      if (lane >= offset) val += tmp;
  }

详见 scan_warp_learning.md
```

### 2.4 mask 参数

```
mask = 0xffffffff 表示全部 32 个 lane 参与
Volta+ (SM 70+) 必须用 _sync 版本并显式传 mask
```

---

## 3. 完整 Block Reduce 架构 (256 threads)

### 两阶段设计

```
Block = 8 warps × 32 lanes

阶段 1: WARP-Reduce (寄存器, 0 sync)
  每个 warp 独立做 tree/butterfly reduce
  → 8 个 warp sums (存在各自 lane0 或所有 lane)

阶段 2: CROSS-WARP Reduce (1 次 sync)
  warp leaders → shared memory[8]
  __syncthreads()
  warp0 的 8 个线程 → 再做一次 warp reduce
  thread0 写 output
```

### 完整代码

```cuda
#define WARP_SIZE 32

template <const int NUM_THREADS = 256>
__global__ void block_reduce_sum_f32_kernel(const float *input, float *output,
                                            int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;  // 8
  __shared__ float smem[NUM_WARPS];                   // 8 floats

  float val = (idx < N) ? input[idx] : 0.0f;

  // === 阶段 1: Warp Reduce ===
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  #pragma unroll
  for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  // val = warp_sum，所有 lane 都有

  // === 阶段 2: Cross-Warp ===
  if (lane == 0) smem[warp] = val;
  __syncthreads();

  val = (tid < NUM_WARPS) ? smem[tid] : 0.0f;
  if (warp == 0) {
    #pragma unroll
    for (int mask = NUM_WARPS >> 1; mask >= 1; mask >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, mask);
    }
  }

  if (tid == 0) output[blockIdx.x] = val;
}
```

### 复杂度对比

| 方法 | Shared Memory 访问 | `__syncthreads()` | 延迟 |
|------|:---:|:---:|:---:|
| Puzzle 8 (纯 SMEM for) | 8×256 reads | 8 | 高 |
| **Warp Shuffle** | 8 writes + 8 reads | 1 | **低** |

---

## 4. Host 端 — 递归处理任意长度 N

N > 256 时需要多轮 reduce:

```
N=1M → grid=4096 blocks → 4096 partial sums
       → grid=16 blocks    → 16 partial sums
       → grid=1 block      → 最终结果
```

```cuda
float launch_reduce_sum(const float *d_input, int N) {
    constexpr int BLOCK = 256;
    float *d_partial;
    int grid = (N + BLOCK - 1) / BLOCK;
    cudaMalloc(&d_partial, grid * sizeof(float));
    float *d_cur = (float*)d_input;

    while (grid > 1) {
        block_reduce_sum_f32_kernel<BLOCK>
            <<<grid, BLOCK>>>(d_cur, d_partial, N);
        N = grid;
        grid = (N + BLOCK - 1) / BLOCK;
        d_cur = d_partial;  // 输入变成上一轮的 partial sums
    }

    // 最后一轮
    block_reduce_sum_f32_kernel<BLOCK>
        <<<1, BLOCK>>>(d_cur, d_partial, N);

    float result;
    cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_partial);
    return result;
}
```

---

## 5. LeetCUDA 中 reduce 的两个变体

### 变体 A: 输出到全局标量 (`atomicAdd`)

`block_all_reduce.cu` 中 `block_all_reduce_sum_f32_f32_kernel` 用 `atomicAdd(y, sum)` 直接把结果累加到全局变量，不需要 host 端递归:

```cuda
if (tid == 0) atomicAdd(y, sum);
```

### 变体 B: `__device__` helper (LayerNorm/Softmax 用)

`softmax.cu` 和 `layer_norm.cu` 中，reduce 是 `__device__` 函数而非 `__global__` kernel:

```cuda
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  // ... 同上两阶段 reduce ...
  // 关键差异: 用 __shfl_sync broadcast 到 warp0 所有 thread
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;  // 调用方每个 thread 都拿到 block sum
}
```

**为什么需要 broadcast?** LayerNorm 中每个 thread 都需要 `mean` 去 normalize 自己的元素：
```
y[i] = (x[i] - mean) / std
```
所以 reduce 结果必须返回给所有 thread，不能只让 thread0 写 global memory。

---

## 6. Max Reduce 写法

```cuda
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}
```

初始化值: max 用 `-FLT_MAX`, min 用 `FLT_MAX`, sum 用 `0.0f`。

---

## 7. online softmax 中的 struct reduce

`softmax.cu` 展示了一个高级用法 — 对 `struct {float m; float d}` 做 warp reduce:

```cuda
struct __align__(8) MD { float m; float d; };

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  #pragma unroll
  for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(0xffffffff, value.m, stride);
    other.d = __shfl_xor_sync(0xffffffff, value.d, stride);
    // online softmax 归约公式:
    bool bigger = (value.m > other.m);
    MD big = bigger ? value : other;
    MD small = bigger ? other : value;
    value.d = big.d + small.d * __expf(small.m - big.m);
    value.m = big.m;
  }
  return value;
}
```

m = running max, d = sum(exp(x - max))。这就是 FlashAttention 的数学基础。

---

## 8. `shfl_down` vs `shfl_xor` 选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 生产代码 (reduce) | `shfl_xor` | 所有 lane 都有结果, 不需要额外 broadcast |
| 教学理解 | `shfl_down` | 数据流更直观 (高位→低位) |
| Prefix sum | `shfl_up` | scan 语义: 低位流向高位 |
| 只需要 lane0 的结果 | 两者皆可 | xor 也行 |

---

## 9. 学习检查清单

- [ ] 能画出 `__shfl_xor_sync` 的 butterfly 数据流图
- [ ] 理解两阶段 block reduce: warp → SMEM → warp0
- [ ] 知道 `__device__` helper 和 `__global__` kernel 版本的区别 (broadcast vs write)
- [ ] 能写出 max/min reduce 的 warp shuffle 版本
- [ ] 理解 host 端递归 reduce 的逻辑
- [ ] 阅读 `block_all_reduce.cu` FP16/BF16 版本，注意累积精度 (f16 input → f32 accumulate)

## 10. 推荐阅读顺序

1. `block_all_reduce.cu` L17-L22 — `warp_reduce_sum_f32` (5 行)
2. `block_all_reduce.cu` L26-L47 — 两阶段 block reduce kernel
3. `softmax.cu` 的 `block_reduce_sum_f32` — `__device__` helper 版本，注意 `__shfl_sync` broadcast
4. `layer_norm.cu` — 连续两次 reduce (mean + variance)

