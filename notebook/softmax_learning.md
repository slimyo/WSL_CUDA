# Softmax: Naive → Safe → Online 学习笔记

> 参考: `third_party/LeetCUDA/kernels/softmax/softmax.cu`
> 前置: `reduce_warp_learning.md`, `bank_conflict_learning.md`

---

## 1. 三种 Softmax 算法

### Naive (有溢出风险)
```cuda
softmax(x_i) = exp(x_i) / sum(exp(x_j))
// exp(89) ≈ 4.5e38 → FP32 overflow (max 3.4e38)
```

### Safe (两遍扫描, 数值稳定)
```cuda
m = max(x)
softmax(x_i) = exp(x_i - m) / sum(exp(x_j - m))
// exp(≤0) ∈ (0,1], 不会溢出
```

### Online Safe (一遍扫描, FlashAttention 数学基础)
维护 running `(m, d)` = (max, denominator):
```cuda
for each x:
    if x > m:
        d = d * exp(m - x) + 1   // 重新缩放旧分母
        m = x
    else:
        d = d + exp(x - m)
```

**合并公式** (两个部分结果 (m1,d1) + (m2,d2), 设 m1 > m2):
```
m_new = m1
d_new = d1 + d2 * exp(m2 - m1)
```

---

## 2. LeetCUDA 核心实现

### `block_reduce_max_f32` — Device Helper

```cuda
template <const int NUM_THREADS = 256>
__device__ float block_reduce_max_f32(float val) {
  // warp reduce → shared memory → warp0 reduce
  value = warp_reduce_max_f32(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = warp_reduce_max_f32_NUM_WARPS(value);
  // ★ __shfl_sync broadcast 给所有 thread
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}
```

与你的 `reduce_wrap.cu` 中 block reduce kernel 的区别:
- 这是 `__device__` 函数 (被 kernel 调用，不是 kernel 本体)
- 最后 broadcast (每个线程都需要 max)
- 用 `static __shared__`

### Safe Softmax Kernel

```cuda
template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float *x, float *y, int N) {
  int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;

  float val = (idx < N) ? x[idx] : -FLT_MAX;
  float max_val = block_reduce_max_f32<NUM_THREADS>(val);   // reduce 1

  float exp_val = (idx < N) ? expf(val - max_val) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // reduce 2

  if (idx < N) y[idx] = exp_val / exp_sum;
}
```

两次 block reduce，两次 `__syncthreads()`。

### Online Softmax: `warp_reduce_md_op`

```cuda
struct __align__(8) MD { float m; float d; };  // 8-byte aligned

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  #pragma unroll
  for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(0xffffffff, value.m, stride);
    other.d = __shfl_xor_sync(0xffffffff, value.d, stride);

    bool bigger = (value.m > other.m);
    MD big = bigger ? value : other;
    MD small = bigger ? other : value;

    value.d = big.d + small.d * __expf(small.m - big.m);
    value.m = big.m;
  }
  return value;  // d 最终 = sum(exp(x - max))
}
```

每次 butterfly swap 后合并两个 `(m,d)` pair。最终一个 warp 内所有 lane 都持有 `(global_max, sum_of_exp)`。

---

## 3. 向量化优化一览

| 版本 | blockDim | 每线程数据 | 效果 |
|------|:---:|:---:|------|
| f32 | K | 1 float | 基准 |
| f32x4 | K/4 | 4 floats (float4) | 线程数 ÷4 |
| f16_f32 | K | 1 half → f32 累积 | FP16 输入 |
| f16x8_pack | K/8 | 8 halfs, 128-bit load | 最大化内存带宽 |

---

## 4. 学习检查清单

- [ ] naive → safe → online 各阶段的问题与解法
- [ ] 能推导 online 合并公式
- [ ] 理解 `__device__ block_reduce` 为什么要 broadcast
- [ ] 能追踪 `warp_reduce_md_op` 一轮 butterfly 的 m/d 变化
- [ ] 知道 online softmax 如何推广到 FlashAttention

## 5. 推荐阅读顺序

1. `safe_softmax_f32_per_token_kernel` — 最直观的两遍实现
2. `warp_reduce_md_op` — online 核心 (仅 13 行)
3. `online_safe_softmax_f32_per_token_kernel` — 完整 online 流程
4. FlashAttention 论文 §2 — 从 softmax 到 attention
