# LayerNorm & RMSNorm 学习笔记

> 参考: `third_party/LeetCUDA/kernels/layer-norm/layer_norm.cu`
> 前置: `reduce_warp_learning.md`, `softmax_learning.md`

## 1. 数学定义

- **LayerNorm**: 两次 reduce (mean sum + variance sum)
- **RMSNorm** (LLaMA/Mistral用): 一次 reduce (x² sum), 少 50% sync

```
LN:  mu = sum(x)/K,  sigma = sqrt(sum((x-mu)²)/K)
     y = gamma * (x-mu)/sigma + beta

RMS: rms = sqrt(sum(x²)/K)
     y = gamma * x / rms + beta
```

## 2. Fused Kernel 架构

输入 x[N][K], 每行一个 block (blockDim=K), 一个 kernel 内完成:
```
load → reduce mean → reduce var → normalize + affine → store
```
Fused: 1× global R/W vs Unfused: 3× global R/W

## 3. 核心实现 (f32)

```cuda
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y, float g, float b,
                                      int N, int K) {
  int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;
  __shared__ float s_mean, s_variance;
  float val = (idx < N*K) ? x[idx] : 0.0f;

  // Reduce mean
  float sum = block_reduce_sum_f32<NUM_THREADS>(val);
  if (tid == 0) s_mean = sum / (float)K;
  __syncthreads();

  // Reduce variance
  float diff = val - s_mean;
  float var = block_reduce_sum_f32<NUM_THREADS>(diff * diff);
  if (tid == 0) s_variance = rsqrtf(var / (float)K + 1e-5f);
  __syncthreads();

  // Normalize + affine (fused in registers)
  if (idx < N*K) y[idx] = (diff * s_variance) * g + b;
}
```

两次 `block_reduce_sum_f32` + 两次 `__syncthreads()`. RMSNorm 只需一次 reduce.

## 4. 关键设计点

### block_reduce 需要 broadcast
每个线程都需要 mean 和 inv_std 去 normalize 自己的元素:
```cuda
value = __shfl_sync(0xffffffff, value, 0, 32); // broadcast to all lanes
```

### FP16 累积用 FP32
warp reduce 内部先用 `__half2float` 转 FP32 再累加:
```cuda
float val_f32 = __half2float(val);
for (...) val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
```

### float4 向量化
```cuda
float4 reg = FLOAT4(x[idx]);
float local_sum = reg.x + reg.y + reg.z + reg.w;
// blockDim = K/4, shared memory 和 sync 减少 4×
```

## 5. 向量化版本一览

| 版本 | blockDim | 每线程处理 | 优化重点 |
|------|:---:|:---:|------|
| f32 | K | 1 float | 基准 |
| f32x4 | K/4 | 4 floats | 减少线程数 |
| f16_f32 | K | 1 half, FP32 accum | FP16 输入 |
| f16x8_pack | K/8 | 8 halfs, 128-bit ld/st | 最大化带宽 |

## 6. 学习检查清单

- [ ] LayerNorm 2次 reduce, RMSNorm 1次
- [ ] Fused kernel 为什么高效 (避免中间 global memory R/W)
- [ ] block_reduce broadcast 的必要性
- [ ] FP16→FP32 精度升级的原因
- [ ] 能用 nsys 对比 fused vs unfused

## 7. 推荐阅读顺序

1. `layer_norm.cu` `layer_norm_f32_kernel` — 最直观版本
2. `layer_norm.cu` `layer_norm_f32x4_kernel` — 向量化
3. `layer_norm.cu` `layer_norm_f16x8_pack_f16_kernel` — 128-bit load/store
4. `layer_norm.cu` `block_reduce_sum_f16_f32` — FP16→FP32 精度
