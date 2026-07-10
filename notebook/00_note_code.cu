#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define WARP_SIZE 32
// 强制类型转换宏，用于向量化加载/存储
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// ============================================================================
// Day 1-2: GPU 基础架构 + 内存层级 + 编程模型
// Focus: Element-wise operations, Vectorization, Memory Coalescing
// ============================================================================

// ElementWise Add: 基础的逐元素加法
// Day 2: 内存层级
// grid(N/128), block(128)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}

// ElementWise Add + Vec4: 向量化优化
// 使用 float4 一次处理 128 bit 数据，减少指令数，提升带宽利用率
// grid(N/128), block(128/4)
__global__ void elementwise_add_vec4(float *a, float *b, float *c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  }
}

// Sigmoid: 激活函数 y=1/(1+exp(-x))
// Day 2: 计算密集度分析
__global__ void sigmoid(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// Sigmoid + Vec4: 向量化版本
__global__ void sigmoid_vec4(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
    reg_y.y = 1.0f / (1.0f + expf(-reg_y.y)); // Typo in original reg_y fixed to reg_x.y logic if intended
    reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
    reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
    reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
    FLOAT4(y[idx]) = reg_y;
  }
}

// Relu: 激活函数 y=max(0,x)
__global__ void relu(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = fmaxf(0.0f, x[idx]);
}

// Relu + Vec4
__global__ void relu_vec4(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

// ============================================================================
// Day 3: Warp Shuffle + Reduce + Scan
// Focus: Warp Primitives, Butterfly Reduce, Shared Memory Bank Conflict
// ============================================================================

// Warp Reduce Sum: 蝴蝶交换算法
// __shfl_xor_sync: 所有参与线程最终都持有结果
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Block Reduce Sum: 两阶段归约
// 1. Warp内通过Shuffle归约
// 2. Warp间通过Shared Memory归约
// grid(N/128), block(128)
template <const int NUM_THREADS = 128>
__device__ __forceinline__ float block_reduce_sum(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  // Stage 1: Warp Level Reduce
  val = warp_reduce_sum<WARP_SIZE>(val);
  
  // Stage 2: Cross-Warp Reduce
  if (lane == 0)
    shared[warp] = val;
  __syncthreads();
  
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum<NUM_WARPS>(val);
  return val;
}

// Block Reduce Max
template <const int NUM_THREADS = 128>
__device__ __forceinline__ float block_reduce_max(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  val = warp_reduce_max<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  val = warp_reduce_max<NUM_WARPS>(val);
  return val;
}

// Histogram: 统计直方图
// Day 3: 原子操作
__global__ void histogram(int *a, int *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    atomicAdd(&(y[a[idx]]), 1);
}

// Histogram + Vec4: 减少原子操作调用次数
__global__ void histogram_vec4(int *a, int *y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    int4 reg_a = INT4(a[idx]);
    atomicAdd(&(y[reg_a.x]), 1);
    atomicAdd(&(y[reg_a.y]), 1);
    atomicAdd(&(y[reg_a.z]), 1);
    atomicAdd(&(y[reg_a.w]), 1);
  }
}

// Block All Reduce Sum: 全局归约
// grid(N/128), block(128)
template <const int NUM_THREADS = 128>
__global__ void block_all_reduce_sum(float *a, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float sum = (idx < N) ? a[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  
  sum = warp_reduce_sum<WARP_SIZE>(sum);
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();
  
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum<NUM_WARPS>(sum);
    
  if (tid == 0)
    atomicAdd(y, sum);
}

// Dot Product: 向量点积
// grid(N/128), block(128)
template <const int NUM_THREADS = 128>
__global__ void dot(float *a, float *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  
  prod = warp_reduce_sum<WARP_SIZE>(prod);
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads();
  
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum<NUM_WARPS>(prod);
    
  if (tid == 0)
    atomicAdd(y, prod);
}

// Dot Product + Vec4
template <const int NUM_THREADS = 128 / 4>
__global__ void dot_vec4(float *a, float *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_a = FLOAT4(a[idx]);
  float4 reg_b = FLOAT4(b[idx]);
  float prod = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y +
                            reg_a.z * reg_b.z + reg_a.w * reg_b.w)
                         : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  
  prod = warp_reduce_sum<WARP_SIZE>(prod);
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads();
  
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum<NUM_WARPS>(prod);
    
  if (tid == 0)
    atomicAdd(y, prod);
}

// ============================================================================
// Day 5: Transformer + Layer/RMS Norm + Softmax
// Focus: Online Softmax, Numerical Stability, Reduction patterns in Norms
// ============================================================================

// RMS Norm: Root Mean Square Normalization
// grid(N*K/K), block(K=128)
// y = x * rsqrtf(mean(x^2) + eps) * scale
template <const int NUM_THREADS = 128>
__global__ void rms_norm(float *x, float *y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_variance;
  float value = (idx < N * K) ? x[idx] : 0.0f;
  float variance = value * value;
  
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();
  
  if (idx < N * K)
    y[idx] = (value * s_variance) * g;
}

// RMS Norm + Vec4: 减少访存
template <const int NUM_THREADS = 128 / 4>
__global__ void rms_norm_vec4(float *x, float *y, float g, int N, int K) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_variance;
  float4 reg_x = FLOAT4(x[idx]);
  float variance = (idx < N * K) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y +
                                    reg_x.z * reg_x.z + reg_x.w * reg_x.w)
                                 : 0.0f;
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();
  
  float4 reg_y;
  reg_y.x = reg_x.x * s_variance * g;
  reg_y.y = reg_x.y * s_variance * g;
  reg_y.z = reg_x.z * s_variance * g;
  reg_y.w = reg_x.w * s_variance * g;
  if (idx < N * K)
    FLOAT4(y[idx]) = reg_y;
}

// Layer Norm: Mean + Variance + Normalize
// grid(N*K/K), block(K=128)
template <const int NUM_THREADS = 128>
__global__ void layer_norm(float *x, float *y, float g, float b, int N, int K) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float value = (idx < N * K) ? x[idx] : 0.0f;
  
  // Compute Mean
  float sum = block_reduce_sum<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / (float)K;
  __syncthreads();
  
  // Compute Variance
  float diff = value - s_mean;
  float variance = diff * diff;
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();
  
  if (idx < N * K)
    y[idx] = (diff * s_variance) * g + b;
}

// Layer Norm + Vec4
template <const int NUM_THREADS = 128 / 4>
__global__ void layer_norm_vec4(float *x, float *y, float g, float b, int N,
                                int K) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;
  __shared__ float s_variance;
  
  // Fix: Added missing semicolon
  float4 reg_x = FLOAT4(x[idx]);
  
  float value = (idx < N * K) ? (reg_x.x + reg_x.y + reg_x.z + reg_x.w) : 0.0f;
  
  // Compute Mean
  float sum = block_reduce_sum<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / (float)K;
  __syncthreads();
  
  // Compute Variance
  float4 reg_x_hat;
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  reg_x_hat.z = reg_x.z - s_mean;
  reg_x_hat.w = reg_x.w - s_mean;
  
  float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y +
                   reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();
  
  float4 reg_y;
  reg_y.x = reg_x_hat.x * s_variance * g + b;
  reg_y.y = reg_x_hat.y * s_variance * g + b;
  reg_y.z = reg_x_hat.z * s_variance * g + b;
  reg_y.w = reg_x_hat.w * s_variance * g + b;
  if (idx < N * K)
    FLOAT4(y[idx]) = reg_y;
}

// Softmax v2: 原子归约 + Grid Fence
// 注意：这是一个简化的 Grid-level Softmax 实现，依赖 atomicAdd 累加全局 sum
// 在高并发下，不同 block 读取 total 的时机不同可能导致结果微小波动
// grid(N/128), block(128)
template <const int NUM_THREADS = 128>
__global__ void softmax_v2(float *x, float *y, float *total, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float sum = block_reduce_sum<NUM_THREADS>(exp_val);
  
  // Accumulate global sum
  if (tid == 0)
    atomicAdd(total, sum);
  
  // Ensure total is written before being read by other threads (weak fence)
  __threadfence(); 
  
  if (idx < N)
    y[idx] = exp_val / (*total);
}

// Softmax v2 + Vec4
template <const int NUM_THREADS = 128 / 4>
__global__ void softmax_v2_vec4(float *x, float *y, float *total, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4;

  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_exp;
  reg_exp.x = (idx < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx < N) ? expf(reg_x.w) : 0.0f;
  
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float sum = block_reduce_sum<NUM_THREADS>(exp_val);
  
  if (tid == 0)
    atomicAdd(total, sum);
  __threadfence();
  
  if (idx < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (*total);
    reg_y.y = reg_exp.y / (*total);
    reg_y.z = reg_exp.z / (*total);
    reg_y.w = reg_exp.w / (*total);
    FLOAT4(y[idx]) = reg_y;
  }
}

// ============================================================================
// Day 6: Tensor Cores + GEMM 优化
// Focus: Tiling, Thread-tiling, Vectorization, Loop Unrolling
// ============================================================================

// SGEMM: 基础 Block Tile + Shared Memory
// Block Tile (BM=32, BN=32), K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
__global__ void sgemm(float *a, float *b, float *c, int M, int N, int K) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  __shared__ float s_a[BM][BK], s_b[BK][BN];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;

  // 计算加载索引
  int load_smem_a_m = tid / 32; 
  int load_smem_a_k = tid % 32;
  int load_smem_b_k = tid / 32;
  int load_smem_b_n = tid % 32;
  
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  float sum = 0.f;
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 加载 A 到 SMEM
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_m * K + load_gmem_a_k];
    
    // 加载 B 到 SMEM
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_k * N + load_gmem_b_n];
    
    __syncthreads();
    
    // 计算
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      sum += s_a[ty][k] * s_b[k][tx];
    }
    __syncthreads();
  }
  
  int store_gmem_c_m = by * BM + ty;
  int store_gmem_c_n = bx * BN + tx;
  c[store_gmem_c_m * N + store_gmem_c_n] = sum;
}

// SGEMM: 高级优化 - Thread Tile + K Tile + Vec4
// Day 6: 增加 TM=8, TN=8 计算密度，每个 Thread 计算 64 个元素
// BM=BN=128, BK=8 (为向量化优化)
__global__ void sgemm_thread_tile_vec4(float *a, float *b, float *c, int M,
                                       int N, int K) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;
  __shared__ float s_a[BM][BK], s_b[BK][BN]; // 8KB Shared Memory

  // 1. Shared Memory 加载索引映射
  // s_a: 128行 x 8列. 每行8个元素, 1个线程读4个, 需2线程/行
  int load_smem_a_m = tid / 2; 
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4; 
  
  // s_b: 8行 x 128列. 每行128个元素, 1个线程读4个, 需32线程/行
  int load_smem_b_k = tid / 32;       
  int load_smem_b_n = (tid % 32) * 4; 

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  float r_c[TM][TN] = {0.0}; // 寄存器累加器

  // 2. K 分块循环
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // Vec4 加载到 SMEM
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_m * K + load_gmem_a_k]);
    
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_k * N + load_gmem_b_n]);
    
    __syncthreads();
    
    // 3. 计算：每个线程计算 TMxTN (8x8) 个元素
#pragma unroll
    for (int k = 0; k < BK; k++) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          int comp_smem_a_m = ty * TM + m; // 映射到 s_a 的行 (0-127)
          int comp_smem_b_n = tx * TN + n; // 映射到 s_b 的列 (0-127)
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
      }
    }
    __syncthreads();
  }

  // 4. 写回结果
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 4) { // 使用 float4 写回
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

// ============================================================================
// Day 8: 推理负载 + Attention 变体
// Focus: SGEMV (Decode阶段瓶颈), Batch=1 优化
// ============================================================================

// SGEMV: Warp SGEMV K32
// Decode阶段: M=Batch*Seq=1, K=Hidden. 这里的K维度较小，适合Warp直接覆盖
__global__ void sgemv_k32(float *a, float *x, float *y, int M, int K) {
  int tx = threadIdx.x;         
  int ty = threadIdx.y;         
  int bx = blockIdx.x;          
  int lane = tx % WARP_SIZE;    
  int m = bx * blockDim.y + ty; 

  if (m < M) {
    float sum = 0.0f;
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = w * WARP_SIZE + lane;
      sum += a[m * K + k] * x[k];
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

// SGEMV: Warp SGEMV K128 + Vec4
__global__ void sgemv_k128(float *a, float *x, float *y, int M, int K) {
  int tx = threadIdx.x;         
  int ty = threadIdx.y;         
  int bx = blockIdx.x;          
  int lane = tx % WARP_SIZE;    
  int m = blockDim.y * bx + ty; 

  if (m < M) {
    float sum = 0.0f;
    // 每个线程负责4个元素，一个warp (32 threads) 覆盖 128 个元素
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = (w * WARP_SIZE + lane) * 4;
      float4 reg_x = FLOAT4(x[k]);
      float4 reg_a = FLOAT4(a[m * K + k]);
      sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z +
              reg_a.w * reg_x.w);
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

// SGEMV: Warp SGEMV K16
// 处理 Hidden Size 很小 (K=16) 的情况
template <const int ROW_PER_WARP = 2>
__global__ void sgemv_k16(float *A, float *x, float *y, int M, int K) {
  constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
  int tx = threadIdx.x;      
  int ty = threadIdx.y;      
  int bx = blockIdx.x;       
  int lane = tx % WARP_SIZE; 
  int k = lane % K_WARP_SIZE; 
  
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
  if (m < M) {
    float sum = A[m * K + k] * x[k];
    sum = warp_reduce_sum<K_WARP_SIZE>(sum);
    // K_WARP_SIZE 个线程参与计算，只有 k=0 的线程持有最终结果
    if (k == 0)
      y[m] = sum;
  }
}
