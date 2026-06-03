/**
   * reduce_warp.cu  —  Warp-Level Reduction with __shfl_down_sync
   * =============================================================================
   * TODO 阶段 1.1: 用树形 warp shuffle 替代 for 循环 + atomicAdd
   * 目标: 256 元素 block 内 reduce 只需 log2(32)=5 轮 shuffle + 1 次 atomic
   *
   * 参考实现 (LeetCUDA):
   *   third_party/LeetCUDA/kernels/reduce/block_all_reduce.cu
   *     - warp_reduce_sum_f32<kWarpSize>()           (L17-L22)
   *     - block_all_reduce_sum_f32_f32_kernel()      (L26-L47)
   *
   * 关键知识点:
   *   - __shfl_down_sync(mask, val, offset, width): lane+offset → lane
   *   - 树形 reduce: offset = 16,8,4,2,1 → 5 轮完成 32 线程
   *   - cross-warp: warp leader 写 shared memory, warp0 做最终 reduce
   * =============================================================================
   */

  #include <cuda_runtime.h>
  #include <cstdio>
  #include <cstdlib>
  #include <cmath>
  #include <cfloat>

  #define WARP_SIZE 32

  // ── 1. Warp-Level Tree Reduce (待实现) ──────────────────────────────────

  template <const int kWarpSize = WARP_SIZE>
  __device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    // TODO: 用 __shfl_down_sync 实现树形 reduce
    // #pragma unroll
    // for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    //     val += __shfl_down_sync(0xffffffff, val, mask);
    // }
    return val;
  }

  template <const int kWarpSize = WARP_SIZE>
  __device__ __forceinline__ float warp_reduce_max_f32(float val) {
    // TODO: warp-level max reduce (用 fmaxf 替代 +=)
    return val;
  }

  // ── 2. Block-Level Reduce (待实现) ──────────────────────────────────────

  template <const int NUM_THREADS = 256>
  __global__ void block_reduce_sum_f32_kernel(const float *input, float *output,
                                              int N) {
    // TODO: 两阶段 block reduce
    //   int tid = threadIdx.x;
    //   int idx = blockIdx.x * NUM_THREADS + tid;
    //   constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    //   __shared__ float smem[NUM_WARPS];
    //
    //   float val = (idx < N) ? input[idx] : 0.0f;
    //   int warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    //   val = warp_reduce_sum_f32<WARP_SIZE>(val);
    //
    //   if (lane == 0) smem[warp] = val;
    //   __syncthreads();
    //
    //   val = (tid < NUM_WARPS) ? smem[tid] : 0.0f;
    //   if (warp == 0) val = warp_reduce_sum_f32<NUM_WARPS>(val);
    //
    //   if (tid == 0) output[blockIdx.x] = val;
  }

  template <const int NUM_THREADS = 256>
  __global__ void block_reduce_max_f32_kernel(const float *input, float *output,
                                              int N) {
    // TODO: max 版本 (初始化值用 -FLT_MAX)
  }

  // ── 3. Host 端接口 (待实现) ─────────────────────────────────────────────

  float launch_reduce_sum(const float *d_input, int N) {
    // TODO: 递归 host wrapper
    //   1) blockSize=256, grid=(N+255)/256
    //   2) 分配 d_partial (grid 个 float)
    //   3) 启动 kernel
    //   4) grid>1 则递归; 否则 cudaMemcpy 回 host
    return 0.0f;
  }

  float launch_reduce_max(const float *d_input, int N) {
    // TODO: max 版本
    return 0.0f;
  }

  // ── 4. 验证 Main (待实现) ───────────────────────────────────────────────

  int main() {
    const int N = 1 << 20;
    printf("=== reduce_warp.cu ===\n");
    printf("状态: 待实现\n\n");
    printf("参考: third_party/LeetCUDA/kernels/reduce/block_all_reduce.cu\n");
    printf("      warp_reduce_sum_f32<WARP_SIZE>()      @L17\n");
    printf("      block_all_reduce_sum_f32_f32_kernel() @L26\n");
    printf("\n编译: nvcc -arch=sm_80 -o reduce_warp reduce_warp.cu\n");
    return 0;
  }