/**
   * scan_warp.cu  —  Warp-Level Prefix Sum with __shfl_up_sync
   * =============================================================================
   * TODO 阶段 1.2: 用 warp-level inclusive/exclusive scan 改写 puzzle 10
   *
   * 参考实现:
   *   - LeetCUDA 的 softmax/layer-norm 大量使用 warp reduce 模式,
   *     warp scan 的 shuffle 原语与 reduce 一致, 只是方向和数据流不同
   *   - third_party/LeetCUDA/kernels/reduce/block_all_reduce.cu
   *     warp_reduce_sum_f32 模板 (L17-L22) — shuffle 用法参考
   *   - third_party/LeetCUDA/kernels/softmax/softmax.cu
   *     online softmax 的 partial reduce 模式
   *
   * 关键知识点:
   *   - __shfl_up_sync(mask, val, offset, width): lane-offset → lane
   *   - Inclusive scan: lane_i = sum(lane_0 .. lane_i)
   *   - Exclusive scan: lane_i = sum(lane_0 .. lane_{i-1}), lane0 = 0
   *   - Block scan (Kogge-Stone / Brent-Kung):
   *     阶段1: warp 内 scan → warp sum 写 shared memory
   *     阶段2: warp0 对 warp sums 做 scan
   *     阶段3: 各 warp 加上对应 offset
   * =============================================================================
   */

  #include <cuda_runtime.h>
  #include <cstdio>
  #include <cstdlib>

  #define WARP_SIZE 32

  // ── 1. Warp-Level Scan (待实现) ─────────────────────────────────────────

  template <const int kWarpSize = WARP_SIZE>
  __device__ __forceinline__ float warp_inclusive_scan_f32(float val) {
    // TODO: 用 __shfl_up_sync 实现 warp inclusive scan
    // 提示:
    //   #pragma unroll
    //   for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    //       float tmp = __shfl_up_sync(0xffffffff, val, offset);
    //       if (lane >= offset) val += tmp;
    //   }
    //   注意: 需要保存当前 lane 的 val 再 shuffle, 防止覆盖
    return val;
  }

  template <const int kWarpSize = WARP_SIZE>
  __device__ __forceinline__ float warp_exclusive_scan_f32(float val) {
    // TODO: 用 inclusive scan 实现 exclusive scan
    // 即 result = inclusive_scan 左移一位, lane0 = 0
    return val;
  }

  // ── 2. Block-Level Scan (待实现) ────────────────────────────────────────

  template <const int NUM_THREADS = 256>
  __global__ void block_inclusive_scan_f32_kernel(const float *input,
                                                  float *output, int N) {
    // TODO: 三阶段 block scan
    //   阶段1: 每个 warp 内做 inclusive scan
    //   阶段2: warp leader 把 warp sum 写入 shared memory,
    //          warp0 对 warp sums 做 scan
    //   阶段3: 各 warp 把对应的 warp offset 加到自己的结果上
  }

  // ── 3. Host 端接口 (待实现) ─────────────────────────────────────────────

  void launch_inclusive_scan(const float *d_input, float *d_output, int N) {
    // TODO: 对任意长度 N 调用 block scan kernel
    // 大数组需要分段处理并累加 block offset
  }

  // ── 4. 验证 Main (待实现) ───────────────────────────────────────────────

  int main() {
    const int N = 1 << 10;
    printf("=== scan_warp.cu ===\n");
    printf("状态: 待实现\n\n");
    printf("参考:\n");
    printf("  third_party/LeetCUDA/kernels/reduce/block_all_reduce.cu\n");
    printf("    warp_reduce_sum_f32 — shuffle 原语用法参考\n");
    printf("  third_party/LeetCUDA/kernels/softmax/softmax.cu\n");
    printf("    online softmax 的 partial reduce 流程\n");
    printf("\n编译: nvcc -arch=sm_80 -o scan_warp scan_warp.cu\n");
    return 0;
  }