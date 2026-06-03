/**
   * bank_conflict.cu  —  Shared Memory Bank Conflict 实验
   * =============================================================================
   * TODO 阶段 1.3: 构造 bank conflict demo, 在 nsys/ncu 中观测,
   *     理解 padding 消除冲突的效果
   *
   * 参考:
   *   - LeetCUDA 中 layer-norm / softmax 的 shared memory 用法
   *   - third_party/LeetCUDA/kernels/layer-norm/layer_norm.cu
   *     warp reduce 中 shared memory 的声明和访问模式
   *   - third_party/LeetCUDA/kernels/swizzle/
   *     swizzle layout 用于避免 bank conflict
   *
   * 关键知识点:
   *   - SM shared memory 有 32 个 bank, 每个 bank 4 bytes 宽
   *   - 同一 warp 内多个线程访问同一 bank 的不同地址 → bank conflict
   *   - 常见冲突模式: stride = 32 (同 bank 同地址偏移)
   *   - 消除方法: padding (加 1 列) 使 stride 变为奇数
   * =============================================================================
   */

  #include <cuda_runtime.h>
  #include <cstdio>
  #include <cstdlib>

  #define WARP_SIZE 32
  #define NUM_BANKS 32

  // ── 1. Bank Conflict 演示 Kernel (待实现) ───────────────────────────────

  /**
   * 无 bank conflict 的 shared memory 访问 (stride=1, 连续读取)
   * 预期: 1-way bank conflict (即无冲突)
   */
  __global__ void smem_read_stride_1(float *output) {
    __shared__ float smem[32][32];  // 32x32 = 1024 floats
    int tid = threadIdx.x;
    // 初始化 shared memory
    for (int i = 0; i < 32; i++) {
      smem[tid][i] = tid * 32 + i;
    }
    __syncthreads();
    // stride=1 读取 (每列连续 → 不同 bank)
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
      sum += smem[tid][i];  // TODO: lane 对应行, 列连续 → 无冲突
    }
    output[tid] = sum;
  }

  /**
   * 严重 bank conflict (stride=32, 列优先读取)
   * 预期: 32-way bank conflict (最坏情况)
   */
  __global__ void smem_read_stride_32(float *output) {
    __shared__ float smem[32][32];
    int tid = threadIdx.x;
    for (int i = 0; i < 32; i++) {
      smem[tid][i] = tid * 32 + i;
    }
    __syncthreads();
    // stride=32 读取 (同行不同列 → 同 bank)
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
      sum += smem[i][tid];  // TODO: lane 对应列, 行变化 → 32-way 冲突
    }
    output[tid] = sum;
  }

  /**
   * 用 padding 消除 bank conflict
   * 多分配 1 列使 stride 变为奇数, 不同线程访问错开到不同 bank
   */
  __global__ void smem_read_with_padding(float *output) {
    __shared__ float smem[32][33];  // 32x33, padding=+1
    int tid = threadIdx.x;
    for (int i = 0; i < 32; i++) {
      smem[tid][i] = tid * 32 + i;
    }
    __syncthreads();
    // 列优先读取, 但 row_stride=33 (奇数) → 无冲突
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
      sum += smem[i][tid];  // TODO: stride=33 访问, 消除冲突
    }
    output[tid] = sum;
  }

  // ── 2. Host 端测量 (待实现) ─────────────────────────────────────────────

  void benchmark_kernel(void (*kernel)(float*), const char *name) {
    // TODO: 用 cudaEvent 计时, 对比三种 kernel 的耗时
    // 同时提示用 ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_*
    //  观测硬件 bank conflict 计数
    printf("  %s — 待测量\n", name);
  }

  // ── 3. Main (待实现) ────────────────────────────────────────────────────

  int main() {
    printf("=== bank_conflict.cu ===\n");
    printf("状态: 待实现\n\n");
    printf("运行后使用 ncu 观测:\n");
    printf("  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld_sum \\\n");
    printf("      --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st_sum \\\n");
    printf("      ./bank_conflict\n\n");
    printf("预期结果:\n");
    printf("  stride_1:     ~0 bank conflicts\n");
    printf("  stride_32:    大量 bank conflicts\n");
    printf("  with_padding: ~0 bank conflicts (padding 消除)\n");
    return 0;
  }