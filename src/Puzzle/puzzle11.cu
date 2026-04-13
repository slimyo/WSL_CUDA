#define TPB 256
__global__ void axis_sum_kernel(float* out, const float* a, int size) {
    // 共享内存，大小为线程块大小(TPB)
    __shared__ float cache[TPB];

    int tid = threadIdx.x;           // 线程块内的列索引
    int col_idx = tid;               // 列索引（因为blockDim.x=1）
    int batch = blockIdx.y;          // 批次索引（行索引）

    // 计算全局内存中该元素的一维偏移
    int global_idx = batch * size + col_idx;

    // 步骤1: 加载数据到共享内存
    if (col_idx < size) {
        cache[tid] = a[global_idx];
    } else {
        cache[tid] = 0.0f; // 边界处理
    }
    __syncthreads();

    // 步骤2: 在共享内存上进行归约求和（需你填充的约12行核心逻辑）
    // ... 实现类似Puzzle 12的并行归约 ...
    for(int stride=1;stride<blockDim.x;stride*=2){
        int idx = 2*stride*tid;
        if(idx<blockDim.x){
            cache[idx]+=cache[idx+stride];
        }
        __syncthreads();
    }

    // 步骤3: 写回结果
    if (tid == 0) {
        out[batch] = cache[0];
    }
}