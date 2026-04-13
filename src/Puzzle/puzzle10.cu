#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
__global__
void prefix_sum(float*a,float*out,int n){
    extern __shared__ float smem[];
    float* a_shared = smem;
    int tid = threadIdx.x;
    int idx = blockDim.x*blockIdx.x+tid;

    if(idx<n){
        a_shared[tid] = a[idx]; 
    }else{
        a_shared[tid] = 0.0f;
    }
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            a_shared[tid]+=a_shared[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        out[blockIdx.x] = a_shared[0];
    }
}


//----------------------------Up-Sweep 版本
__global__ void block_sum_kernel(float* out, const float* a, int size) {
    // 使用静态共享内存，假设TPB（线程块大小）已知，例如 256
    __shared__ float cache[256]; 

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    // 1. 加载数据，并处理数组边界
    cache[local_idx] = (global_idx < size) ? a[global_idx] : 0.0f;
    __syncthreads();

    // 2. 在共享内存上进行归约 (Up-Sweep 版本，与图一致)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // 活跃线程：满足 (local_idx % (2*stride) == 0)
        int index = 2 * stride * local_idx;
        if (index < blockDim.x) {
            cache[index] += cache[index + stride];
        }
        __syncthreads();
    }

    // 3. 将每个块的结果写入输出数组的对应位置
    if (local_idx == 0) {
        out[blockIdx.x] = cache[0];
    }
}