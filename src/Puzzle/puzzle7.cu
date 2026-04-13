#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
# define windowsize_3 256
__global__
void pool (const int*a,int * out,int n){
    __shared__ float shares[windowsize_3];
    int index = blockDim.x*blockIdx.x +threadIdx.x;
    if(index<n){
        shares[threadIdx.x] = a[threadIdx.x];
    }
    __syncthreads();

    // 2. 计算（边界在内部处理）
    if (index < n) {
        int sum = 0;
        
        // 窗口[i-2, i-1, i]
        int start = max(index - 2, 0);  // 边界处理
        int end = index;
        
        // 在共享内存中计算
        for (int j = start; j <= end; j++) {
            // 注意：j可能是全局索引，需要转换为共享内存索引
            // 这里需要额外处理...
            sum+=a[j];
        }
        
        out[index] = sum;
    }

}