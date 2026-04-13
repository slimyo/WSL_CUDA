#include <__clang_cuda_runtime_wrapper.h>
__global__
void shared_map(const int * a,int *out,int n){
    __shared__ int shares[4];
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if(index<n){
        shares[threadIdx.x] = a[index];
        // 必须同步：确保所有写入完成
        __syncthreads();    
        out[index] = shares[threadIdx.x]+10;
    }
  
}