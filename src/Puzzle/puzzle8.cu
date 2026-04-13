// dot product
#include <__clang_cuda_runtime_wrapper.h>
# define SHARED_SIZE 256
__global__
void dot_product(const int*a,const int*b,int *out,int n){
    __shared__ float shares[SHARED_SIZE];
    int idx = blockDim.x*blockIdx.x +threadIdx.x;
    // 3. 边界检查：数组和共享内存都要检查
    if (threadIdx.x < SHARED_SIZE) {
        shares[threadIdx.x] = (idx < n) ? a[idx] * b[idx] : 0;
    } else {
        // 如果block大小大于共享内存大小，多余线程不参与
        return;
    }
    __syncthreads();
    if(threadIdx.x==0)
    {
        int temp =0;
        // 5. 归约范围是block实际大小和共享内存的较小值
        int valid_threads = min(SHARED_SIZE, blockDim.x);
        
        for (int i = 0; i < valid_threads; i++) {
            temp += shares[i];
        }
         // 6. 使用原子操作避免竞争
        atomicAdd(out, temp);
    }   
}

__global__
void dot_(const int*a,const int*b,int*out,int n){
    __shared__  int shares[SHARED_SIZE];
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(threadIdx.x<SHARED_SIZE){
        shares[threadIdx.x] = idx<n? a[idx]*b[idx]:0;
    }
    else{return;}
    __syncthreads();
    for(int thread=SHARED_SIZE/2;thread>0;thread>>=1){
        shares[threadIdx.x] += threadIdx.x<thread ? shares[threadIdx.x+thread] :0;
        __syncthreads();
    }
    // 3. 只有线程0执行原子操作
    if (threadIdx.x == 0) {
        atomicAdd(out, shares[0]);
    }
    
}