#include <__clang_cuda_runtime_wrapper.h>
# define SHARED_WINDOW 256
# define MAX_WINDOW 256
__global__
void conv1d(const int*a,const int*b,int*out,int n,int window){
    __shared__ int windows[MAX_WINDOW];
    __shared__ int shares[SHARED_WINDOW];
    // 如何解决：如果数据存共享内存，如何解决当前block共享内存没有所需idx-1的数据？
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;
    //tid超过MAX_WINDOW
    if(tid<MAX_WINDOW){
        windows[tid] = (tid<window)? b[tid] : 0;
        shares[tid] = (idx<n)? a[idx] : 0;
    }
    __syncthreads();
    int temp =0;
    for(int w=0;w<window;w++)
    {
        //不考虑中间断开情况
        temp+=shares[tid+w]*windows[tid+w];
    }
    atomicAdd(out+idx,temp);
}


#define TILE_SIZE 256
__global__
void conv1d_basic_(const int* a, const int* b, int* out, int n, int window) {
    // 共享内存：存储输入数据和卷积核
    extern __shared__ int smem[];
    int* a_shared = smem;                     // 大小: TILE_SIZE + window - 1
    int* b_shared = &smem[TILE_SIZE + window - 1];  // 大小: window
    
    int tid = threadIdx.x;
    int block_start = blockIdx.x * TILE_SIZE;

    //加载b-开始window个线程
    if(tid<window){
        b_shared[tid] = b[tid];
    }
    //这里不需要同步

    //加载a和halo数据
    int halo_end = TILE_SIZE + window -1;
    int idx = block_start + tid;
    if(idx<n){
        a_shared[tid] = a[idx]; 
    }
    else{
        a_shared[tid] = 0;
    }

    if(block_start+tid+TILE_SIZE<n&&tid<window-1){
        a_shared[TILE_SIZE+tid] = a[block_start+tid+TILE_SIZE];
    }
    __syncthreads();

    //卷积计算：只有TILE_SIZE个线程
    if(tid<TILE_SIZE){
        int temp = 0;
        for(int j=0;j<window;j++)
        {
            temp += a_shared[tid+j]*b_shared[j];
        }
        out[idx] = temp;
    }



    
}


//-------------------------------------------------------------------deepseek-ref----------------
__global__
void conv1d_basic(const int* a, const int* b, int* out, int n, int window) {
    // 共享内存：存储输入数据和卷积核
    extern __shared__ int smem[];
    int* a_shared = smem;                     // 大小: TILE_SIZE + window - 1
    int* b_shared = &smem[TILE_SIZE + window - 1];  // 大小: window
    
    int tid = threadIdx.x;
    int block_start = blockIdx.x * TILE_SIZE;
    
    // 1. 加载卷积核到共享内存（所有线程协作）
    if (tid < window) {
        b_shared[tid] = b[tid];
    }
    
    // 2. 加载输入数据到共享内存（包含halo区域）
    int load_idx = block_start + tid;
    
    // 主区域
    if (load_idx < n) {
        a_shared[tid] = a[load_idx];
    } else {
        a_shared[tid] = 0;
    }
    
    // 右侧halo区域（最后一个线程加载额外数据）
    int halo_idx = block_start + TILE_SIZE + tid;
    if (tid < window - 1 && halo_idx < n) {
        a_shared[TILE_SIZE + tid] = a[halo_idx];
    }
    __syncthreads();
    
    // 3. 计算卷积（只计算有效输出）
    int output_idx = block_start + tid;
    if (output_idx < n) {
        int temp = 0;
        for (int w = 0; w < window; w++) {
            temp += a_shared[tid + w] * b_shared[w];
        }
        out[output_idx] = temp;
    }
}