#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <iostream>
#include <cuda_runtime.h>

__global__
void zip_map(int*a,int*b,int*out,int n){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index<n){
        out[index] = a[index] + b[index] ;  
    }

}

void main(){
    int a[4] = {1,2,3,4};
    int b[4] = {1,2,3,4};
    int out[4];

    int * d_a;
    int * d_b;
    int * d_out;
    cudaMalloc((void**)&d_a, 4 * sizeof(int));
    cudaMalloc((void**)&d_b, 4 * sizeof(int));  
    cudaMalloc((void**)&d_out, 4 * sizeof(int));

    cudaMemcpy(d_a, a,sizeof(int)* 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b,sizeof(int)* 4, cudaMemcpyHostToDevice);

    int block_size =256;
    int block_in_grim =     (4 + block_size - 1) / block_size;
    zip_map<<<block_in_grim,block_size>>>(d_a, d_b, d_out, 4);

// 5. 同步，检查错误
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    
    cudaMemcpy(out, d_out, 4*sizeof(int), cudaMemcpyDeviceToHost);
    // 7. 打印结果
    std::cout << "Input: ";
    for (int i = 0; i < 4; i++) std::cout << a[i] << " ";
    std::cout << "\nOutput: ";
    for (int i = 0; i < 4; i++) std::cout << out[i] << " ";
    std::cout << std::endl;
    
    // 8. 清理设备内存
    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_b);
}