#include <iostream>
#include <cuda_runtime.h>

// GPU核函数
__global__ void map_kernel(int* a, int* out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = a[idx] + 10;
    }
}

int main() {
    const int SIZE = 4;
    
    // 主机数据
    int h_a[SIZE] = {1, 2, 3, 4};
    int h_out[SIZE] = {0};
    
    // 设备指针
    int *d_a, *d_out;
    
    // 1. 分配设备内存
    cudaMalloc((void**)&d_a, SIZE * sizeof(int));
    cudaMalloc((void**)&d_out, SIZE * sizeof(int));
    
    // 2. 拷贝数据到设备
    cudaMemcpy(d_a, h_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    
    // 3. 设置执行配置
    int threadsPerBlock = 256;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    
    // 4. 启动kernel
    map_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_out, SIZE);
    
    // 5. 同步，检查错误
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    
    // 6. 拷贝结果回主机
    cudaMemcpy(h_out, d_out, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 7. 打印结果
    std::cout << "Input: ";
    for (int i = 0; i < SIZE; i++) std::cout << h_a[i] << " ";
    std::cout << "\nOutput: ";
    for (int i = 0; i < SIZE; i++) std::cout << h_out[i] << " ";
    std::cout << std::endl;
    
    // 8. 清理设备内存
    cudaFree(d_a);
    cudaFree(d_out);
    
    return 0;
}