/*
输入
a: 形状为 (SIZE, 1)的列向量
当SIZE=2时：a = [[0], [1]]
b: 形状为 (1, SIZE)的行向量
当SIZE=2时：b = [[0, 1]]
目标：计算out = a + b，通过广播机制
输出
out: 形状为 (SIZE, SIZE)的矩阵
当SIZE=2时：out = [[0, 1], [1, 2]]
*/
#include <__clang_cuda_runtime_wrapper.h>
__global__
void broadcast_(const int*a,const int*b,int *out,int width,int height){
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if(row<height&&col<width){
        out[row*width+col] = a[row] + b[col]; 
    }
}