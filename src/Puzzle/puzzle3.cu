#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
__global__
void guard_(int*a,int*out,int n){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if(index<n){
        out[index] = a[index]+10;
    }
}