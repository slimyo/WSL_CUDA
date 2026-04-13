__global__
void map2d_(const int*a,int*out,int width,int height){
    //// 矩阵尺寸：height行 × width列
    //float* matrix = (float*)malloc(height * width * sizeof(float));
    // 访问(i,j)元素：i行j列    
    //matrix[i * width + j] = value;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    if(row<height&&col<width){
        out[row*width+col] = a[row*width+col]+10;
    }
}