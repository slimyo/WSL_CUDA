#define TPB 3  // 线程块大小
__global__ void matmul_kernel(float* out, const float* a, const float* b, int size) {
    // 声明共享内存块
    __shared__ float a_shared[TPB][TPB];
    __shared__ float b_shared[TPB][TPB];
    
    // 计算全局线程索引（输出矩阵的(i,j)位置）
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 行索引
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // 列索引
    
    // 线程在线程块内的局部索引
    int local_i = threadIdx.x;
    int local_j = threadIdx.y;
    
    // 初始化累加器
    float sum = 0.0f;
    
    // 边界检查：只处理有效的矩阵元素
    if (i < size && j < size) {
        // 情况1：简单情况 - 矩阵能完全放入共享内存
        if (size <= TPB) {
            // 步骤1: 加载矩阵a和b到共享内存
            // 提示：每个线程负责加载矩阵中的一个元素
            if (local_i < size && local_j < size) {
                a_shared[local_i][local_j] = a[local_i * size + local_j];  // 需要你填写正确的索引
                b_shared[local_i][local_j] = b[local_i * size + local_j];  // 需要你填写正确的索引
            }
            
            // 等待所有线程完成加载
            __syncthreads();
            
            // 步骤2: 计算点积
            // 提示：使用共享内存中的数据计算a的行和b的列的点积
            for (int k = 0; k < size; k++) {
                // 需要你填写计算逻辑
                // sum += ???
                sum+=a_shared[local_i][k]*b_shared[k][local_j];
            }
            
            // 步骤3: 将结果写入全局内存
            out[i * size + j] = sum;
        }
        // 情况2：困难情况 - 矩阵大于共享内存块---每个线程确定处理输出位置（i，j）处的值
        else {
            // 步骤1: 迭代计算分块矩阵乘法
            // 提示：外层循环，每次处理一个TPB x TPB的块
            for (int tile = 0; tile < (size + TPB - 1) / TPB; tile++) {
                // 步骤2: 从全局内存加载当前块到共享内存
                // 提示：计算a和b当前块的全局索引
                int a_row = i;
                int a_col = tile * TPB + local_j;  // 需要你完善边界检查
                int b_row = tile * TPB + local_i;  // 需要你完善边界检查
                int b_col = j;
                
                // 边界检查
                if (a_col < size && a_row < size) {
                    a_shared[local_i][local_j] = a[a_row * size + a_col];
                } else {
                    a_shared[local_i][local_j] = 0.0f;
                }
                
                if (b_row < size && b_col < size) {
                    b_shared[local_i][local_j] = b[b_row * size + b_col];
                } else {
                    b_shared[local_i][local_j] = 0.0f;
                }
                
                // 等待共享内存加载完成
                __syncthreads();
                
                // 步骤3: 计算当前块的部分点积
                for (int k = 0; k < TPB; k++) {
                    // 需要你填写计算逻辑
                    sum += a_shared[local_i][k] * b_shared[k][local_j];
                }
                
                // 等待所有线程完成当前块的计算
                __syncthreads();
            }
            
            // 步骤4: 写入最终结果
            out[i * size + j] = sum;
        }
    }
}