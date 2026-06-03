# 02 CUDA 编程模型: Thread / Block / Grid 与 Kernel Launch

> 对象: CUDA / GPU 零基础
> 前置: 01_gpu_hardware_architecture.md
> 目标: 面试能写 CUDA kernel，理解层级结构的每一个参数

---

## 1. 一个极简 CUDA 程序

```cuda
#include <cstdio>

// GPU 上执行的函数 = kernel
__global__ void add_one(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) data[idx] += 1;
}

int main() {
    int N = 1024;
    int *d_data;
    cudaMalloc(&d_data, N * sizeof(int));           // 在 GPU 上分配内存

    add_one<<<32, 32>>>(d_data, N);                 // 启动 kernel

    cudaDeviceSynchronize();                         // 等 GPU 完成
    cudaFree(d_data);                                // 释放
    return 0;
}
```

每一个细节都值得展开, 下面逐层拆解。

---

## 2. Kernel 函数: `__global__`

CUDA 有三种函数前缀:

| 前缀 | 调用方 | 执行位置 | 常见用途 |
|------|------|------|------|
| `__global__` | Host (CPU) | Device (GPU) | Kernel 入口 |
| `__device__` | Device (GPU) | Device (GPU) | 被 kernel 调用的辅助函数 |
| `__host__` | Host (CPU) | Host (CPU) | 普通 CPU 函数 (默认) |

**关键点**:
- `__global__` 返回类型必须是 `void`
- `__global__` 不能是类的成员函数
- `__global__` 是异步的: host 发起后立即返回, 不等待 GPU 完成
- `__device__` 函数在编译时必须对编译器可见 (通常在 `.h` 或 `.cuh` 中, 不能放在 `.cpp`)

```cuda
// device helper — 只能被其他 device/global 函数调用
__device__ float helper_func(float x) {
    return x * x + 1.0f;
}

__global__ void my_kernel(float *data) {
    data[threadIdx.x] = helper_func(data[threadIdx.x]);  // OK
}

// host 不能直接调用 helper_func() — 编译错误
```

---

## 3. Kernel Launch 语法: `<<<grid, block>>>`

```cuda
kernel_name<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|:---:|
| `gridDim` | `dim3` 或 `int` | Grid 中有多少个 block | 必填 |
| `blockDim` | `dim3` 或 `int` | 每个 block 有多少个 thread | 必填 |
| `sharedMemBytes` | `int` | 动态 shared memory 大小 (字节) | 0 |
| `stream` | `cudaStream_t` | 关联的 CUDA stream | 0 (默认流) |

### 3.1 grid / block 可以是一维、二维、三维

```cuda
// 一维: 256 个 block, 每个 block 128 个 thread
kernel<<<256, 128>>>();

// 二维: 16×16 blocks, 每个 block 32×4 threads
dim3 grid(16, 16);
dim3 block(32, 4);
kernel<<<grid, block>>>();
```

**为什么要多维?** 处理图像/矩阵时, `(x, y)` 坐标映射更自然:

```cuda
__global__ void process_image(float *image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        image[y * width + x] = ...;
    }
}
```

### 3.2 内置变量: 在哪里知道"我是谁"

**面试必须背熟的 6 个变量:**

| 变量 | 类型 | 含义 |
|------|------|------|
| `threadIdx.x/y/z` | `uint3` | 当前 thread 在 block 内的索引 (从 0 开始) |
| `blockIdx.x/y/z` | `uint3` | 当前 block 在 grid 内的索引 (从 0 开始) |
| `blockDim.x/y/z` | `dim3` | 当前 block 的维度 (即 `<<<grid,block>>>` 的 block 参数) |
| `gridDim.x/y/z` | `dim3` | 当前 grid 的维度 (即 `<<<grid,block>>>` 的 grid 参数) |

**全局 thread id 公式:**

```cuda
// 一维
int global_id = threadIdx.x + blockIdx.x * blockDim.x;

// 二维
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int global_id_2d = y * gridDim.x * blockDim.x + x;
```

---

## 4. Grid / Block / Thread 的物理含义 (对照硬件)

回顾 01 文档的内容, 这里做精确对应:

```
Grid
 ├── Block 0  ──→  SM 0
 ├── Block 1  ──→  SM 1
 ├── Block 2  ──→  SM 0  (SM 上可同时驻留多个 block)
 ├── Block 3  ──→  SM 2
 ├── ...
 └── Block N-1 →  SM M

每个 Block 内部:
 Block
  ├── Warp 0 (thread 0-31)
  ├── Warp 1 (thread 32-63)
  ├── ...
  └── Warp K (thread blockDim-32 ... blockDim-1)

每个 Warp:
  按 32 个 thread 为一组 → 由 SM Partition 的一个 Warp Scheduler 调度
```

---

## 5. Block 和 Grid 的大小限制

| 限制项 | A100 值 | 说明 |
|------|:---:|------|
| 每 block 最大 thread 数 | 1024 | 硬件上限 |
| 每 block x 维最大 | 1024 | `blockDim.x ≤ 1024` |
| 每 block y 维最大 | 1024 | `blockDim.y ≤ 1024` |
| 每 block z 维最大 | 64 | `blockDim.z ≤ 64` |
| 每 grid x 维最大 | 2^31-1 | ~21 亿 |
| 每 grid y/z 维最大 | 65535 | |

**常见 blockDim 选择 (面试可能问为什么):**

| blockDim | warp 数 | 使用场景 |
|:---:|:---:|------|
| 32 | 1 | 最小 block, 低延迟 |
| 64 | 2 | 寄存器压力大时 |
| **128** | **4** | 常用, 每个 block 4 warps, 良好 latency hiding |
| **256** | **8** | 最常用, LeetCUDA 几乎所有 kernel 都用 256 |
| 512 | 16 | 需要更多 shared memory 协作时 |
| 1024 | 32 | block 维度的理论上限 |

**为什么 256 最常用?**
- 8 warps × 4 SM Partitions = 每个 Partition 2 warps, 调度良好
- 寄存器: 256 threads × 64 regs = 16384, 每 SM 可驻留 4 blocks
- Shared memory: 256 threads 通常不会吃满 shared memory
- 网格大小合理: 足够多的 block 来隐藏延迟

---

## 6. 内存管理: 基本 API

```cuda
// 分配 device 内存
cudaError_t cudaMalloc(void **devPtr, size_t size);

// 释放 device 内存
cudaError_t cudaFree(void *devPtr);

// Host → Device 拷贝
cudaError_t cudaMemcpy(void *dst, const void *src,
                       size_t count, cudaMemcpyKind kind);
// kind: cudaMemcpyHostToDevice
//        cudaMemcpyDeviceToHost
//        cudaMemcpyDeviceToDevice
```

**完整示例:**

```cuda
#include <cstdio>

__global__ void vector_add(const float *a, const float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main() {
    int N = 1 << 20;  // 1M elements
    size_t bytes = N * sizeof(float);

    // Host 端
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f; h_b[i] = 2.0f;
    }

    // Device 端
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // H→D
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // D→H
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < N; i++)
        if (fabs(h_c[i] - 3.0f) > 1e-5) printf("Error at %d!\n", i);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

---

## 7. 错误检查: 必须养成的习惯

CUDA API 都返回 `cudaError_t`, 不检查等于白干:

```cuda
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", \
                    __FILE__, __LINE__, \
                    cudaGetErrorString(err_)); \
            exit(1); \
        } \
    } while(0)

// 用法
CUDA_CHECK(cudaMalloc(&d_data, bytes));
CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
```

**Kernel 启动后也必须检查:**

```cuda
kernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());        // 检查 launch 是否成功
CUDA_CHECK(cudaDeviceSynchronize());   // 等 kernel 跑完, 并检查运行时错误
```

---

## 8. 编译: nvcc

```bash
# 基本编译
nvcc -o vector_add vector_add.cu

# 指定架构 (重要!)
nvcc -arch=sm_80 -o vector_add vector_add.cu   # A100
nvcc -arch=sm_89 -o vector_add vector_add.cu   # RTX 4090 (Ada)
nvcc -arch=sm_90 -o vector_add vector_add.cu   # H100

# 查看支持的架构
nvcc --list-gpu-arch

# 编译多个架构 (fat binary)
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     -o vector_add vector_add.cu
```

**`-arch` 的含义:**
- `sm_80` = 生成只能在 SM 8.0+ 上运行的 SASS 代码
- `compute_80` = 生成 PTX 中间码, 驱动可在运行时 JIT 编译到实际架构

---

## 9. `__syncthreads()` — Block 内同步

```cuda
__global__ void example(float *data) {
    // 阶段 1: 所有线程写 shared memory
    __shared__ float smem[256];
    smem[threadIdx.x] = data[threadIdx.x];

    __syncthreads();  // Barrier: 等待 block 内所有线程完成阶段 1

    // 阶段 2: 读取 shared memory (保证数据已就绪)
    if (threadIdx.x < 128)
        smem[threadIdx.x] += smem[threadIdx.x + 128];

    __syncthreads();  // Barrier: 等待阶段 2 完成
    // ...
}
```

**关键规则 (面试常考):**

1. `__syncthreads()` 只同步**同一个 block 内**的线程, 不跨 block
2. 必须在所有线程会执行到的路径上 — 不能放在条件分支里:
   ```cuda
   // ❌ 错误: 部分线程永远不会执行到 __syncthreads()
   if (threadIdx.x < 16) {
       __syncthreads();  // 死锁!
   }

   // ✅ 正确
   if (threadIdx.x < 16) {
       // 做一些事情
   }
   __syncthreads();  // 所有线程都会到这里
   ```
3. `__syncthreads()` 也作为内存 fence: 确保之前的写操作对所有线程可见

---

## 10. 异步执行与 Stream

```cuda
// 默认行为: kernel launch 是异步的 (host 不等待)
kernel<<<1, 1>>>();
printf("Launched!\n");  // 可能在 kernel 跑完之前就打印

// 强制同步
cudaDeviceSynchronize();  // 等所有 stream 完成

// 多个 stream 实现并发
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

kernel_A<<<grid, block, 0, s1>>>();  // stream 1
kernel_B<<<grid, block, 0, s2>>>();  // stream 2, 可能和 A 并行

cudaStreamSynchronize(s1);  // 只等 stream 1
cudaDeviceSynchronize();     // 等所有
```

Stream 是实现 compute-memory overlap 的基础 (后面 HGEMM 教程会深入)。

---

## 11. 面试高频问题

### Q1: blockDim, gridDim 怎么选?
**答:** 
- blockDim: 必须是 32 的倍数 (warp 对齐原则), 最常用 128 或 256
- gridDim: `(N + blockDim - 1) / blockDim`, 保证覆盖所有元素
- 每 SM 应驻留至少 4-8 warps 来隐藏延迟

### Q2: `__global__` 和 `__device__` 的区别?
**答:** `__global__` 是 kernel 入口, host 调用, 在 device 执行, 返回 `void`, 异步。`__device__` 是辅助函数, 只能在 device 端调用, 可返回值, 同步执行。

### Q3: threadIdx 和 blockIdx 的范围?
**答:** `threadIdx ∈ [0, blockDim-1]`, `blockIdx ∈ [0, gridDim-1]`。都在 kernel launch 时由 `<<<grid,block>>>` 定义。

### Q4: `__syncthreads()` 同步范围是多大?
**答:** 仅同步同一 block 内的所有线程。不能跨 block 同步。跨 block 同步需要多次 kernel launch。

### Q5: 为什么 CUDA 用 `cudaMalloc` 而不是 `malloc`?
**答:** `malloc` 分配 CPU 内存 (host)。GPU 有自己的显存 (device), 需要 `cudaMalloc` 在 device 上分配。CPU 不能直接访问 device 指针, 反之亦然 (除非 Unified Memory)。

### Q6: block size 必须是 32 的倍数吗?
**答:** 技术上不是硬要求, 但如果不是 32 倍数, 最后一个 warp 会有 inactive thread, 浪费硬件。最佳实践: blockDim 一定是 32 倍数。

---

## 12. 参考链接

- [CUDA C++ Programming Guide - Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)
- [CUDA C++ Programming Guide - Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)
- [LeetCUDA](/home/hz/code/cpp/third_party/LeetCUDA/README.md)
- [An Easy Introduction to CUDA - NVIDIA Developer Blog](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

---

## 13. 学习检查清单

- [ ] 能手写一个 vector_add kernel (包括 host 端代码)
- [ ] 理解 `<<<gridDim, blockDim>>>` 四个参数的含义
- [ ] 背熟 6 个内置变量
- [ ] 能用 `dim3` 写二维 kernel
- [ ] 正确使用 `CUDA_CHECK` 错误处理
- [ ] 知道 `__syncthreads()` 的三个关键规则
- [ ] `nvcc -arch=sm_80` 的含义
- [ ] 能回答上面 6 个面试问题

