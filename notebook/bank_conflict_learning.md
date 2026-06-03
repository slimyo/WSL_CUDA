# Shared Memory Bank Conflict 学习笔记

> 对应文件: `src/Puzzle/bank_conflict.cu`
> 前置知识: Puzzle 8/11 (shared memory reduce), `reduce_warp_learning.md`

---

## 1. Shared Memory Bank 结构

每个 SM 的 shared memory 被分成 **32 个 bank**，每个 bank 4 bytes 宽:

```
Bank:   0     1     2    ...   31
       [4B]  [4B]  [4B]  ...  [4B]
       [4B]  [4B]  [4B]  ...  [4B]
       [4B]  [4B]  [4B]  ...  [4B]
```

地址到 bank 的映射: `bank = (byte_address / 4) % 32`

连续 4-byte 元素映射到连续 bank，第 32 个元素绕回 bank 0:

```
float arr[64]:
  arr[0] → bank 0
  arr[1] → bank 1
  ...
  arr[31] → bank 31
  arr[32] → bank 0  ← 和 arr[0] 同 bank!
```

---

## 2. Bank Conflict 的定义

**同一 warp 内多个线程访问同一 bank 的不同地址** → bank conflict。

- 无冲突 (1-way): 每个 bank 最多被 1 个线程访问 → 1 cycle
- n-way conflict: n 个线程访问同一 bank → 请求串行化 → n cycles

### 2.1 stride=1 (无冲突)

```cuda
// 32 threads, 每个读自己行的连续列
for (int i = 0; i < 32; i++)
    sum += smem[threadIdx.x][i];  // bank = i, 各线程不同 bank
```
**1-way conflict = 无冲突。**

### 2.2 stride=32 (全冲突)

```cuda
// 32 threads, 每个读不同行的同一列
for (int i = 0; i < 32; i++)
    sum += smem[i][threadIdx.x];  // 全部 hit bank threadIdx.x
```
`smem[i][tid]` 的 bank = (i×32 + tid) % 32 = tid。所有线程同一 bank。
**32-way conflict (最坏情况)。**

### 2.3 Padding 消除冲突

```cuda
__shared__ float smem[32][33];  // 32×33, +1 padding
// smem[i][tid] 的 bank = (i×33 + tid) % 32 = (i + tid) % 32
// i 变化时 bank 也在变 → 无冲突!
```

多加 1 列→行 stride=33(奇数)→破坏与 32 的整除→bank 错开。

---

## 3. 你的 bank_conflict.cu 三个 kernel

### Kernel 1: stride=1 (基线)

```cuda
__global__ void smem_read_stride_1(float *output) {
    __shared__ float smem[32][32];
    int tid = threadIdx.x;
    for (int i = 0; i < 32; i++) smem[tid][i] = tid * 32 + i;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) sum += smem[tid][i];
    output[tid] = sum;
}
```
访问 `smem[tid][i]` → bank = i，各线程不同 bank。**0 conflicts。**

### Kernel 2: stride=32 (全冲突)

```cuda
__global__ void smem_read_stride_32(float *output) {
    __shared__ float smem[32][32];
    int tid = threadIdx.x;
    for (int i = 0; i < 32; i++) smem[tid][i] = tid * 32 + i;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) sum += smem[i][tid];
    output[tid] = sum;
}
```
访问 `smem[i][tid]` → 全部 hit bank tid。**32-way conflict。**

### Kernel 3: padding 消除

```cuda
__global__ void smem_read_with_padding(float *output) {
    __shared__ float smem[32][33];  // ← +1 padding
    int tid = threadIdx.x;
    for (int i = 0; i < 32; i++) smem[tid][i] = tid * 32 + i;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) sum += smem[i][tid];
    output[tid] = sum;
}
```
访问 `smem[i][tid]`，bank = (i×33+tid)%32 = (i+tid)%32。i 变化→bank 轮转。**~0 conflicts。**

---

## 4. ncu 测量

```bash
nvcc -arch=sm_80 -o bank_conflict bank_conflict.cu
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld_sum \
  ./bank_conflict
```

预期:

| Kernel | Bank Conflicts |
|--------|:---:|
| stride_1 | ~0 |
| stride_32 | 大量 |
| with_padding | ~0 |

---

## 5. 常见冲突场景及修复

### 矩阵转置

```cuda
// ❌ 列优先写入 — 32-way conflict
smem[threadIdx.y][threadIdx.x] = val;

// ✅ Padding
__shared__ float smem[32][33];           // +1 padding
smem[threadIdx.y][threadIdx.x] = val;   // bank = (y+x)%32
```

### Broadcast 特例 (同 bank 同地址 → 无冲突)

```cuda
float val = smem[0];  // 所有 32 线程读同一个地址 → broadcast, 1 cycle
```

但要区分: 同 bank **不同地址**才是 conflict。同 bank **同一地址**是 broadcast。

---

## 6. Padding 策略速查

| 数据类型 | Bank 数 | 冲突 stride | Padding |
|----------|:---:|-------------|---------|
| float (4B) | 32 | 32 的倍数 | +1 float |
| double (8B) | 16 | 16 的倍数 | +1 double |

通用模式: `smem[ROWS][COLS + 1]`，使行 stride 为奇数。

---

## 7. 学习检查清单

- [ ] 理解 32 bank × 4B 的结构
- [ ] 能计算 `byte_addr/4%32` 确定 bank
- [ ] 知道 stride=1 和 stride=32 的冲突差异
- [ ] 理解 padding 为什么能消除冲突
- [ ] 知道 broadcast 是同 bank 同地址的特例
- [ ] 能用 ncu 实测 bank conflict 计数

## 8. 推荐阅读

1. `bank_conflict.cu` 三个 kernel — 直接看访问模式差异
2. LeetCUDA `layer_norm.cu` 的 `__shared__ float shared[NUM_WARPS]` — warp reduce 阶段天生无冲突
3. LeetCUDA `swizzle/` — swizzle layout 是更高级的冲突消除方案

