# 05 Warp Shuffle 原语: GPU 最快的线程间通信
> 对象: CUDA / GPU 零基础
> 前置: 01_gpu_hardware_architecture.md, 04_warp_execution_model.md
> 目标: 面试能手写所有 shuffle 原语, 懂 mask, 懂数据流, 懂和 shared memory 的对比
---
## 1. Warp Shuffle 是什么
**Warp Shuffle 让同一个 warp 内的 32 个线程直接交换寄存器中的值，不需要经过 shared memory。**
**传统方式（shared memory）:**
- thread 0 写 `val` 到 `smem[0]`
- `__syncthreads()` （昂贵的 barrier 同步）
- thread 1 从 `smem[0]` 读 `val`
- 延迟：`smem write` + `barrier` + `smem read` ≈ **20–30 cycles**
**Warp Shuffle:**
- thread 1 直接从 thread 0 的寄存器取值
- 延迟：~ **5 cycles**
- 不需要 shared memory，**不需要** `__syncthreads()`
**Shuffle 是一条硬件指令**，在 warp 内的 32 个寄存器间直接路由数据，这是 GPU 上最快的线程间通信方式。
---
## 2. 核心概念: Lane ID 与 Mask
### 2.1 Lane ID
```
warp = 32 threads
每个 thread 在 warp 内的编号 = lane_id = threadIdx.x % 32
warp 内 32 个 lane:
  lane 0, lane 1, lane 2, ..., lane 31
```
Shuffle 操作总是在同一个 warp 的 32 个 lane 之间进行。
### 2.2 Mask 参数
从 Volta (SM70) 开始，所有 shuffle 函数必须带 `_sync` 后缀并显式传 mask：
```cuda
T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width=warpSize);
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=warpSize);
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
```
| mask 值 | 含义 |
|--------|------|
| `0xffffffff` | 全部 32 个 lane 参与 |
| `0x0000ffff` | 只有 lane 0–15 参与 |
| `__activemask()` | 当前活跃的 lane（考虑了分支发散） |
**mask 的作用:**
1. 告诉硬件哪些 lane 参与通信。
2. 不参与的 lane 的数据未定义。
3. **参与的 lane 必须到达同一条 shuffle 指令** (否则行为未定义)。
> **关键约束:** 同一个 warp 内所有 lane 必须执行同一条 shuffle 指令（在 `__syncwarp()` 之后确保），否则行为未定义。如果某些 lane 因为分支发散没执行到，用 `__activemask()` 作为 mask。
---
## 3. `__shfl_down_sync` — 向下传递，树形归约
### 3.1 语义
```cuda
__shfl_down_sync(mask, val, delta):
  lane i 收到 lane (i + delta) 的 val
  如果 i + delta >= 32 (或 >= width), lane i 保持自己的 val
```
### 3.2 数据流动图（delta 逐步减半）
```
初始值:     [v0, v1, v2, v3, v4, v5, v6, v7, ... v31]
delta=16:
  lane 0 ← lane 16:  [v16, ...]
  lane 1 ← lane 17:  [v17, ...]
  ...
  lane 15 ← lane 31: [v31, ...]
  lane 16-31: 保持原值 (i+16 >= 32, 不变)
delta=8:
  lane 0 ← lane 8:   进一步合并
  lane 1 ← lane 9:
  ...
  lane 23 ← lane 31:
  lane 24-31: 保持
delta=4 → 2 → 1:
  最终: lane 0 持有所有 32 个值的归约结果
```
### 3.3 树形 Reduce 代码
```cuda
// 用 __shfl_down_sync 实现 warp-level sum reduce
__device__ float warp_reduce_sum_down(float val) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, delta);
    }
    // 此时只有 lane 0 持有 sum
    return val;
}
// 使用:
float my_val = data[threadIdx.x];
float warp_sum = warp_reduce_sum_down(my_val);
if (lane_id == 0) {
    // 只有 lane 0 有结果!
    shared[lane_id] = warp_sum;
}
```
**特点:**
- 只有 lane 0 持有最终结果。
- 其他 lane 的值不完整。
- 简单直观，适合教学理解数据流。
---
## 4. `__shfl_xor_sync` — 蝴蝶交换，全广播归约
### 4.1 语义
```cuda
__shfl_xor_sync(mask, val, laneMask):
  lane i 收到 lane (i ^ laneMask) 的 val
  (^ 是按位异或)
```
### 4.2 蝴蝶数据流图（butterfly）
```
初始:     [v0, v1, v2, v3, v4, v5, v6, v7, ... v31]
laneMask=16:
  lane 0 ↔ lane 16  (0^16=16, 16^16=0)
  lane 1 ↔ lane 17  (1^16=17, 17^16=1)
  ...
  lane 15 ↔ lane 31
交换后 (加和): lane 0 和 lane 16 都持有 v0+v16, lane1和lane17都有v1+v17...
laneMask=8:
  lane 0 ↔ lane 8, lane 1 ↔ lane 9, ..., lane 7 ↔ lane 15
  lane 16 ↔ lane 24, ..., lane 23 ↔ lane 31
laneMask=4:
  lane 0 ↔ lane 4, lane 1 ↔ lane 5, ...
laneMask=2:
  lane 0 ↔ lane 2, lane 1 ↔ lane 3, ...
laneMask=1:
  lane 0 ↔ lane 1, lane 2 ↔ lane 3, ...
最终: 所有 32 个 lane 都持有完整的 sum!
```
### 4.3 Butterfly Reduce 代码 (LeetCUDA 的标准写法)
```cuda
// 用 __shfl_xor_sync 实现 warp-level sum reduce
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
        // 相当于：
        //int received_val = __shfl_xor_sync(0xffffffff, old_val, mask); // 2. 硬件交换
        //val = old_val + received_val;         // 3. 将接收到的值 加给 当前线程的旧值
    }
    return val;  // 所有 lane 都有完整 sum!
}
```
**特点:**
- **所有 32 个 lane 都有最终结果** (自动 broadcast)。
- 不需要额外的 broadcast 步骤。
- 这是 LeetCUDA 和几乎所有生产代码的默认选择。
### 4.4 为什么 xor 比 down 更好?
| 方面 | `shfl_down` | `shfl_xor` |
|------|------|------|
| 结果持有者 | 只有 lane 0 | 所有 32 个 lane |
| 后续 cross-warp 阶段 | 需要先 broadcast lane0 → 30 lanes | 直接用, 所有 lane 都有值 |
| 代码量 | 需要额外 `__shfl_sync` broadcast | 不需要 |
**所以生产代码用 `shfl_xor`, 教学用 `shfl_down` 帮助理解。**
---
## 5. `__shfl_up_sync` — 向上传递, Prefix Sum
### 5.1 语义
```cuda
__shfl_up_sync(mask, val, delta):
  lane i 收到 lane (i - delta) 的 val
  如果 i - delta < 0, lane i 保持自己的 val
```
### 5.2 用于 Prefix Sum (Hillis-Steele 算法)
```cuda
// 初始: [a, b, c, d, e, f, g, h, ...]
// offset=1: lane i 收 lane i-1
// → [a, a+b, b+c, c+d, d+e, e+f, f+g, g+h, ...]
// offset=2: lane i 收 lane i-2
// → [a, a+b, a+b+c, a+b+c+d, a+b+c+d+e, ...]
// offset=4: ...
// offset=8: ...
// offset=16: ...
// 最终: lane i 持有 v0+v1+...+vi (inclusive prefix sum)
```
### 5.3 完整代码
```cuda
__device__ float warp_inclusive_scan_f32(float val) {
    int lane = threadIdx.x % 32;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float tmp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += tmp;
    }
    return val;
}
```
**关键: 为什么 scan 必须用 `tmp` 暂存, 而前面的 reduce 不用?**
```cpp
// ✅ 正确
float tmp = __shfl_up_sync(0xffffffff, val, offset);
if (lane >= offset) val += tmp;
// ❌ 错误 — shuffle 被关进了 if 里
if (lane >= offset) val += __shfl_up_sync(0xffffffff, val, offset);
```
**深入解析：**
1.  **问题本质**：不是 C++ 的“求值顺序”问题（`val += __shfl(...)` 这一行本身是安全的，shuffle 是单条指令、写回 `val` 在最后一步，编译器不可能把写回提前到 shuffle 之前）。
2.  **真正原因**：scan 多了一个 `if (lane >= offset)` 条件，这是为了**保证逻辑正确性**（低位 lane 不应该加上自己的值）。
3.  **硬件约束**：`__shfl_up_sync` 是 **Warp 级集体操作**。硬件要求 `mask` 中指定的所有 lane **必须同时执行** 这条 shuffle 指令。
4.  **冲突点**：
    - 如果你把 shuffle 塞进 `if` 里面：
    - `lane < offset` 的线程会**跳过**该指令（不执行）。
    - 但 `mask=0xffffffff` 告诉硬件“32 个 lane 都在参与”。
    - 这导致硬件状态不一致（预期 32 路交换，实际只有 16 路在跑），结果是**行为未定义**（可能死锁、读到垃圾值）。
5.  **解决方案**：
    - 为了同时满足“低位不加值”和“全员 shuffle”，必须把 **shuffle 指令无条件提升到 `if` 外面**。
    - 用 `tmp` 接住交换回来的值。
    - 再用 `if` 决定是否把 `tmp` 加到 `val` 上。
**一句话总结：** `tmp` 是为了“所有 lane 都 shuffle，但只有一部分 lane 加”，本质是“条件相加 + shuffle 必须全员执行”这两个约束的产物。reduce 是无条件 `val +=`，没有 `if`，直接内联即可。
### 5.4 Exclusive Scan
```cuda
__device__ float warp_exclusive_scan_f32(float val) {
    int lane = threadIdx.x % 32;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float tmp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += tmp;
    }
    // exclusive: 每个 lane 拿它之前所有 lane 的和
    float result = __shfl_up_sync(0xffffffff, val, 1);
    return (lane == 0) ? 0.0f : result;
}
```
---
## 6. `__shfl_sync` — 广播（Broadcast）
```cuda
T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
```
**语义:** 所有 lane 收到 `srcLane` 的 `var`。
```cuda
// 把 lane 0 的值广播给所有 lane
float val = __shfl_sync(0xffffffff, my_val, 0);
// 也用于 cross-warp reduce 后把结果广播给 block 内所有 thread
// (在 block_reduce 的第二步, warp0 的 lane0 持有结果后):
value = __shfl_sync(0xffffffff, value, 0, 32);
```
---
## 7. 四种 Shuffle 对比总结
| 原语 | 数据流向 | 主要用途 | 结果分布 |
|------|------|------|------|
| `__shfl_down_sync` | lane i ← lane i+delta | **树形 Reduce** | 仅 lane 0 有最终结果 |
| `__shfl_xor_sync` | lane i ↔ lane i^mask | **蝴蝶 Reduce (生产首选)** | **所有 lane 都有** |
| `__shfl_up_sync` | lane i ← lane i-delta | **Prefix Sum (Scan)** | 每个 lane 有其前缀和 |
| `__shfl_sync` | 所有人 ← srcLane | **Broadcast** | 所有 lane 相同 |
---
## 8. Warp-Level Min/Max Reduce
```cuda
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
__device__ __forceinline__ float warp_reduce_min_f32(float val) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        val = fminf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
// 初始化: max 用 -FLT_MAX, min 用 FLT_MAX
```
---
## 9. 其他 Warp-Level 原语
### 9.1 Vote 函数
```cuda
// 所有 lane 的 predicate 都为 true?
int __all_sync(unsigned mask, int predicate);
// 任一 lane 的 predicate 为 true?
int __any_sync(unsigned mask, int predicate);
// 返回 32-bit mask, bit i=1 表示 lane i 的 predicate 为 true
unsigned __ballot_sync(unsigned mask, int predicate);
```
**典型用法:**
```cuda
// 检查 warp 内是否有 lane 的数据满足条件
if (__any_sync(0xffffffff, my_val > threshold)) {
    // 至少有一个 lane 满足
}
// 收集哪些 lane 是 active 的
unsigned active = __ballot_sync(0xffffffff, is_valid);
```
**为什么 `predicate` 是 `int` 而不是 `bool`?**
`my_val > threshold` 在 C++ 里确实是 `bool`, 但这些 vote intrinsic 的签名沿用了 **C 语言的"真值(truthy)"约定**: 没有 `bool` 类型, 一律用 `int` 表示条件, 规则是 **"非 0 即真, 0 即假"**。CUDA 的 vote/ballot 内建函数早于(且兼容)C++ 的 `bool`, 所以参数类型固定为 `int`。
- 你传进去的 `bool` 会**隐式转换**成 `int`: `true → 1`, `false → 0`, 转换是自动且无损的, 所以 `__any_sync(mask, my_val > threshold)` 写法完全正确。
- intrinsic 内部**只看 predicate 是否为 0**, 不在乎具体数值。所以下面几种写法等价:
```cuda
__any_sync(mask, my_val > threshold);  // bool → int(0/1)
__any_sync(mask, flag);                // flag 是 int, 非 0 算 true
__any_sync(mask, count);               // count=5 也算 true (非 0)
__any_sync(mask, ptr != nullptr);      // 同样是 bool → int
```
- 返回值同理是 `int`: `__all_sync`/`__any_sync` 返回 0 或 1, 可直接当条件用。
> 小结: `int predicate` 是 C 风格 API 的历史产物, "条件"在这里 = "非零的整数"。传 `bool` 没问题(会自动变 0/1), 传任意非零 `int` 也会被当成 true。
**`__ballot_sync` 详解: 把 32 个 lane 的判断压成一个 32-bit 整数**
```cuda
unsigned active = __ballot_sync(0xffffffff, is_valid);
//  active 的 bit i = 1  ⟺  lane i 的 is_valid 非 0 (且 lane i 在 mask 内)
//  例: lane 0/1/3 满足, 其余不满足 → active = 0b...0001011 = 0xB
```
`__any_sync`/`__all_sync` 只回答"有没有/是不是全部", 而 `__ballot_sync` 回答"**具体是哪些 lane**", 信息量最大——它把全 warp 的布尔判断打包成一个位掩码, 后续用位运算就能算计数、排名、找头一个。
**第一步: `is_valid` 怎么设?** 它就是这个 lane"是否参与/是否命中"的布尔条件, 几种典型来源:
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
bool is_valid = (idx < N);                 // ① 边界检查: 尾块越界的 lane 不算
bool is_valid = (data[idx] > threshold);   // ② 数据筛选: 命中条件的 lane
bool is_valid = (data[idx] != 0);          // ③ 稀疏: 非零元素
bool is_valid = predicate_fn(data[idx]);   // ④ 任意谓词
```
**第二步: `active` 怎么用?** 配合几个位操作 intrinsic(都是单条硬件指令):
| intrinsic | 作用 |
|---|---|
| `__popc(active)` | active 中 1 的个数 = warp 内有几个 lane 满足 |
| `__ffs(active)` | 最低位 1 的位置(从 1 计数, 0 表示没有), `__ffs(active)-1` = 第一个满足的 lane id |
| `active & (1u << lane)` | 判断本 lane 是否满足(是否被选中) |
| `__popc(active & ((1u << lane) - 1))` | 本 lane 之前有几个满足的 lane = 本 lane 在"满足者"里的**排名** |
**用法 A — 统计 warp 内命中数量:**
```cuda
unsigned active = __ballot_sync(0xffffffff, data[idx] > threshold);
int num_hit = __popc(active);   // 这个 warp 有几个元素超过阈值
```
**用法 B — Leader 选举(选一个 lane 代表整个 warp 做事):**
```cuda
unsigned active = __ballot_sync(0xffffffff, is_valid);
int leader = __ffs(active) - 1;        // 取最低位满足的 lane 当 leader
if (lane == leader) {
    // 只让 leader 执行一次(如打印、占坑、写一个 header)
}
```
**用法 C — Warp 聚合原子操作:**
把 warp 内所有满足条件的元素紧凑写到输出数组。**关键: 不让每个 lane 各做一次 `atomicAdd`, 而是一个 warp 只做一次**, 把全局原子竞争降低到 1/32。这是 FlashAttention 等高性能 Kernel 中处理不规则访问的标准手法。
```cuda
// 每个满足 is_valid 的 lane 要往 out[] 追加自己的 value
unsigned active = __ballot_sync(0xffffffff, is_valid);
int leader = __ffs(active) - 1;                       // warp 代表
int total  = __popc(active);                          // 本 warp 共追加几个
int rank   = __popc(active & ((1u << lane) - 1));     // 本 lane 在其中排第几
int base;
if (lane == leader) {
    base = atomicAdd(out_count, total);   // 整个 warp 只 1 次原子操作, 抢一段连续空间
}
// 把 leader 抢到的起始下标广播给 warp 内所有 lane
base = __shfl_sync(active, base, leader);
if (is_valid) {
    out[base + rank] = value;             // 各 lane 按 rank 错位写入, 无冲突
}
```
### 9.2 Shuffle 支持的数据类型
Shuffle 支持 `int`, `unsigned`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float`, `double`。每次 shuffle 交换 32 bits (4 bytes)。如果要传更大的 struct, 需要拆分。
**哪些类型一次硬件 shuffle 就能搞定, 哪些不能?**
关键: **硬件 SHFL 指令一次只能搬运一个 32-bit 寄存器**。所以是否"一次完成"只看类型有多少 bit:
| 类型 | 大小 | 底层 SHFL 指令数 | 一次完成? |
|---|---|---|---|
| `int` / `unsigned` | 4 B (32 bit) | 1 | ✅ |
| `float` | 4 B (32 bit) | 1 | ✅ |
| `long long` / `unsigned long long` | 8 B (64 bit) | 2 | ❌ (拆高低 32 位) |
| `double` | 8 B (64 bit) | 2 | ❌ (拆高低 32 位) |
| `long` / `unsigned long` | 平台相关* | 1 或 2 | 视平台 |
> \* `long` 的大小看 ABI: Linux/macOS(LP64, CUDA 用这个)是 **8 B → 2 条指令**; Windows(LLP64)是 4 B → 1 条。所以同一份代码里 `long` 的行为跨平台不同, 想明确就用 `int`(32)或 `long long`(64)。
**那为什么列表里 64-bit 类型也"支持"?** 因为 CUDA 给这些类型提供了**库层面的重载**: 你写一次 `__shfl_xor_sync(mask, my_double, ...)`, 编译器自动把它拆成"搬高 32 位 + 搬低 32 位"**两条 SHFL 指令**, 再拼回一个 `double`。**从代码看是一次调用, 从硬件看是两条指令**——功能正确, 但 64-bit 类型的 shuffle 成本约是 32-bit 的 2 倍。
**为什么 struct 不行, 必须手动拆?** 因为这些重载只覆盖上面列出的标量类型, **没有针对任意 struct 的重载**。编译器不知道怎么把一个自定义 struct 映射到 32-bit 寄存器序列, 所以你得自己按字段拆开, 每个字段单独 shuffle:
```cuda
// 传一个 struct 的两个 float 字段:
struct __align__(8) MD { float m; float d; };
__device__ MD warp_reduce_md(MD value) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(0xffffffff, value.m, mask);
        other.d = __shfl_xor_sync(0xffffffff, value.d, mask);
        // 合并逻辑...
    }
    return value;
}
```
这是 FlashAttention 中 online softmax 的基础。
---
## 10. 完整的 Block Reduce 架构（Shuffle + Shared Memory）
### 10.1 两阶段设计
```
Block = 256 threads = 8 warps × 32 lanes
阶段 1: WARP-SHUFFLE (寄存器通信, 0 次 sync)
  每个 warp 独立用 shfl_xor 做 reduce
  → 8 个 warp sums
阶段 2: CROSS-WARP (1 次 sync)
  8 个 warp sums → shared memory[8]
  __syncthreads()
  warp0 的 8 个线程 → 再做一次 warp shuffle reduce
  thread0 写 output
```
### 10.2 完整代码
```cuda
#define WARP_SIZE 32
template <const int NUM_THREADS = 256>
__global__ void block_reduce_sum_f32_kernel(
    const float *input, float *output, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];
    float val = (idx < N) ? input[idx] : 0.0f;
    // === 阶段 1: Warp Reduce (butterfly) ===
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    // === 阶段 2: Cross-Warp Reduce ===
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    val = (tid < NUM_WARPS) ? smem[tid] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int mask = NUM_WARPS >> 1; mask >= 1; mask >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, mask);
        }
    }
    if (tid == 0) output[blockIdx.x] = val;
}
```
### 10.3 复杂度对比
| 方法 | Shared Memory 访问 | `__syncthreads()` | 延迟 |
|------|-------------------|-----------------|------|
| 纯 Shared Memory (Puzzle 8) | 8×256 reads | 8 | 高 |
| **Warp Shuffle + Shared** | 8 writes + 8 reads | **1** | **低** |
---
## 11. `__device__` Helper 版本 (LayerNorm/Softmax 用)
当 reduce 作为 kernel 内的一步 (而不是独立的 kernel), 需要所有 thread 都拿到结果:
```cuda
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    __shared__ float shared[NUM_WARPS];
    // Warp reduce
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    // Cross-warp
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? shared[threadIdx.x] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int mask = NUM_WARPS >> 1; mask >= 1; mask >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, mask);
        }
    }
    // ★ 关键: broadcast 给所有 thread
    val = __shfl_sync(0xffffffff, val, 0, 32);
    return val;  // block 内每个 thread 都拿到 sum
}
```
**区别:** 
- `__global__` kernel 版本: 只有 thread0 写 global memory
- `__device__` helper 版本: broadcast 给所有 thread, 让调用方每个 thread 都能用
---
## 12. 面试高频问题
### Q1: Warp Shuffle 比 Shared Memory 快多少? 为什么?
**答:** Shuffle ~5 cycles/轮，Shared Memory ~20–30 cycles (含 barrier)。Shuffle 在寄存器间直接路由数据，不需要经过 shared memory pipeline 和 `__syncthreads()` barrier。
### Q2: `shfl_down` 和 `shfl_xor` 的核心区别?
**答:** `shfl_down` 数据从高位流向低位，最终只有 lane 0 有归约结果。`shfl_xor` 双向交换，最终所有 32 个 lane 都有结果 (自动 broadcast)。生产代码几乎都用 `shfl_xor`。
### Q3: `shfl_up` 的 prefix sum 为什么要用临时变量? reduce 为什么不用?
**答:** 不是求值顺序问题(`val += __shfl(...,val,...)` 单行其实是安全的, shuffle 是单条指令、写回 val 在最后一步)。真正原因是 scan 多了一个 `if (lane >= offset)` 条件: scan 每个 lane 的前缀和都要正确, 必须挡住 `lane < offset` 的 lane(否则它们 `val += 自己 = 2*val`)。但 shuffle 是 warp 级集体操作, mask 内所有 lane 必须执行同一条 shuffle, 不能塞进 `if` 里, 所以要把 shuffle 无条件提到外面, 用 `tmp` 接住返回值再条件相加。reduce 是无条件 `val += 收到值`, 没有 `if`, 故直接内联、不需要 `tmp`。
### Q4: mask 参数 `0xffffffff` 和 `__activemask()` 的区别?
**答:** `0xffffffff` 表示所有 32 个 lane 参与 (即使有些 lane 因为分支发散被禁用)。`__activemask()` 只包含当前活跃的 lane。在有分支发散时前者可能死锁, 后者安全。
### Q5: 为什么 Warp Shuffle 不能跨 warp?
**答:** Shuffle 是硬件寄存器路由, 只在一个 warp 的 32 个寄存器间工作。跨 warp 通信必须经过 shared memory 或 global memory。
### Q6: block reduce 为什么是两阶段?
**答:** 阶段 1 (warp reduce) 用 shuffle 在寄存器里做, 0 sync。阶段 2 (cross-warp) 必须用 shared memory + 1 次 `__syncthreads()`, 因为 8 个 warp 不共享寄存器。
---
## 13. 参考链接
- [CUDA C++ Programming Guide - Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- [Using CUDA Warp-Level Primitives - NVIDIA Developer Blog](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [LeetCUDA block_all_reduce.cu](https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/reduce/block_all_reduce.cu)
- `reduce_warp_learning.md` — Warp reduce 实战 (Puzzle 对应)
- `scan_warp_learning.md` — Warp scan 实战 (Puzzle 对应)
---
## 14. 学习检查清单
- [ ] 能画 `shfl_down_sync` 的树形数据流图 (delta=16,8,4,2,1)
- [ ] 能画 `shfl_xor_sync` 的蝴蝶数据流图
- [ ] 能手写 `warp_reduce_sum_f32` (用 shfl_xor)
- [ ] 能手写 `warp_inclusive_scan_f32` (用 shfl_up)
- [ ] 理解 `shfl_down` (仅 lane0) vs `shfl_xor` (全 lane) 的选择
- [ ] 能手写完整的 block_reduce (两阶段: warp shuffle + SMEM)
- [ ] 知道 `__device__` helper 版本需要 broadcast
- [ ] 理解 mask 的含义和 `__activemask()` 的使用场景
- [ ] 能手写 `warp_reduce_max_f32` 和 `warp_reduce_min_f32`
- [ ] 能回答上面 6 个面试问题
