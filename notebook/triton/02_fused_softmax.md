# Triton 02 · Fused Softmax 与归约类 kernel

> 目标: 掌握"每行一个 program"的归约范式，写出 fused softmax 和 RMSNorm
> 时长: 2 天 ｜ 前置: 01 篇、softmax_learning.md（safe softmax 数学）

---

## 1. 归约范式：行内归约 = tile 内 tl.max/tl.sum

softmax 每行需要 max 和 sum 两个归约。CUDA 里这是 warp shuffle + smem 两级
reduce 的体力活（reduce_warp_learning.md 整篇）；Triton 里是一个函数调用——
**编译器自动生成 shuffle/smem 归约**，`num_warps` 决定用几个 warp 协作。

```python
@triton.jit
def softmax_kernel(x_ptr, o_ptr, M, N, stride_m,
                   BLOCK_N: tl.constexpr):       # BLOCK_N = 覆盖整行的 2 的幂
    row = tl.program_id(0)                       # ① 每行一个 program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_m + cols, mask=mask,
                other=-float('inf'))             # ② max 的恒等元! 填 0 就错
    x = x - tl.max(x, axis=0)                    # ③ safe softmax
    num = tl.exp(x)
    den = tl.sum(num, axis=0)                    # mask 处 exp(-inf)=0，不污染 sum
    tl.store(o_ptr + row * stride_m + cols, num / den, mask=mask)

def softmax(x):
    M, N = x.shape
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_N < 2048 else (8 if BLOCK_N < 8192 else 16)  # ④
    o = torch.empty_like(x)
    softmax_kernel[(M,)](x, o, M, N, x.stride(0),
                         BLOCK_N=BLOCK_N, num_warps=num_warps)
    return o
```

**四个考点（标号对应代码）：**
1. **并行切分**：行间天然独立 → grid=(M,)；一行的全部数据进一个 program
2. **other 的选择**：参与 max 的填 `-inf`、参与 sum 的位置自然变 0 —— 归约恒等元
3. 这是 **fused** softmax：max/exp/sum/除法在寄存器里一气呵成，
   HBM 流量 = 读 1 + 写 1（对照 naive 三 kernel 的 5 次，P1 Step3 量化过）
4. **num_warps**：唯一需要你"想线程"的旋钮。行越长→需要越多 warp 分摊
   load 和归约。它只是 launch 提示，块内怎么分还是编译器定

## 2. 为什么这是面试金牌题

```
fused softmax = 归约 + 数值稳定 + fusion + mask 边界，四个核心点一题打包。
追问链通常是: 写 softmax → "行长超过 BLOCK 上限怎么办?"(见 §3)
            → "改成 online softmax?"(为 04 篇 flash 铺垫) → "反向呢?"
```

## 3. 行长超限：两遍法 vs online 单遍法

BLOCK_N 受寄存器/smem 限制（实测 SM75 上 16K 左右就紧张）。N 很大时：

```
方案 A 两遍: kernel1 算每行 (max, sum)，kernel2 归一化 —— 简单但多读一遍 x
方案 B 单遍 online: 块内循环扫行，维护 (m, d) 递推 —— softmax_learning §2 的实现版

@triton.jit  # 方案 B 的循环骨架（flash attention 内循环的前身!）
def softmax_online(x_ptr, o_ptr, ..., BLOCK_N: tl.constexpr):
    m = -float('inf'); d = 0.0
    for start in range(0, N, BLOCK_N):           # 编译器会展开/流水这个循环
        x = tl.load(... start + cols ..., other=-float('inf'))
        m_new = tl.maximum(m, tl.max(x, 0))
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
    # 第二遍循环用 (m, d) 归一化写出（或缓存 exp 结果, 看寄存器预算）
```
写过这个，04 篇的 flash attention 只是"把标量 (m,d) 换成按行向量、
把 x 换成 QKᵀ 的 tile"。

## 4. RMSNorm：归约范式的第二次练习（必做）

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, o_ptr, M, N, stride_m, eps,
                   BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N); mask = cols < N
    x = tl.load(x_ptr + row*stride_m + cols, mask=mask, other=0.).to(tl.float32) # ★
    rms = tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=mask)
    y = (x / rms) * w.to(tl.float32)
    tl.store(o_ptr + row*stride_m + cols, y.to(tl.float16), mask=mask)
```
★ **精度纪律：fp16 输入也要 fp32 算平方和**（layernorm_rmsnorm_learning.md 的
教训在 Triton 里同样成立，`.to(tl.float32)` 一行的事）。
验收：对照 `torch.nn.functional.rms_norm`（或手写 ref），N=4096 时带宽 >250 GB/s。

## 5. 本篇练习

1. softmax 支持 temperature（除 T）和 log_softmax 变体
2. LayerNorm forward（两个归约：mean 和 var——一遍算 sum 和 sum(x²) 的技巧）
3. 方案 A"两遍大行 softmax"完整实现，与方案 B 对比 N=64K 时的耗时
4. 【对照实验】用 ncu 抓你的 fused softmax 和 `torch.softmax`：
   对比 `dram__bytes` —— PyTorch 的也是 fused 的吗？（是，但看它额外做了什么）

## 6. 要点回顾

- 归约类 kernel 的设计三问：行怎么分给 program / other 填什么 / num_warps 多大
- fusion 的收益直接读 HBM 字节数（P1 方法论），不要只看耗时
- online 递推是从 softmax 通往 flash attention 的桥，必须亲手写一遍
