# 算子接入 PyTorch：扩展、注册、autograd 与数值验证

> 对象: 算子岗（"会写 kernel"和"kernel 能进生产框架"之间隔着本章）
> 前置: triton/ 教程、14_kernel_routes.md；写过任一 P2/P3 的 kernel
> 目标: 面试能答"你的 kernel 怎么接进 PyTorch、怎么不破坏 autograd 和 torch.compile、
>       怎么证明它是对的"——这三问是算子岗工程面的标配

---

## 1. 接入方式全景（按工程重量排序）

| 方式 | 适用 | 一句话 |
|------|------|------|
| ① `torch.utils.cpp_extension.load()` | 实验/学习 | JIT 编译 .cu，改完即跑，LeetCUDA 全用它 |
| ② setup.py + pybind11 | 正式包 | 同上的 AOT 版，发布用 |
| ③ `torch.library.custom_op` | Triton/Python 算子 | 2024+ 推荐路径，自动处理 compile 兼容 |
| ④ ATen dispatcher 注册 | 框架级开发 | 给已有 op（如 aten::mm）注册新 backend 实现 |

### ① 最小可用模板（学习期就用这个）

```python
# bind.cpp ---------------------------------------------------
#include <torch/extension.h>
torch::Tensor my_rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "fp16 only");
    auto y = torch::empty_like(x);
    launch_rmsnorm(                       // 你的 .cu 里的 launcher
        (half*)x.data_ptr(), (half*)w.data_ptr(), (half*)y.data_ptr(),
        x.size(0), x.size(1), (float)eps,
        at::cuda::getCurrentCUDAStream());          // ★用 PyTorch 的当前流!
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("rmsnorm", &my_rmsnorm); }
```
```python
# python 侧
from torch.utils.cpp_extension import load
ext = load(name="myops", sources=["bind.cpp", "rmsnorm.cu"],
           extra_cuda_cflags=["-O3", "-arch=sm_75", "--use_fast_math"])
```

**三条铁律（违者必出诡异 bug）：**
1. **拿 `getCurrentCUDAStream()`**，别用默认流——否则和 PyTorch 的异步序错乱（20 章）
2. 进 kernel 前 `TORCH_CHECK` 设备/dtype/contiguous——非 contiguous 输入按
   contiguous 解读是经典静默错误（要么 check 拒掉，要么支持 stride）
3. 输出用 `torch.empty_like` 等 PyTorch 分配器创建——别自己 cudaMalloc
   （绕开 caching allocator = 内存碎片 + 同步点）

## 2. autograd：让 kernel 可训练

```python
class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        y, rstd = ext.rmsnorm_fwd(x, w, eps)   # fwd 顺便存反向要用的统计量
        ctx.save_for_backward(x, w, rstd)
        return y
    @staticmethod
    def backward(ctx, dy):
        x, w, rstd = ctx.saved_tensors
        dx, dw = ext.rmsnorm_bwd(dy, x, w, rstd)
        return dx, dw, None                     # 与 forward 入参一一对应
```
面试追问点：**save_for_backward 存什么是性能决策**——存 rstd（重计算省显存）
还是存归一化后的 x_hat（省计算费显存），这就是"重计算 vs 缓存"trade-off 的微观版
（和 FlashAttention backward 的重计算同源, 10 章）。

## 3. torch.compile 兼容（2025+ 面试新热点）

```
问题: compile 的图捕获(Dynamo)遇到不认识的扩展函数 → graph break → 性能损失
解法: torch.library.custom_op 注册 + 提供 FakeTensor 实现（"meta kernel"）:

@torch.library.custom_op("myops::rmsnorm", mutates_args=())
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    ...  # 调真 kernel

@rmsnorm.register_fake                    # 只算输出 shape/dtype，不碰数据
def _(x, w, eps): return torch.empty_like(x)

→ compile 时用 fake 版做 shape 推导，图不断；运行时走真 kernel
→ torch.library.opcheck(rmsnorm, (x, w, 1e-5)) 一键检查注册完整性
```
为什么需要 fake/meta：编译期没有真数据，但图优化要知道每个节点的输出形状——
**"shape 函数与计算分离"是所有 AI 编译器的共同设计**（19 章 TVM/XLA 同理）。

## 4. 数值验证方法论（算子岗的"测试驱动"）

```
分层验收（每层过了才看下一层）:
  L1 形状/dtype/边界: 空 tensor、单元素、非整除尺寸、非 contiguous 输入
  L2 数值对拍: 与参考实现比（参考 = fp64 的朴素实现，不是另一个 fp16 kernel!
       两个 fp16 实现互相对 = 可能一起错）
  L3 容差怎么定（背下来）:
       fp32: atol≈1e-6  rtol≈1e-5
       fp16: atol≈1e-3  rtol≈1e-3   （尾数 10 bit → 相对误差 ~5e-4 起步）
       且误差随归约长度 K 增长 ~sqrt(K)（随机舍入游走）→ 大 K 的 GEMM 放宽容差
  L4 统计验证: 随机 1000 组形状/种子扫一遍（fuzz）；
       含 atomic 的 kernel 注意运行间不确定性（求和顺序不同），
       面试题"为什么 atomicAdd 的结果每次不一样、怎么办"——答: 浮点加法不结合；
       需要确定性时换树形归约/固定顺序
  L5 梯度检查: torch.autograd.gradcheck（用 fp64 输入）
```

## 5. 工业参照

- **LeetCUDA**: 每个 kernel 目录的 `pybind/` + `*.py` 就是方式①②的实例，
  挑 `kernels/rms-norm/` 完整读一遍 binding→python 测试的链路
- **vLLM**: `csrc/torch_bindings.cpp`（方式④，TORCH_LIBRARY 宏注册 + meta 函数）
- **flash-attn 包**: setup.py AOT 编译 + autograd.Function 包装，方式②+autograd 的范本

## 6. 动手任务（半天～1 天，强烈建议做）

把 P3 的 CUDA flash attention（或 triton/02 的 RMSNorm）走完全流程：
cpp_extension 接入 → TORCH_CHECK 防御 → custom_op + fake 注册 →
`torch.compile` 一个含它的小模型确认无 graph break（`TORCH_LOGS=graph_breaks`）→
opcheck + L1-L4 验证脚本。**做完你就拥有了工程面的完整答案。**

## 7. 自测 / 面试题

1. 自定义 kernel 为什么必须用 PyTorch 的 current stream 和 allocator？各举一个出错场景。
2. 非 contiguous tensor 传进只支持 contiguous 的 kernel，最坏会发生什么？两种处理策略？
3. fp16 GEMM 和 fp64 参考比，容差怎么定？为什么和 K 有关？
4. torch.compile 遇到你的扩展算子时发生什么？fake tensor 实现解决了什么？
5. 两次运行同一 kernel 结果不同，可能原因？什么时候可接受、什么时候必须修？

## 8. 参考

- PyTorch docs: Custom C++/CUDA Extensions、torch.library（custom_op 教程）
- PyTorch 源码: `aten/src/ATen/native/` 任一算子的 native_functions.yaml 注册链
- notebook: 14 章（kernel 路线）、triton/05 §4（Triton 侧的同一问题）
