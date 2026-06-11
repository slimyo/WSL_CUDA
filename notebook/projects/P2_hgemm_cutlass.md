# P2 · HGEMM 三连：手写 WMMA → 优化到位 → CUTLASS/CuTe 重写（2-2.5 周）

> 前置阅读: tensor_cores_intro.md, hgemm_optimization.md, 14_kernel_routes.md, bank_conflict_learning.md
> 产出: ①一条 WMMA HGEMM 优化演进链（4-5 版）②同一问题的 CUTLASS 实现 ③三方对比报告（手写 vs CUTLASS vs cuBLAS）
> 简历叙事: "手写 HGEMM 达 cuBLAS xx%，并能从 CUTLASS 视角解释剩余差距"

---

## 1. 项目定义

固定问题：`C[M,N] = A[M,K] × B[K,N]`，half 输入、float 累加，
M=N=K ∈ {1024, 2048, 4096}（4096 时 3 个矩阵共 ~100MB，6GB 显存无压力）。

三条路线做同一件事，**重点不是性能数字，是"能讲清三者差异"**（14 章选型轴的实证）。

## 2. 工程框架

```
src/projects/p2_hgemm/
├── hgemm_v0_naive.cuh        # FP32 CUDA core，对照组
├── hgemm_v1_wmma.cuh         # WMMA 16x16x16 直取直算
├── hgemm_v2_smem.cuh         # smem 分块 + 共享内存重用
├── hgemm_v3_padded.cuh       # +smem padding 消 bank conflict
├── hgemm_v4_dbuf.cuh         # +双缓冲（SM75 无 cp.async 的软流水）
├── hgemm_v5_vec_swizzle.cuh  # +float4 加载 +block swizzle（选做）
├── hgemm_cutlass.cu          # CUTLASS 2.x device GEMM
├── main.cu                   # 统一 bench: 正确性(vs cuBLAS) + 计时
└── plot_gflops.py            # GFLOPS-vs-size 曲线（5 版 + cutlass + cublas）
```

编译注意：`-arch=sm_75`；CUTLASS 直接用本地仓库，不用下载：
```cmake
include_directories(${CMAKE_SOURCE_DIR}/third_party/LeetCUDA/third-party/cutlass/include)
```

## 3. 分步任务

### Step 1（2 天）v0+v1：跑通 WMMA

```cuda
// v1 核心：每 warp 算 C 的一个 16×16 tile，沿 K 循环
wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16,16,16, float> acc;   // ★float 累加
wmma::fill_fragment(acc, 0.0f);
for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a_frag, A + row*K + k, K);   // 直接从 global 读
    wmma::load_matrix_sync(b_frag, B + k*N + col, N);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
}
wmma::store_matrix_sync(C + row*N + col, acc, N, wmma::mem_row_major);
```
验收：结果对（vs cuBLAS，相对误差 <1e-2）；ncu 确认 tensor pipe 已激活
（`sm__pipe_tensor_cycles_active` > 0）。此版性能很差——正常，它是基线。

### Step 2（3 天）v2/v3：smem 分块 + 去 conflict

Block tile 128×128×32（A tile 128×32 + B tile 32×128，half ≈ 16KB，在 64KB smem 内），
每 block 8 warps、每 warp 算 64×32 的 C 子块（4×2 个 wmma fragment）。
v3 给 smem 行加 padding（或学 `kernels/swizzle/` 用 xor-swizzle）。
**验收：ncu `l1tex__data_bank_conflicts` 降到 ~0；GFLOPS 至少 3× 于 v1。**

### Step 3（2 天）v4：软流水双缓冲

SM75 没有 `cp.async`，软流水 = global→寄存器→smem 两段搬运手动交错：
```
载入 tile[k+1] 到寄存器  →  用 smem 中 tile[k] 做 mma  →  寄存器写入 smem[(k+1)%2]
```
**验收：ncu `stall_long_scoreboard` 占比显著下降**（这就是"用计算掩盖访存"的直接证据，
报告里贴前后对比）。这一步做完，对照读 LeetCUDA `kernels/hgemm/wmma/hgemm_wmma_stage.cu`
（它是 SM80 cp.async 多 stage 版）——能讲出"我的双缓冲 vs cp.async 多 stage 差在哪"
（不占寄存器/不占指令发射、stage 数可以 >2），就把 SM80 的知识也拿到了。

### Step 4（3-4 天）CUTLASS 路线

1. 先跑通官方例子感受 API 分层：
   `third-party/cutlass/examples/00_basic_gemm`（2.x style 适配 SM75）
2. 用 `cutlass::gemm::device::Gemm` 实例化你的问题：
```cpp
using Gemm = cutlass::gemm::device::Gemm<
    half_t, layout::RowMajor, half_t, layout::RowMajor,
    half_t, layout::RowMajor, float,
    arch::OpClassTensorOp, arch::Sm75,
    gemm::GemmShape<128,128,32>,   // threadblock tile ← 和你 v2 的选择对照!
    gemm::GemmShape<64,64,32>,     // warp tile
    gemm::GemmShape<16,8,8>>;      // SM75 指令 tile
```
3. 改 epilogue：换上 `LinearCombinationRelu`，体会 epilogue fusion（14 章 §3.3）
   不改 mainloop 一行代码。
4. CuTe 概念落地（SM90 mainloop 用不了，但 layout 代数在哪都能跑）：
   读 `kernels/cutlass/cute/` 两个例子 + cutlass/examples/cute/tutorial/，
   用 `print_layout` 打印你 v3 的 swizzled smem layout，
   验证"swizzle 也是一个 Layout 变换"（14 章的话在代码里看见）。

**验收：CUTLASS 版性能 ≥ 你的 v4；能口头回答——CUTLASS 把 GEMM 拆成哪几层
（device/threadblock/warp/instruction + epilogue），你手写代码的每一段对应哪一层。**

### Step 5（1 天）三方对比报告

GFLOPS 曲线（7 条线 × 3 个规模）+ 差距归因。预期格局：
cuBLAS ≈ 100% > CUTLASS ≈ 90-100% > 你的 v4 ≈ 60-85% > v1 ≈ 10-20%。
重点写"我离 CUTLASS 还差什么"（指令级流水、更优 tile 形状搜索、swizzle 质量）。

## 4. 关键能力

1. **Tensor Core 编程模型**：fragment 是寄存器上的分布式 tile、warp 协作语义
2. **GEMM 优化的标准套路链**：分块→去 conflict→流水→向量化，每步对应 ncu 证据
3. **CUTLASS 分层抽象**：能按 threadblock/warp/instruction 三层 tile 讲清模板参数含义
4. **选型论证**（14 章面试题实证）：什么时候手写/CUTLASS/cuBLAS，用自己的数据回答

## 5. 常见坑

- WMMA 要求指针 256-bit 对齐、leading dimension 是 16 的倍数
- half 累加会明显掉精度——必须 float 累加（07 章"FP16→FP32 累加"的亲身验证）
- ncu 下锁频，绝对 GFLOPS 失真：性能数字用 cudaEvent 测，ncu 只看比率指标
- CUTLASS 编译慢且报错离谱：模板参数不合法时先对照 `tools/library` 里已知可用组合

## 6. 扩展方向

- INT8 Tensor Core GEMM（SM75 支持！为 P5 量化项目铺路）
- 不规则形状：M=1（GEMV，decode 的真实形状）→ 体会 tensor core 在小 M 下的浪费
- 用 `cutlass_profiler` 工具扫 tile 形状，理解"为什么没有万能 tile"
- 读 CUTLASS 3.x 的 SM90 example（51_hopper_gett）做"纸上谈兵"——面试讲 Hopper 用

## 7. 参考

- LeetCUDA: `kernels/hgemm/wmma/hgemm_wmma.cu`（先抄它）、`hgemm_wmma_stage.cu`、
  `kernels/swizzle/`、`kernels/hgemm/README.md`（完整优化叙事）
- CUTLASS: `third_party/LeetCUDA/third-party/cutlass/` 的 examples 00/cute tutorial、
  media/docs 下的 `efficient_gemm.md`（GEMM 分层图，必读）
- 博客: 知乎"CUDA 矩阵乘优化"系列（李少侠等）、NVIDIA CUTLASS blog
- notebook: tensor_cores_intro.md、hgemm_optimization.md、14 章 §3
