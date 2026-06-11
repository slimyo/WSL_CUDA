# Triton 06 · 阶梯练习题集（12 题，带验收标准）

> 用法: 每题先自己写，卡 >30 分钟再看提示。验收 = verify 通过 + 达到性能线。
> 性能线按 RTX 2060（336 GB/s, fp16 TC ~26 TFLOPS）标定。
> ★ 标注的题对应真实工业 kernel，可直接写进简历项目。

---

## 第一梯队：elementwise 与广播（练 01 篇）

**E1. fused bias + GeLU** `y = gelu(x + b)`，x:[M,N], b:[N]
- 提示: 2D 偏移 + b 沿行广播；gelu 用 tanh 近似公式自己写
- 验收: vs torch 误差<1e-2；带宽 >250 GB/s（读x+读b+写y ≈ 2MN×2B）
- 对照: LeetCUDA `kernels/gelu/gelu.cu`（CUDA 版要 4 个变体做向量化，你 1 个）

**E2. ★RoPE** 对 q:[seq, heads, dim] 按位置旋转（08 章 §2.2 公式）
- 提示: 每 program 处理一个 (seq_pos, head)；cos/sin 用 tl.cos/tl.sin 现算
- 验收: vs HF transformers 的 `apply_rotary_pos_emb`
- 对照: LeetCUDA `kernels/rope/rope.cu`

**E3. 残差+量化写出** `y_fp8 = clamp(x + res, -448, 448)`（fp16 进、int8 模拟 fp8 出 + scale）
- 练习点: 输出 dtype 转换、per-tensor scale 的两遍法（先 max 后 quant）

## 第二梯队：归约与 norm（练 02 篇）

**E4. ★Fused Add+RMSNorm**（residual add 和 norm 融合——vLLM 真实算子）
- `out, residual_new = rmsnorm(x + residual)`，两个输出
- 验收: 带宽 >250 GB/s；对照 vLLM `csrc/layernorm_kernels.cu` 的融合思路

**E5. Softmax backward** `dx = (dy - rowsum(dy*y)) * y`
- 练习点: 一遍循环同时算 elementwise 乘和行归约

**E6. 大行 online softmax**（02 篇 §3 方案 B 完整版，N=64K）
- 验收: 和两遍法结果一致；耗时 < 两遍法的 70%

## 第三梯队：matmul 家族（练 03 篇）

**E7. ★Matmul + epilogue**（bias + silu + 残差，一个 kernel 出 FFN 的 gate 路径）
- 验收: 4096² 时 ≥ 纯 matmul 性能的 95%（epilogue 应近乎免费——为什么？）

**E8. Batched GEMV**（M=1 专用，不用 tl.dot，用乘加+tl.sum）
- 验收: decode 形状 [1,4096]×[4096,4096] 下 **快于** 你 03 篇的 tl.dot 版
- 这题做完你就理解了"为什么 vLLM 有专门的 GEMV 路径"

**E9. ★W4A16 dequant-GEMM**（P5 项目的 kernel 部分，可在此先做小规模版）
- 验收: 见 P5 Step2/3

## 第四梯队：attention 家族（练 04 篇）

**E10. Flash attention 默写**（04 篇 §4 的第 3 遍：空白文件 30 分钟限时）

**E11. ★merge-attn-states**（P4 的 merge kernel：合并 num_splits 份 (O,m,l)）
- 验收: 和 LeetCUDA `kernels/openai-triton/merge-attn-states/` 输出一致

**E12. Sliding-window causal flash attention**（04 篇扩展 4）
- 验收: window=256 时耗时 ≈ 与 seq 无关（线性复杂度的体感）

---

## 完成后的能力对照表

| 完成度 | 对应水平 |
|------|------|
| E1-E6 | 能胜任日常 fusion 算子需求（torch.compile 不满足时的手动兜底） |
| +E7-E9 | 能写量化/GEMM 变体——P2/P5 项目的 Triton 侧齐活 |
| +E10-E12 | attention 全家桶在手，达到 ROUTE M2/M6 的 🔴L3 标准 |

## 出题人视角（自检是否真会了）

每题做完问自己三个问题——答不上来等于没做：
1. 这个 kernel 的理论访存量是多少？实测带宽利用率多少？差距在哪？
2. BLOCK/num_warps 改一档，性能怎么变？为什么？
3. 如果用 CUDA 写，多写哪些代码？哪部分是 Triton 编译器替你做的？
