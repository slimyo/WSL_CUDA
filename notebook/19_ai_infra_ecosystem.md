# AI Infra 生态全景：术语地图、工业分层与你的定位

> 对象: 准备推理/算子岗，需要"全局视角"——每个名词在体系里的位置、谁在用、和你什么关系
> 用法: 本章是【地图】不是教材。每个术语给"是什么/在哪层/工业怎么用/你要掌握到什么程度"。
> 必须 L3 的（OpenAI Triton）有专门教程目录 [triton/](triton/00_README.md)；CUTLASS 见 [projects/P2](projects/P2_hgemm_cutlass.md)。
> 能力分级沿用 ROUTE.md：🟢L1 知道是什么 / 🟡L2 讲清原理与 trade-off / 🔴L3 会写会改

---

## 0. 一张全景分层图（先把架子立起来）

```
┌─ L8 业务/产品 ──────────────── ChatBot、Copilot、Agent 平台
├─ L7 集群编排与调度 ─────────── K8s、Slurm、Ray、Volcano/Kueue、NVIDIA Dynamo
├─ L6 推理服务层 ─────────────── Triton Inference Server(NVIDIA)、KServe、网关/路由
├─ L5 推理引擎 ───────────────── vLLM、SGLang、TensorRT-LLM、LMDeploy、llama.cpp
│      训练框架(平行格子)──────── Megatron-LM、DeepSpeed、FSDP、verl(RL)
├─ L4 深度学习框架 ───────────── PyTorch、JAX；图编译: torch.compile、XLA
├─ L3 Kernel 生成与编译 ──────── OpenAI Triton、TVM、MLIR、TorchInductor  ←┐
├─ L2 高性能库 ───────────────── cuBLAS、cuDNN、CUTLASS、FlashAttention/    │ LLVM
│                                FlashInfer、NCCL、CUB/Thrust              ←┘ 是底座
├─ L1 编程模型与系统软件 ─────── CUDA C++、PTX/SASS、Driver；ROCm(AMD)、CANN(昇腾)
└─ L0 硬件 ───────────────────── GPU/TPU/NPU、NVLink/NVSwitch、InfiniBand/RoCE、HBM
```

**全局规律（背下来，胜过背 100 个名词）：**
1. **越往下越通用、越往上越贴业务**；每层只跟相邻层打交道（K8s 不关心 PTX）
2. 同一层的东西是**竞争/替代关系**（vLLM vs SGLang），跨层的是**依赖关系**（vLLM 调 FlashInfer）
3. 你的岗位（推理/算子）核心在 **L2-L5**，向下要懂 L0-L1（性能从哪来），向上知道 L6-L7（你的 kernel 被谁调用）

---

## 1. L0-L1 硬件与系统软件

| 术语 | 是什么 | 工业视角 | 等级 |
|------|------|------|:---:|
| CUDA | NVIDIA 的 GPU 编程模型+工具链（语言扩展、runtime、driver API） | 整个生态的事实标准；"CUDA 护城河"指 L1-L2 二十年的库积累 | 🔴 |
| PTX / SASS | PTX=虚拟指令集（前向兼容），SASS=真机器码（每代不同） | 看 kernel 真正生成的代码：`nvcc -ptx` / `cuobjdump -sass`；极限优化对着 SASS 调 | 🟡 |
| Driver / nvidia-smi | 内核驱动 + 管理工具 | MIG 切分、锁频、ECC；容器里要装 nvidia-container-toolkit | 🟢 |
| ROCm / HIP | AMD 的 CUDA 对位（HIP 是兼容层，hipify 转换 CUDA 代码） | MI300X 上量后大厂开始要求"CUDA/ROCm 双栈"；vLLM/Triton 都有 ROCm 后端 | 🟢 |
| CANN / 昇腾 | 华为 NPU 的软件栈（对位 CUDA），AscendC 写算子，MindIE 推理 | 国内大厂真实需求：vLLM-Ascend 适配岗位多 | 🟢 |
| NVLink / NVSwitch | GPU 间高速互联（H100 900GB/s）/ 多卡全互联交换机 | 决定 TP 域边界（16 章）；GB200 NVL72 把域扩到 72 卡 | 🟡 |
| InfiniBand / RoCE / RDMA | 节点间网络：IB 是专用网络，RoCE 是以太网上跑 RDMA；RDMA=绕过 CPU 直接读写远端内存 | 跨节点 EP/PP/KV 传输的物理底座；NCCL 自动用它 | 🟡 |
| GPUDirect | GPU 显存直通技术族：P2P(卡间)、RDMA(网卡直读显存)、Storage(SSD直读) | P/D 分离 KV 传输、checkpoint 加载提速的关键 | 🟢 |

## 2. L2 高性能库（你的"竞争对手"和"弹药库"）

| 术语 | 是什么 | 工业视角 | 等级 |
|------|------|------|:---:|
| cuBLAS / cuBLASLt | NVIDIA 官方 BLAS（GEMM 等）；Lt 版支持融合 epilogue、量化 | 你手写 GEMM 的对照上限；框架的 matmul 默认走它 | 🟡 |
| cuDNN | 卷积/attention 等 DNN 原语库 | CV 时代的核心；LLM 时代地位让位给 FA/自研 kernel | 🟢 |
| CUTLASS / CuTe | 开源 GEMM 模板库 / layout 代数（14 章） | 自研 kernel 的标准起点；FA3/FA4、vLLM 大量 kernel 基于它 | 🔴(P2) |
| FlashAttention (库) | Tri Dao 的官方 attention 库（flash-attn 包） | 训练侧标配；推理侧被 FlashInfer/自研逐步替代 | 🔴(P3) |
| FlashInfer | 推理专用 attention/采样 kernel 库（paged KV、split-KV、MLA） | vLLM/SGLang 的 attention 后端，MLSys'25 最佳论文 | 🟡 |
| NCCL | GPU 集合通信库（all-reduce/all-to-all...），拓扑感知 | 一切多卡训练/推理的通信底座；调试集群先看 NCCL 环境变量 | 🟡 |
| CUB / Thrust | block/device 级原语（reduce/scan/sort）/ STL 风格容器算法 | 写自定义 kernel 时别重造 reduce 轮子 | 🟢 |

## 3. L3 Kernel 生成与编译器（含命名大坑）

### ⚠️ 命名消歧（面试/聊天第一坑）

```
OpenAI Triton  = Python 风格的 GPU kernel DSL/编译器（写算子用）   ← 你要 L3 的
NVIDIA Triton  = Triton Inference Server，模型服务框架（部署用）   ← L6 层，完全不同的东西!
平时说 "Triton 写 kernel" 指前者；说 "Triton 部署/ensemble" 指后者。
```

| 术语 | 是什么 | 工业视角 | 等级 |
|------|------|------|:---:|
| **OpenAI Triton** | tile 级 kernel DSL：写 block 逻辑，编译器管 thread/smem/流水（14 章） | OpenAI/Meta/字节等内部大量使用；TorchInductor 的 GPU 后端就是它 → **见 [triton/ 教程](triton/00_README.md)** | 🔴 |
| LLVM | 通用编译器基础设施：IR + 优化 pass + 各架构后端 | **整个 L3 的地基**：nvcc 前端基于 LLVM(clang)、Triton 经 LLVM 生成 PTX、TVM/XLA 的 CPU 后端、MLIR 是其子项目。不用会写，要懂"IR+pass"思想 | 🟢 |
| MLIR | LLVM 旗下"多层 IR"框架：可自定义方言(dialect)逐层 lower | 新一代 ML 编译器全在用：Triton(ttir/ttgir 就是 MLIR 方言)、IREE、各家 NPU 编译器 | 🟢 |
| TVM | Apache 端到端 ML 编译器：图优化 + 自动调度(Ansor)生成算子 | 跨硬件部署（手机/NPU/边缘）主力；LLM 时代热度下降，MLC-LLM 是其 LLM 应用 | 🟢 |
| XLA | Google 的线性代数编译器（JAX/TF 后端，TPU 原生） | TPU 生态核心；GPU 上 OpenXLA 社区维护 | 🟢 |
| torch.compile / TorchInductor / Dynamo | PyTorch 2.x 编译栈：Dynamo 抓图 → Inductor 生成 Triton/C++ kernel | "写 PyTorch 自动得到融合 kernel"；推理引擎也用它做非核心算子 | 🟡 |
| ONNX / ONNX Runtime | 模型交换格式 / 跨平台推理运行时 | 传统部署管线"训练→ONNX→TensorRT/ORT"；LLM 时代引擎直接吃 HF 权重，ONNX 边缘化 | 🟢 |

**这一层的历史脉络（"AI 编译器"叙事，面试聊起来加分）：**
```
2017-2021 图编译时代: TVM/XLA 想"编译整个模型图"——通用但天花板低
2021-2023 DSL 转向:   Triton 证明"人写 tile 逻辑+编译器管细节"更实用
2023-2026 分工定型:   核心算子(attention/GEMM) = 专家手写(CUTLASS/Triton)
                      长尾算子+融合 = torch.compile 自动生成
                      跨硬件 = MLIR 系各自为战
教训: 编译器没有消灭 kernel 工程师，反而抬高了"会写关键 kernel"的溢价。
```

## 4. L4-L5 框架与引擎

### 训练侧（推理岗需 L1-L2，理解概念即可）

| 术语 | 是什么 | 关键概念 | 等级 |
|------|------|------|:---:|
| PyTorch | 事实标准框架 | eager/compile 两种模式、autograd、ATen 算子注册（你写的 kernel 怎么接进去） | 🟡 |
| JAX | Google 函数式框架 | jit/vmap/pmap、XLA 编译；TPU 生态主力 | 🟢 |
| Megatron-LM | NVIDIA 大模型训练框架 | TP/PP/SP 的发源地（16 章），3D 并行 | 🟡 |
| **DeepSpeed** | 微软训练优化库 | **ZeRO 1/2/3**：把 optimizer state/梯度/参数分片到各卡省显存；ZeRO-Offload 卸到 CPU。推理侧 DeepSpeed-Inference 已被 vLLM 系取代 | 🟡 |
| FSDP | PyTorch 原生的 ZeRO-3 等价物 | 中小规模训练默认选择 | 🟢 |
| verl / OpenRLHF | RL 后训练框架 | 内嵌 vLLM/SGLang 做 rollout（18 章 §1.3）——推理引擎的第二客户 | 🟢 |

### 推理引擎（你的主战场，全部 🟡 以上）

| 引擎 | 定位与差异化 | 工业现状 (2025-26) |
|------|------|------|
| **vLLM** | 学界出身、社区最大；PagedAttention 发源地；V1 引擎重构 | 开源 serving 事实标准，国内外大厂皆用/魔改 |
| **SGLang** | RadixAttention 前缀缓存 + overlap 调度 + 前端结构化 DSL | 大规模 P/D+EP 部署先行者（DeepSeek 官方推荐过）；与 vLLM 双雄 |
| TensorRT-LLM | NVIDIA 官方，TensorRT 的 LLM 特化：手调 kernel + engine 编译 | N 卡极限性能；闭源部分多、灵活性差；FP8/FP4 支持最快 |
| LMDeploy | 上海 AI Lab（InternLM 系），TurboMind 引擎 | 国内常见备选；TokenAttention（12 章） |
| llama.cpp / GGUF | 纯 C++ 端侧推理 + 自有量化格式 | 端侧/个人部署王者；ollama 是它的易用封装 |
| HF transformers / TGI | 模型库的参考实现 / HF 的 serving | transformers 是"正确性基准"，性能不是它的目标 |

**怎么记：所有引擎都在卷同三件事——①attention kernel 与 KV 管理（M2/M4）
②调度（M5）③量化与并行（M7/M8）。** 你学的理论就是它们的差异化维度。

## 5. L6-L7 服务与集群编排

| 术语 | 是什么 | 工业视角 | 等级 |
|------|------|------|:---:|
| **NVIDIA Triton Inference Server** | 多框架模型服务器：HTTP/gRPC、动态 batch、多模型 ensemble、(vLLM/TRT-LLM 可作为其 backend) | 传统 ML/CV 部署标配；LLM 场景常被 vLLM 自带 server 替代 | 🟢 |
| **K8s (Kubernetes)** | 容器编排系统：声明式管理"什么程序跑在哪台机器" | AI 用法：GPU device plugin 暴露卡、topology-aware 调度（同 NVLink 域优先）、**gang scheduling**（分布式任务要么全启动要么全不启,Volcano/Kueue 解决）、弹性扩缩容 | 🟢 |
| Slurm | HPC 作业调度器（超算传统） | 训练集群主流仍是 Slurm；"K8s 管推理、Slurm 管训练"是常见格局 | 🟢 |
| Ray | Python 分布式计算框架（actor/task 模型） | vLLM 多节点用 Ray 起 worker；verl 用 Ray 编排 RL 角色 | 🟢 |
| KServe / Dynamo | K8s 上的模型服务 CRD / NVIDIA 的 LLM 编排层（13/18 章） | Dynamo: P/D 分离路由 + KV 感知调度的开源参照 | 🟢 |
| Prometheus / Grafana | 指标采集/可视化 | SLO 监控（TTFT/TPOT P99 看板）的标准件 | 🟢 |

## 6. 一个 token 的生命周期（把所有层串起来）

**面试被问"讲讲你理解的推理全链路"，照这个讲：**

```
用户发请求 "讲个笑话"
 ↓ L7  K8s ingress → 推理网关（鉴权/限流）→ 路由器（前缀亲和/PD-aware, 13章）
 ↓ L5  vLLM 实例收到请求:
        tokenizer 编码 → 调度器把它加入下一个 iteration batch（continuous batching）
        BlockManager 分配 KV pages（PagedAttention, 12章）
 ↓ L4  模型 forward: PyTorch 层面逐层调用算子
 ↓ L3/L2 每个算子分发到具体 kernel:
        attention → FlashInfer (CUTLASS/手写)   GEMM → cuBLAS
        RMSNorm/RoPE → torch.compile 生成的 Triton kernel
        多卡 → NCCL all-reduce (经 NVLink)
 ↓ L1  所有 kernel 编译为 PTX → JIT 成 SASS → 由 driver 提交 GPU
 ↓ L0  SM 上 warp 调度执行：tensor core 算 GEMM、HBM 搬权重和 KV（06章的 roofline 在这里发生!）
 ↑     logits → 采样出 token → detokenize → SSE 流式返回用户
       同时: 调度器记录指标 → Prometheus → Grafana 的 P99 TPOT 看板（SLO）
```

**你的岗位 = L2-L5 这段的性能负责人**；P1-P5 项目就是在这段的不同点位上动手。

## 7. 必学清单与学习入口

| 优先级 | 内容 | 入口 |
|:---:|------|------|
| 🔴 必须 L3 | **OpenAI Triton**（现场手写概率最高的工具） | **[triton/ 完整教程](triton/00_README.md)**（本次新建，6 篇） |
| 🔴 必须 L3 | CUDA + profiling + CUTLASS 入门 | [projects/P1](projects/P1_profiling_gym.md), [P2](projects/P2_hgemm_cutlass.md) |
| 🟡 L2 | vLLM/SGLang 内部机制 | 12/13 章 + 各自 docs 的 architecture 页 + 读 scheduler 源码 |
| 🟡 L2 | torch.compile 行为 | 官方 tutorial: `TORCH_LOGS=output_code` 看它生成的 Triton 代码（学完 triton/ 教程后看，全能看懂） |
| 🟢 L1（7 天速览路线） | TensorRT-LLM: 跑一次官方 quickstart，理解 engine 构建流程 | trtllm docs |
| 🟢 L1 | TVM: 读 "How TVM works" 概览 + MLC-LLM README，理解 auto-scheduling 思想 | tvm.apache.org |
| 🟢 L1 | K8s: 本机 minikube 起一个带 GPU 的 pod（或读 NVIDIA device plugin README） | k8s docs |
| 🟢 L1 | DeepSpeed: 只精读 ZeRO 论文的图 1（三阶段分片），能复述显存账即可 | ZeRO paper |

## 8. 自测 / 面试题

1. OpenAI Triton 和 NVIDIA Triton Inference Server 分别是什么？在栈的哪层？
2. torch.compile、TVM、手写 CUTLASS 三者的适用边界？（用 §3 的历史脉络回答）
3. 讲一个 token 从 HTTP 请求到 tensor core 的完整路径，标出每层的名字。
4. ZeRO-3 和 TP 都是"参数分片"，本质区别是什么？（提示：计算时是否 gather）
5. K8s 调度分布式推理任务比调度无状态 Web 服务难在哪？（gang/拓扑/显存非弹性）
6. 为什么 LLM 时代 ONNX/cuDNN 地位下降而 CUTLASS/Triton 上升？
