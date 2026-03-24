# NVIDIA GPU 架构演进

> 从 Volta 到 Blackwell，理解每一代架构的核心改进及其对 AI 工作负载的影响。

## 核心概念

### Streaming Multiprocessor（SM）— GPU 的基本计算单元

SM 是 GPU 的核心构建模块。一个 GPU 由数十到上百个 SM 组成，每个 SM 包含：

```
┌──────────────────── SM (Streaming Multiprocessor) ────────────────────┐
│                                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Sub-core 0  │  │ Sub-core 1  │  │ Sub-core 2  │  │ Sub-core 3  │ │
│  │ 32 FP32     │  │ 32 FP32     │  │ 32 FP32     │  │ 32 FP32     │ │
│  │ 1 Tensor    │  │ 1 Tensor    │  │ 1 Tensor    │  │ 1 Tensor    │ │
│  │ Core        │  │ Core        │  │ Core        │  │ Core        │ │
│  │ Warp Sched  │  │ Warp Sched  │  │ Warp Sched  │  │ Warp Sched  │ │
│  │ Register    │  │ Register    │  │ Register    │  │ Register    │ │
│  │ File 16K    │  │ File 16K    │  │ File 16K    │  │ File 16K    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │         Shared Memory / L1 Cache (可配置，如 164 KB)            │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

以上以 Hopper (H100) 的 SM 为例，细节因架构代际而异。

### 架构代际演进

```mermaid
timeline
    title NVIDIA GPU 架构演进（AI 视角）
    2017 : Volta (V100)
         : 首次引入 Tensor Core
         : FP16 Tensor: 125 TFLOPS
         : 16 GB / 32 GB HBM2
    2020 : Ampere (A100)
         : 第三代 Tensor Core (TF32, BF16, INT8)
         : Sparsity 支持 (2:4 稀疏)
         : 40 GB / 80 GB HBM2e
         : MIG (Multi-Instance GPU)
    2022 : Hopper (H100)
         : 第四代 Tensor Core (FP8!)
         : Transformer Engine
         : NVLink 4.0 (900 GB/s)
         : 80 GB HBM3
    2024 : Blackwell (B100/B200/GB200)
         : 第五代 Tensor Core (FP4!)
         : 双 die 设计
         : NVLink 5.0 (1800 GB/s)
         : 最高 192 GB HBM3e
```

### 关键架构对比

| 特性 | V100 | A100 | H100 | B200 |
|------|------|------|------|------|
| SM 数量 | 80 | 108 | 132 | 160+ |
| CUDA Cores / SM | 64 | 64 | 128 | 128 |
| Tensor Cores / SM | 8 | 4 (更强) | 4 (更强) | 4 (更强) |
| FP16 Tensor (TFLOPS) | 125 | 312 | 1,979 | ~4,500 |
| FP8 Tensor (TFLOPS) | — | — | 3,958 | ~9,000 |
| HBM 带宽 (TB/s) | 0.9 | 2.0 | 3.35 | 8.0 |
| HBM 容量 | 32 GB | 80 GB | 80 GB | 192 GB |
| NVLink 带宽 (双向) | 300 GB/s | 600 GB/s | 900 GB/s | 1,800 GB/s |
| TDP | 300W | 400W | 700W | 1000W |

> 注意 FP16 Tensor TFLOPS 从 V100→H100 增长了 **~16x**，但 HBM 带宽只增长了 **~3.7x**。这意味着 **memory bandwidth 越来越成为瓶颈**。

## 关键细节

### Tensor Core 的工作方式

Tensor Core 的核心操作是 **WMMA（Warp Matrix Multiply-Accumulate）**：

$$D = A \times B + C$$

其中 A、B、C、D 是小矩阵（如 16×16），一条指令完成整个矩阵乘加：

```
一个 Tensor Core 指令:
  A (16×16, FP16) × B (16×16, FP16) + C (16×16, FP32) → D (16×16, FP32)

vs CUDA Core:
  每条指令只做 a * b + c（标量 FMA）
```

不同精度的 Tensor Core 支持：

| 架构 | 支持的输入精度 |
|------|---------------|
| Volta | FP16 |
| Ampere | FP16, BF16, TF32, INT8, INT4 |
| Hopper | FP16, BF16, TF32, FP8 (E4M3/E5M2), INT8 |
| Blackwell | FP16, BF16, TF32, FP8, FP4, INT8 |

### NVLink & NVSwitch — GPU 间通信

多卡训练时，GPU 之间需要高速通信（如 AllReduce 梯度）。

```
PCIe Gen5:  ~64 GB/s（双向）    ← 太慢
NVLink 4.0: ~900 GB/s（双向）   ← H100，比 PCIe 快 14x
NVSwitch:   全互联拓扑           ← 8 卡间任意两卡 900 GB/s
```

```
8x H100 DGX 系统（NVSwitch 全互联）:

  GPU0 ──── GPU1
  │  ╲    ╱  │
  │   ╲  ╱   │
  │    ╲╱    │
  │    ╱╲    │
  │   ╱  ╲   │
  │  ╱    ╲  │
  GPU2 ──── GPU3
    ...
  每对 GPU 之间都有 NVLink 直连
  总 bisection bandwidth: 900 GB/s × 8 = 7.2 TB/s
```

### MIG（Multi-Instance GPU）— A100+ 独有

MIG 允许将一个物理 GPU 划分为最多 7 个独立的 GPU 实例，每个实例有独立的 SM、显存和带宽。适用于推理服务中多个小模型共享一张卡。

### Transformer Engine — H100+ 独有

H100 引入的 Transformer Engine 能在 FP8 和 FP16 之间**动态切换精度**：

```
Forward Pass:
  每一层的输入/输出统计 → 动态计算 scale factor → 决定使用 FP8 还是 FP16
  
目标: 尽可能用 FP8（速度翻倍）同时保持精度不损失
```

这是 H100 能在 LLM 训练中比 A100 快 3-4x 的核心原因之一。

## 常见问题

**Q: 为什么 Tensor Core 的 TFLOPS 比 CUDA Core 高这么多？**

A: 因为 Tensor Core 做的是矩阵级别的 FMA，一条指令完成 16×16×16 = 4096 次乘加运算。而 CUDA Core 一条指令只做 1 次 FMA。所以同样的时钟周期内，Tensor Core 的"有效 FLOPS"远高于 CUDA Core。

**Q: TF32 是什么？为什么 A100 引入它？**

A: TF32 = TensorFloat-32，是 NVIDIA 发明的一种特殊格式：8-bit 指数（和 FP32 一样）+ 10-bit 尾数（和 FP16 一样）。它的数值范围和 FP32 一样大，但精度和 FP16 接近。Ampere 的 Tensor Core 能以接近 FP16 的速度运算 TF32，且对大多数深度学习任务不影响收敛。PyTorch 默认开启 TF32（`torch.backends.cuda.matmul.allow_tf32 = True`）。

**Q: 选 GPU 时最该关注哪个指标？**

A: 取决于工作负载：
- **训练**：Tensor Core TFLOPS（决定 GEMM 速度） + HBM 带宽（决定数据搬运速度） + NVLink 带宽（决定多卡通信速度）
- **推理（生成式）**：HBM 带宽是首要瓶颈（token 生成是 memory-bound），其次是显存容量（决定能装多大的模型）

**Q: 消费级 GPU（如 RTX 4090）能用于训练吗？**

A: 能，但有限制：
- RTX 4090 有 16,384 CUDA Cores + 第四代 Tensor Core，FP16 算力很强
- 但只有 24 GB GDDR6X（vs H100 的 80 GB HBM3），显存容量和带宽都差距大
- 没有 NVLink（消费级已取消），多卡只能走 PCIe，通信是大瓶颈
- 适合小模型训练/微调和推理实验，不适合大规模分布式训练

## 延伸阅读

- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core) — 官方架构白皮书
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) — B200 架构介绍
- [A100 vs H100 Deep Dive](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) — Tim Dettmers 的 GPU 选购指南
- [Tensor Core 工作原理](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/) — NVIDIA Developer Blog
