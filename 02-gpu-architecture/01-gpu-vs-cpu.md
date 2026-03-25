# GPU vs CPU：设计哲学的根本差异

> CPU 为**延迟**（latency）优化——让单个任务尽快完成；GPU 为**吞吐**（throughput）优化——让大量任务同时推进。理解这一根本差异是理解一切 GPU 编程的起点。

## 为什么需要 GPU？CPU 不够用吗？

如果你是后端工程师，你已经习惯了 CPU 的思维模式：写好代码，操作系统调度到某个核心上执行，单线程跑完算一个请求。CPU 的设计目标就是让**每个任务尽快完成**——所以它有复杂的分支预测、乱序执行、大容量缓存。

但深度学习的工作负载和后端业务完全不同。一次矩阵乘法 `C = A × B`，其中 A 是 4096×4096 的矩阵，需要做 `2 × 4096³ ≈ 1370 亿次`浮点运算。这些运算之间高度独立——`C[i][j]` 的计算和 `C[k][l]` 完全无关。

CPU 能做这个计算，但它的策略是"用 64 个很强的核心，每个核心轮流处理几百万个元素"。GPU 的策略则是"用 16000 个很弱的核心，每个核心只处理几个元素，但所有核心**同时**开工"。

对于这种"海量独立、计算规则"的任务，GPU 的策略碾压 CPU——这就是为什么深度学习需要 GPU。

## 核心概念

### 设计哲学对比

```
CPU: 少量强大的核心，复杂控制逻辑          GPU: 大量简单的核心，简单控制逻辑
┌─────────────────────────────┐           ┌─────────────────────────────┐
│  ┌───────┐  ┌───────┐      │           │ ■■■■■■■■■■■■■■■■■■■■■■■■■ │
│  │ Core  │  │ Core  │      │           │ ■■■■■■■■■■■■■■■■■■■■■■■■■ │
│  │ (强)  │  │ (强)  │      │           │ ■■■■■■■■■■■■■■■■■■■■■■■■■ │
│  └───────┘  └───────┘      │           │ ■■■■■■■■■■■■■■■■■■■■■■■■■ │
│  ┌─────────────────────┐   │           │ ■■■■■■■■■■■■■■■■■■■■■■■■■ │
│  │    大 Cache         │   │           │ ■■■■■■■■■■■■■■■■■■■■■■■■■ │
│  └─────────────────────┘   │           │ ■ = 一个简单核心 (数千个)    │
│  ┌─────────────────────┐   │           │ 小 Cache，简单控制逻辑       │
│  │  复杂控制/预测逻辑    │   │           └─────────────────────────────┘
│  └─────────────────────┘   │
└─────────────────────────────┘
```

### 关键指标对比（以 2024 年旗舰为例）

| 指标 | Intel Xeon w9-3595X (CPU) | NVIDIA H100 (GPU) |
|------|--------------------------|-------------------|
| 核心数 | 60 cores | 132 SM × 128 CUDA Cores = 16,896 |
| 时钟频率 | ~3.0-4.8 GHz | ~1.6-1.8 GHz |
| FP32 峰值算力 | ~5 TFLOPS | ~67 TFLOPS |
| FP16 Tensor 算力 | — | ~1,979 TFLOPS |
| 内存带宽 | ~300 GB/s (DDR5) | ~3,350 GB/s (HBM3) |
| 内存容量 | 数百 GB - TB | 80 GB |
| 单线程性能 | **极强** | 弱 |
| 并行吞吐 | 中等 | **极强** |

### 为什么 GPU 适合深度学习？

深度学习的核心计算是**矩阵乘法**（GEMM，General Matrix Multiply），它具有三个特点：

1. **高度并行**：输出矩阵中每个元素的计算可以独立完成，天然适合大规模并行
2. **计算密集**：算术运算次数（FLOPs）远多于内存访问量（高算术强度 / arithmetic intensity）
3. **规则的访存模式**：数据按固定步长读取，不需要复杂的分支预测和乱序执行

这恰好匹配 GPU 的设计：大量简单核心 + 高内存带宽 + 简单控制逻辑。

## 关键细节

### Latency vs Throughput：两种完全不同的优化目标

这是理解 CPU 和 GPU 差异最核心的一对概念。用一个类比来说：

> **CPU 像一辆跑车**：速度极快，但一次只能坐 2 个人。适合把少数乘客尽快送到目的地。
> **GPU 像一辆公交车**：速度不快，但一次能坐 100 个人。适合把大量乘客一起运走。

```
CPU 策略（Latency-oriented，延迟导向）:
- 大 cache 减少内存访问延迟 → 跑车自带导航，不用等红灯
- 分支预测 + 乱序执行 减少 pipeline stall → 跑车可以超车
- 少量线程，每个线程尽快完成 → 一次只拉一两个人，但秒到

GPU 策略（Throughput-oriented，吞吐导向）:
- 小 cache，但配超高带宽的 HBM → 公交站台很宽，上车速度快
- 没有分支预测，遇到 stall 就切换到其他 warp → 这趟车等红灯了？换另一趟先跑
- 海量线程，通过线程切换隐藏延迟 → 虽然每个人到达慢一点，但同时运了一万人
```

关键洞察：**GPU 不是通过减少延迟来提高性能，而是通过同时执行大量线程来"隐藏"延迟（latency hiding）。** 当某组线程在等内存数据时，GPU 立刻切换到另一组线程继续执行——这个切换是零开销的（硬件层面，不需要上下文切换）。所以 GPU 需要大量线程才能高效工作：线程越多，可切换的"备选项"越多，延迟隐藏越充分。

### Arithmetic Intensity（算术强度）

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

- **Compute-bound**（AI > 硬件的 ops:byte ratio）：性能由算力决定 → GPU 优势巨大
- **Memory-bound**（AI < 硬件的 ops:byte ratio）：性能由带宽决定 → GPU 带宽优势仍有帮助

H100 的 ops:byte ratio ≈ 67 TFLOPS / 3350 GB/s ≈ **20 FLOPs/Byte**

这意味着：如果你的 kernel 每读一个 byte 做不到 20 次浮点运算，它就是 memory-bound 的。

### Roofline Model

```
         性能 (FLOPS)
           │
  算力上限 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
           │                 ╱─────────── Compute-bound
           │               ╱
           │             ╱
           │           ╱  ← 带宽上限斜线
           │         ╱
           │       ╱
           │     ╱  Memory-bound
           │   ╱
           │ ╱
           └──────────────────────── Arithmetic Intensity
```

Roofline Model 帮助你判断一个 kernel 的瓶颈在哪里：
- 在斜线（带宽上限）左侧 → Memory-bound，优化方向是减少内存访问
- 在水平线（算力上限）下方 → Compute-bound，优化方向是提高计算效率

## 代码示例

一个简单的对比实验：CPU vs GPU 矩阵乘法

```python
import torch
import time

N = 4096

# CPU
a_cpu = torch.randn(N, N)
b_cpu = torch.randn(N, N)
start = time.time()
c_cpu = torch.mm(a_cpu, b_cpu)
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

# GPU
a_gpu = torch.randn(N, N, device='cuda')
b_gpu = torch.randn(N, N, device='cuda')
torch.cuda.synchronize()  # 确保数据已传输完毕
start = time.time()
c_gpu = torch.mm(a_gpu, b_gpu)
torch.cuda.synchronize()  # 等待 GPU 计算完成
gpu_time = time.time() - start
print(f"GPU: {gpu_time:.4f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")

# 计算 TFLOPS
flops = 2 * N**3  # 矩阵乘法的 FLOPs = 2*N^3
print(f"GPU TFLOPS: {flops/gpu_time/1e12:.1f}")
```

> 注意 `torch.cuda.synchronize()` 的使用——GPU 操作是异步的，不 sync 就测不准时间。

## 常见问题

**Q: GPU 核心数这么多，为什么不能完全取代 CPU？**

A: GPU 核心非常简单，没有复杂的分支预测和乱序执行能力。对于包含大量条件判断、递归、不规则内存访问的任务（如操作系统、数据库查询、编译器），CPU 的复杂控制逻辑仍然不可替代。GPU 擅长的是 **SPMD（Single Program Multiple Data）** 模式的并行计算。

**Q: 什么是 Tensor Core？和 CUDA Core 有什么区别？**

A: CUDA Core 执行标量浮点运算（每周期 1 个 FMA），Tensor Core 执行小矩阵乘法（如 4×4 FMA），单条指令完成的计算量远大于 CUDA Core。这就是为什么 H100 的 Tensor Core FP16 算力（1979 TFLOPS）远高于 CUDA Core FP32 算力（67 TFLOPS）。现代深度学习几乎完全依赖 Tensor Core。

**Q: HBM 和 GDDR 有什么区别？**

A: HBM（High Bandwidth Memory）通过 3D 堆叠 + 超宽总线（如 HBM3 的 5120-bit）实现极高带宽。GDDR（如 GDDR6X）是传统显存，用在消费级 GPU（如 RTX 4090）。数据中心 GPU（A100/H100）用 HBM，因为训练/推理对带宽要求极高。

## 延伸阅读

- [NVIDIA CUDA C++ Programming Guide - Introduction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)
- [Stanford CS149 - Parallel Computing](https://gfxcourses.stanford.edu/cs149/fall24)
- [Roofline Model 论文](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf) — Williams et al., 2009

---

## 术语表

| 术语 | 说明 |
|------|------|
| **GPU** | Graphics Processing Unit，图形处理器。原本用于图形渲染，现被广泛用于深度学习等并行计算任务 |
| **CUDA Core** | NVIDIA GPU 中执行浮点/整数标量运算的基本计算单元，每个时钟周期完成一次 FMA（乘加）操作 |
| **Tensor Core** | NVIDIA GPU 中专用于矩阵运算的加速单元，单条指令可完成一个小矩阵（如 4×4 或 16×16）的乘加运算，算力远超 CUDA Core |
| **SM（Streaming Multiprocessor）** | GPU 的基本计算模块，每个 SM 包含多个 CUDA Core、Tensor Core、共享内存等，详见下一节 |
| **TFLOPS** | Tera Floating Point Operations Per Second，每秒万亿次浮点运算，衡量算力的单位 |
| **FP32 / FP16 / BF16** | 不同精度的浮点数格式。FP32 = 32 位（标准精度），FP16 = 16 位（半精度），BF16 = 16 位（与 FP32 同范围但精度稍低） |
| **HBM（High Bandwidth Memory）** | 高带宽显存，通过 3D 芯片堆叠和超宽总线实现极高带宽，用于 A100/H100 等数据中心 GPU |
| **GDDR** | Graphics DDR，传统显存技术（如 GDDR6X），用于消费级显卡（如 RTX 4090） |
| **GEMM** | General Matrix Multiply，通用矩阵乘法，深度学习中最核心、最耗时的计算操作 |
| **FMA** | Fused Multiply-Add，融合乘加运算，一条指令完成 a×b+c，是 GPU 计算的基本操作 |
| **Arithmetic Intensity（算术强度）** | 每字节内存访问对应的浮点运算次数，用于判断计算是带宽瓶颈还是算力瓶颈 |
| **Roofline Model（屋顶线模型）** | 将硬件的算力上限和带宽上限画在同一图中，帮助判断 kernel 的性能瓶颈所在 |
| **Latency Hiding（延迟隐藏）** | GPU 的核心策略：当某组线程等待内存数据时，立即切换到另一组线程执行，用线程切换来"隐藏"等待时间 |
| **SPMD** | Single Program Multiple Data，所有线程执行同一段程序但处理不同数据，GPU 编程的基本范式 |
