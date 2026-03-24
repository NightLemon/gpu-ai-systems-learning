# GPU vs CPU：设计哲学的根本差异

> CPU 为延迟优化，GPU 为吞吐优化——理解这一点是理解一切 GPU 编程的起点。

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

深度学习的核心计算是**矩阵乘法（GEMM）**，其特点：

1. **高度并行**：矩阵中每个元素的计算相互独立
2. **计算密集**：算术运算远多于内存访问（高 arithmetic intensity）
3. **规则的访存模式**：不需要复杂的分支预测和乱序执行

这恰好匹配 GPU 的设计：大量简单核心 + 高内存带宽 + 简单控制逻辑。

## 关键细节

### Latency vs Throughput

```
CPU 策略（Latency-oriented）:
- 大 cache 减少内存访问延迟
- 分支预测 + 乱序执行 减少 pipeline stall
- 少量线程，每个线程尽快完成

GPU 策略（Throughput-oriented）:
- 小 cache，但超高带宽的 HBM
- 没有分支预测，遇到 stall 就切换到其他 warp
- 海量线程，通过线程切换隐藏延迟（latency hiding）
```

关键洞察：**GPU 不是通过减少延迟来提高性能，而是通过同时执行大量线程来隐藏延迟。**

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
