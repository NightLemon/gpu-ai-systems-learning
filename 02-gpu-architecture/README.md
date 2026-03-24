# 02 - GPU 架构

> 理解 GPU 硬件是所有 CUDA 编程和模型优化的基础。

## 本章内容

| 文件 | 主题 | 要点 |
|------|------|------|
| [01-gpu-vs-cpu.md](01-gpu-vs-cpu.md) | GPU vs CPU | 设计哲学差异、适用场景、吞吐 vs 延迟 |
| [02-nvidia-gpu-architecture.md](02-nvidia-gpu-architecture.md) | NVIDIA GPU 架构 | SM 结构、代际演进（Volta→Ampere→Hopper→Blackwell） |
| [03-memory-hierarchy.md](03-memory-hierarchy.md) | 显存层级 | Register → Shared → L2 → HBM，带宽与延迟 |
| [04-execution-model.md](04-execution-model.md) | 执行模型 | Grid/Block/Thread/Warp、调度与 Occupancy |

## 前置知识

- 了解 CPU 的 cache 层级和流水线概念
- 知道什么是 SIMD（Single Instruction Multiple Data）

## 学完本章你能回答

1. 为什么 GPU 的峰值算力比 CPU 高 10-100 倍，但单线程性能远不如 CPU？
2. 一个 A100 GPU 有多少个 SM？每个 SM 有多少 CUDA Core 和 Tensor Core？
3. Shared Memory 和 L1 Cache 是什么关系？
4. 什么是 Warp？为什么 Warp Divergence 会导致性能下降？
