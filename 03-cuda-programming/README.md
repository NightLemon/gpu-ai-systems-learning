# 03 - CUDA 编程

> 从第一个 kernel 到 GEMM 优化实战，系统掌握 GPU 编程。

## 本章内容

| 文件 | 主题 | 要点 |
|------|------|------|
| [01-cuda-basics.md](01-cuda-basics.md) | CUDA 基础 | 编程模型、编译、kernel 启动、错误处理 |
| [02-memory-management.md](02-memory-management.md) | 内存管理 | cudaMalloc/cudaFree、Unified Memory、异步拷贝 |
| [03-shared-memory.md](03-shared-memory.md) | Shared Memory | Bank conflict、Tiling、同步原语 |
| [04-optimization-techniques.md](04-optimization-techniques.md) | 优化技巧 | Occupancy、Coalescing、Divergence、Stream |
| [05-gemm-case-study.md](05-gemm-case-study.md) | GEMM 优化实战 | 从朴素到高性能：tiling、向量化、Tensor Core |
| [06-triton-intro.md](06-triton-intro.md) | Triton 入门 | 用 Python 写 GPU kernel 的新范式 |

## 环境准备

```bash
# 检查 CUDA 环境
nvcc --version
nvidia-smi

# 编译 CUDA 程序
nvcc -o mykernel mykernel.cu -O2

# 推荐开发环境
# - CUDA Toolkit 12.x
# - VSCode + CUDA C++ 插件
# - Nsight Systems / Nsight Compute（Profiling）
```

## 学完本章你能回答

1. 如何确定 kernel 的 gridDim 和 blockDim？
2. `cudaMemcpy` 和 `cudaMemcpyAsync` 有什么区别？
3. 如何避免 Shared Memory 的 bank conflict？
4. 什么是 memory coalescing？为什么它对性能至关重要？
5. 如何用 Nsight Compute 分析一个 kernel 的性能瓶颈？
6. GEMM 优化中 tiling 的核心思想是什么？
