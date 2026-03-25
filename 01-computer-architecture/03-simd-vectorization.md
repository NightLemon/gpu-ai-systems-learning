# SIMD 向量化

> CPU 上的并行计算能力。理解 SIMD 有助于对比 GPU 的 SIMT 模型，也能帮助你理解为什么 GPU 更适合大规模并行计算。

## 这一节在讲什么？为什么要学？

你马上要学 GPU 的 SIMT（Warp 执行模型）——但在那之前，先理解 CPU 上的 SIMD 是怎么回事，会让你更清楚地看到两种并行方式的本质区别和各自的优劣。

一句话总结：**SIMD 和 SIMT 都是"让硬件同时处理多个数据"的方式，但 CPU 的 SIMD 像是"一个工人同时搬 16 块砖"（一条指令处理一个向量），GPU 的 SIMT 像是"32 个工人同时各搬一块砖"（32 个线程同时执行同一条指令）。**

这个区别直接决定了：
- CPU 擅长少量数据的低延迟处理（一个核心很强，但核心少）
- GPU 擅长大量数据的高吞吐计算（每个线程很弱，但线程极多）

## 核心概念

### SIMD vs SIMT：一字之差，模式大不同

这两个缩写容易混淆，但含义不同：

- **SIMD**（Single Instruction, Multiple Data，单指令多数据）：CPU 上的向量化技术。一条指令同时处理多个数据元素，向量宽度由硬件固定（如 AVX-512 一次处理 16 个 float）。程序员需要通过特殊的向量指令（intrinsics）或依赖编译器来使用这一能力。
- **SIMT**（Single Instruction, Multiple Threads，单指令多线程）：GPU 上的执行模型。32 个线程组成一个 warp，同时执行同一条指令。程序员只需要写针对单个线程的标量代码，硬件自动让一个 warp 中的 32 个线程并行执行。

```
SIMD (CPU - 以 AVX-512 为例):
  一条指令处理 512 bit / 32 bit = 16 个 float
  需要程序员显式使用向量指令，或依赖编译器自动向量化
  向量宽度固定（由指令集决定）

SIMT (GPU - CUDA):
  一个 warp = 32 个线程同步执行同一条指令
  程序员写的是单线程的标量代码，硬件自动将 32 个线程并行化
  比 SIMD 更灵活（支持线程间走不同分支，虽然有性能代价）
```

### CPU 向量化示例

以下代码使用 Intel AVX-512 intrinsics 实现了一个向量加法，每次循环同时计算 16 个 float 的加法：

```c
#include <immintrin.h>  // AVX-512 intrinsics 头文件

void vec_add_avx512(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i += 16) {              // 每次跳 16 个 float
        __m512 va = _mm512_load_ps(a + i);          // 一次加载 16 个 float
        __m512 vb = _mm512_load_ps(b + i);          // 一次加载 16 个 float
        __m512 vc = _mm512_add_ps(va, vb);          // 16 个加法同时完成
        _mm512_store_ps(c + i, vc);                 // 一次写回 16 个 float
    }
}
```

### 自动向量化

手写 intrinsics 门槛高、可读性差。好消息是现代编译器可以在满足条件时自动把标量循环转为向量化代码——你写普通的 for 循环，编译器帮你换成 SIMD 指令：

```bash
# GCC 开启自动向量化并输出报告
gcc -O3 -march=native -ftree-vectorize -fopt-info-vec my_code.c
# -fopt-info-vec 会告诉你哪些循环被成功向量化了，哪些没有（以及为什么）
```

编译器能自动向量化的前提：循环迭代之间无数据依赖、数据地址对齐（或编译器能判断对齐）、循环体足够简单。这些条件本质上和 GPU 编程中"让 warp 中的线程做同样的事"是一个道理——并行的前提是各路计算互不干扰。

### 从 SIMD 到 GPU：并行规模的飞跃

理解了 CPU SIMD，你就能更清楚地看到 GPU 的优势和代价：

| 维度 | CPU SIMD (AVX-512) | GPU SIMT (CUDA) |
|------|-------------------|-----------------|
| **并行宽度** | 一次 16 个 float | 一个 warp 32 线程，GPU 上可同时调度数千个 warp |
| **编程方式** | 手写 intrinsics 或依赖编译器 | 写单线程标量代码，硬件自动并行 |
| **分支处理** | 用 mask 指令选择性计算（灵活但复杂） | Warp Divergence：分支导致串行（简单但有代价） |
| **适合场景** | 少量数据、低延迟（数据库、网络包处理） | 大量数据、高吞吐（矩阵乘法、模型推理） |

**核心差异**：CPU SIMD 是在一个很强的核心上"拓宽每条指令的处理宽度"，GPU SIMT 是用大量简单核心"堆线程数量"。两者都是数据级并行，但实现路径完全不同。

**带到 GPU 章节的关键认知**：当你在 Ch02 看到 GPU 的 Warp 概念时，可以把它理解为"SIMD 的极端扩展版"——向量宽度从 16 扩展到 32，然后通过调度数千个 warp 实现百万级线程并行。

## 延伸阅读

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **SIMD** | Single Instruction, Multiple Data，CPU 上的向量化指令技术，一条指令同时处理多个数据 |
| **SIMT** | Single Instruction, Multiple Threads，GPU 的执行模型，一个 warp（32 线程）同步执行同一指令 |
| **AVX-512** | Intel 的 512 位 SIMD 指令集扩展，一次可处理 16 个 32 位浮点数 |
| **Intrinsics** | 编译器提供的内置函数，直接映射到特定的 CPU 向量指令，介于汇编和 C 代码之间 |
| **Warp** | GPU 中 32 个线程的执行组，是 GPU 调度和执行的基本单位 |
| **向量化** | 将标量循环（一次处理一个元素）转换为向量操作（一次处理多个元素）的优化过程 |
| **Divergence（分支发散）** | 同一 warp 中的线程走了不同的条件分支，导致部分线程空等，降低执行效率 |
