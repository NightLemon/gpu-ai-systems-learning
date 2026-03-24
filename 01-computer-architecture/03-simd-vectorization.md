# SIMD 向量化

> CPU 上的并行计算能力。理解 SIMD 有助于对比 GPU 的 SIMT 模型，也能帮助你理解为什么 GPU 更适合大规模并行计算。

## 核心概念

### SIMD vs SIMT

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

不想手写 intrinsics？现代编译器可以在满足条件时自动将标量循环转为向量化代码：

```bash
# GCC 开启自动向量化并输出报告
gcc -O3 -march=native -ftree-vectorize -fopt-info-vec my_code.c
# -fopt-info-vec 会打印哪些循环被成功向量化了
```

编译器能自动向量化的前提：循环迭代之间无数据依赖、数据地址对齐（或编译器能判断对齐）、循环体足够简单。

### 与 GPU 的联系

理解 CPU SIMD 有助于理解为什么 GPU 更适合大规模并行任务：

- **并行宽度**：CPU SIMD 一次处理 16 个 float（AVX-512），而 GPU 可以同时调度数千个 warp（每个 warp 32 线程）——并行规模差两个数量级。
- **编程模型**：CPU SIMD 需要手动使用向量指令或依赖编译器，GPU 的 SIMT 模型让程序员写标量代码即可自动并行。
- **适用场景**：CPU 适合少量数据的低延迟处理（如分支密集的逻辑），GPU 适合大量数据的高吞吐计算（如矩阵乘法）。

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
