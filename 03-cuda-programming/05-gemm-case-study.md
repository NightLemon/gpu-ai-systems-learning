# GEMM 优化实战

> 矩阵乘法（GEMM，General Matrix Multiply）是深度学习中最核心的计算操作，也是衡量 GPU 优化能力的试金石。全连接层、注意力机制、卷积层的底层计算本质上都是 GEMM。

## 核心概念

### 为什么关注 GEMM？

深度学习中，以下操作本质都是 GEMM：
- **全连接层**：$Y = XW + b$
- **注意力机制**：$QK^T$、$\text{softmax}(QK^T/\sqrt{d})V$
- **卷积**：通过 im2col 转化为 GEMM

大模型训练和推理 **70-80% 的时间花在 GEMM 上**。理解 GEMM 优化就是理解 GPU 性能优化的核心。

### GEMM 定义

$$C_{M \times N} = A_{M \times K} \times B_{K \times N}$$

- 总计算量：$2MNK$ FLOPs（每个元素需要 $K$ 次乘法 + $K-1$ 次加法 ≈ $2K$ FLOPs）
- 总数据量：$(MK + KN + MN) \times \text{sizeof(dtype)}$ bytes

### 优化层级概览

```
版本          性能 (相对 cuBLAS)    优化手段
─────────────────────────────────────────────────────
V0 朴素       ~1-2%               直接三重循环
V1 Tiling     ~10-15%             Shared Memory Tiling
V2 2D Tiling  ~25-35%             每线程多元素 + Register Tiling
V3 向量化     ~45-55%             float4 加载 + 双缓冲
V4 Warp Tiling~60-70%             Warp 级 WMMA / MMA
V5 Tensor Core~80-90%             WMMA / CUTLASS
cuBLAS        100%                高度优化的闭源实现
```

## 各版本代码与分析

### V0: 朴素实现

```cuda
// 每个线程计算 C 的一个元素
__global__ void sgemm_naive(float *A, float *B, float *C, 
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
            //     ^ coalesced      ^ strided (bad!)
            //                        不同线程(不同row)访问 B 的不同行
        }
        C[row * N + col] = sum;
    }
}
```

**分析**：
- 每个线程从 Global Memory 读 $2K$ 个 float → 巨大的带宽浪费
- `B[k * N + col]` 的访问在 K 维度上是列方向，对 B 的 cache 利用率极低
- 典型性能：~1-2% cuBLAS

### V1: Shared Memory Tiling

```cuda
#define BM 32
#define BN 32
#define BK 32

__global__ void sgemm_tiled(float *A, float *B, float *C, 
                            int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    float sum = 0.0f;
    
    for (int tile = 0; tile < K; tile += BK) {
        // 协作加载 A 和 B 的 tile 到 shared memory
        As[threadIdx.y][threadIdx.x] = 
            (row < M && tile + threadIdx.x < K) ? 
            A[row * K + tile + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = 
            (tile + threadIdx.y < K && col < N) ? 
            B[(tile + threadIdx.y) * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

**分析**：
- 每个 tile 从 Global Memory 读一次，被 32×32=1024 个线程共享
- Global Memory 读取量降低了约 BK 倍
- 问题：每线程只计算 1 个输出元素，计算/访存比仍然不够高

### V2: Register Tiling（每线程多元素）

核心思想：**每个线程计算 C 的一个小块（如 8×8），而不是一个元素**

```
Block (128×128) 的结构:
┌────────────────────┐
│ T(0,0)  T(0,1) ... │ ← 每个 T 是一个线程负责的 8×8 小块
│ T(1,0)  T(1,1) ... │
│  ...     ...       │
│ T(15,0) T(15,1) .. │ ← 16×16 个线程 = 256 threads/block
└────────────────────┘

每个线程:
  - 持有 8×8 = 64 个 register 做累加（register tiling）
  - 从 shared memory 加载 A 的 8 个元素 + B 的 8 个元素
  - 做 8×8 = 64 次 FMA
  
计算/访存比: 64 FMA / 16 loads = 4 FMA/load
（远好于 V1 的 1 FMA / 2 loads）
```

```cuda
// 简化伪代码
#define BM 128
#define BN 128
#define BK 8
#define TM 8   // 每线程在 M 方向处理 8 行
#define TN 8   // 每线程在 N 方向处理 8 列

__global__ void sgemm_register_tiling(float *A, float *B, float *C,
                                       int M, int N, int K) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    
    // 每线程的累加寄存器
    float accum[TM][TN] = {0.0f};
    float a_reg[TM], b_reg[TN];
    
    for (int tile = 0; tile < K; tile += BK) {
        // 协作加载 A, B tile 到 shared memory
        // ... (略)
        __syncthreads();
        
        for (int k = 0; k < BK; k++) {
            // 加载到 register
            #pragma unroll
            for (int m = 0; m < TM; m++)
                a_reg[m] = As[k][threadIdx.y * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; n++)
                b_reg[n] = Bs[k][threadIdx.x * TN + n];
            
            // Register 里做外积
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    accum[m][n] += a_reg[m] * b_reg[n];
        }
        __syncthreads();
    }
    
    // 写回 C
    // ... (略)
}
```

### V3: 双缓冲（Prefetch + Overlap）

```
目标：隐藏 Global → Shared Memory 的加载延迟

普通流程:
  [Load tile 0] → [Compute tile 0] → [Load tile 1] → [Compute tile 1] → ...
  
双缓冲 (Ping-Pong):
  [Load tile 0 → buf A]
  [Load tile 1 → buf B] + [Compute tile 0 from buf A]
  [Load tile 2 → buf A] + [Compute tile 1 from buf B]
  ...

→ 加载和计算重叠，流水线化
```

### V4/V5: 使用 Tensor Core

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// Tensor Core 操作以 warp 为单位
// 每个 warp 计算 16×16×16 的矩阵乘加
__global__ void sgemm_wmma(half *A, half *B, float *C, 
                            int M, int N, int K) {
    // 声明 fragment
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K; k += 16) {
        // 从 global memory 加载到 fragment
        load_matrix_sync(a_frag, A + row * K + k, K);
        load_matrix_sync(b_frag, B + k * N + col, N);
        
        // Tensor Core MMA
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 写回
    store_matrix_sync(C + row * N + col, c_frag, N, mem_row_major);
}
```

> 实际生产中建议使用 **CUTLASS** 库而非手写 WMMA，它提供了高度模板化的 GEMM 实现。

## 关键细节

### 性能分析框架

对于 M=N=K=4096 的 FP32 GEMM：

```
理论计算量: 2 × 4096³ = 137.4 GFLOP
H100 FP32 峰值: 67 TFLOPS

理论最短时间: 137.4G / 67T = 2.05 ms
cuBLAS 实际: ~2.3 ms (达到峰值 ~89%)

数据量: (4096² × 3) × 4 = 192 MB
HBM 带宽: 3.35 TB/s → 最短传时间: 0.057 ms
Arithmetic Intensity: 137.4G / 192M = 716 FLOPs/Byte >> 20

→ 这个问题是 Compute-bound（好消息：GEMM 天然适合 GPU）
```

### 不同精度的 GEMM 性能

| 精度 | H100 峰值 | 典型应用 |
|------|----------|---------|
| FP32 | 67 TFLOPS | Legacy |
| TF32 | 989 TFLOPS | 默认训练精度（PyTorch 自动使用） |
| FP16/BF16 | 1,979 TFLOPS | 混合精度训练 |
| FP8 | 3,958 TFLOPS | Hopper+ 训练/推理 |
| INT8 | 3,958 TOPS | 推理量化 |

## 常见问题

**Q: 为什么不直接用 cuBLAS，还要学 GEMM 优化？**

A: 三个原因：
1. **理解底层**：GEMM 是理解 GPU 性能优化的最佳教材
2. **融合算子**：实际中需要将 GEMM 和其他操作融合（如 GEMM+Bias+ReLU），这时需要修改 GEMM kernel
3. **非标准形状**：对于某些特殊的矩阵形状/精度组合，cuBLAS 可能不是最优的

**Q: CUTLASS 和 cuBLAS 的关系？**

A: cuBLAS 是闭源的高度优化库（即开即用），CUTLASS 是开源的模板库（可定制）。cuBLAS 在常见场景通常更快，但 CUTLASS 可以实现自定义的融合 kernel。CUTLASS 的设计模式也是面试常考的。

**Q: FlashAttention 和 GEMM 优化有什么关系？**

A: FlashAttention 的核心思想就是将 Attention 的多个步骤（$QK^T$、softmax、$\times V$）融合，利用 tiling 将中间结果保持在 SRAM 中，避免写回 HBM。其优化技巧（tiling、register blocking、软件流水线）与 GEMM 优化一脉相承。

## 延伸阅读

- [CUDA GEMM Optimization Guide](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm 的经典博客（必读）
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/fundamental_types.md) — NVIDIA CUTLASS
- [How to Optimize a CUDA Matmul Kernel](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) — Lei Mao
- [Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md) — CUTLASS 官方解释

---

## 术语表

| 术语 | 说明 |
|------|------|
| **GEMM** | General Matrix Multiply，$C = A \times B$，深度学习中 70-80% 的计算时间花在 GEMM 上 |
| **cuBLAS** | NVIDIA 的闭源 BLAS（基础线性代数子程序）库，提供高度优化的 GEMM 实现 |
| **CUTLASS** | NVIDIA 的开源 C++ 模板库，可以自定义和融合 GEMM kernel |
| **Register Tiling** | 每个线程计算输出矩阵的一个小块（如 8×8），将中间结果保存在寄存器中，提高计算与访存的比值 |
| **双缓冲（Double Buffering）** | 用两块 Shared Memory 交替使用：计算当前块的同时预取下一块，隐藏数据加载延迟 |
| **WMMA** | Warp Matrix Multiply-Accumulate，Tensor Core 的编程接口，以 warp 为单位执行矩阵乘加 |
| **FLOPs** | Floating Point Operations，浮点运算次数。矩阵乘法 $C_{M \times N} = A_{M \times K} \times B_{K \times N}$ 的 FLOPs = $2MNK$ |
