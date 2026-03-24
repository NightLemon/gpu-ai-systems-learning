# 实战：GEMM 优化

> 从朴素实现到接近 cuBLAS 性能——GPU 优化能力的终极检验。

## 项目目标

写一个 SGEMM kernel（FP32 矩阵乘法），逐步优化到 cuBLAS 60%+ 性能。

## 环境准备

```bash
# 需要: CUDA Toolkit 12.x, 一张 NVIDIA GPU (最好 A100/H100)
# 编译
nvcc -o sgemm sgemm.cu -O2 -arch=sm_80  # A100
nvcc -o sgemm sgemm.cu -O2 -arch=sm_90  # H100
```

## 步骤

### Step 1: 朴素实现（目标: 正确性）

```cuda
// 每个线程计算 C 的一个元素
__global__ void sgemm_v0(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
```

**验证**：和 cuBLAS 结果对比，误差 < 1e-3。
**Profile**：用 Nsight Compute 看 roofline，确认当前是 memory-bound。

### Step 2: Shared Memory Tiling（目标: 10-15% cuBLAS）

- Block Size = 32×32
- Tile Size = 32
- 协作加载 A 和 B 的 tile 到 shared memory
- `__syncthreads()` 同步

**Profile**：查看 shared memory 使用率，确认 global memory 访问减少。

### Step 3: Register Tiling（目标: 25-35% cuBLAS）

- 每线程计算 8×8 的子矩阵（而非单个元素）
- Block Size = 128×128，Thread Block = 16×16
- 寄存器中做外积累加

**Profile**：查看寄存器使用和 occupancy 的 trade-off。

### Step 4: 向量化 + 双缓冲（目标: 45-55% cuBLAS）

- 使用 `float4` 向量化加载
- 双缓冲（ping-pong）: 预取下一个 tile 的同时计算当前 tile
- 消除 bank conflict（padding 或 swizzle）

**Profile**：查看 memory throughput 是否接近理论值。

### Step 5: 使用 Tensor Core（目标: 70-90% cuBLAS）

- 使用 `wmma` API 或 CUTLASS 模板
- 数据格式改为 FP16
- Warp 级别的 tiling

### Step 6 (可选): 用 Triton 实现

- 用 30 行 Python 达到和 Step 3-4 类似的性能
- 感受 Triton 的自动优化能力

## 性能测量

```cuda
// 正确的 GPU 计时方式
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
myKernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);

float gflops = 2.0f * M * N * K / (ms * 1e6);
printf("Performance: %.1f GFLOPS (%.1f%% of cuBLAS)\n", gflops, gflops/cublas_gflops*100);
```

## 参考资料

- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — **必读**，从V0到V5的完整优化过程
- [Lei Mao: CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- [CUTLASS GEMM Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
