# CUDA 编程基础

> 第一个 CUDA 程序：理解 Host/Device 模型、kernel 编写与启动。CUDA（Compute Unified Device Architecture）是 NVIDIA 提供的 GPU 编程框架，允许你用类 C++ 的语法编写运行在 GPU 上的并行程序。

## 为什么要学 CUDA？不是有 PyTorch 了吗？

PyTorch 确实封装了绝大多数 GPU 操作，你可以用 `tensor.cuda()` 就把数据传到 GPU 上。但当你需要做以下事情时，就必须理解 CUDA：

- **写自定义算子（Custom Operator）**：比如实现一种新的 Attention 变体，PyTorch 没有现成的
- **理解性能瓶颈**：为什么某个操作在 Profile 里显示很慢？不懂 CUDA 就无法理解 kernel 的行为
- **读懂框架源码**：FlashAttention、vLLM、NCCL 的核心代码都是 CUDA
- **面试**：几乎所有 GPU 系统岗位都会问 CUDA 基础

本节从零开始，带你写出第一个能跑在 GPU 上的程序。

## 核心概念

### Host 与 Device：两个世界

CUDA 编程的核心思维是“两个世界”——CPU（Host）和 GPU（Device）是两个独立的计算单元，有各自的内存。你需要显式地：分配 GPU 内存 → 把数据从 CPU 拷到 GPU → 启动 GPU 上的计算（kernel）→ 把结果拷回 CPU。虽然看起来繁琐，但这个明确的分离让你对数据在哪里、怎么搬运有完全的控制权。

```
┌─────────── Host (CPU) ────────────┐    ┌─────────── Device (GPU) ──────────┐
│                                    │    │                                    │
│  int main() {                      │    │  __global__ void kernel() {        │
│    // 分配 GPU 内存               │    │    // 在 GPU 上执行的代码          │
│    // 拷贝数据到 GPU              │───→│    int idx = threadIdx.x +         │
│    // 启动 kernel                 │    │            blockIdx.x*blockDim.x; │
│    // 拷贝结果回 CPU              │←───│  }                                 │
│  }                                 │    │                                    │
│  系统内存 (DDR)                    │    │  显存 (HBM)                        │
└────────────────────────────────────┘    └────────────────────────────────────┘
                        PCIe / NVLink
```

### CUDA 函数修饰符

| 修饰符 | 执行位置 | 调用者 | 说明 |
|--------|---------|--------|------|
| `__global__` | Device | Host（或 Device，Dynamic Parallelism） | kernel 函数，返回值必须为 void |
| `__device__` | Device | Device | 只能被 kernel 或其他 device 函数调用 |
| `__host__` | Host | Host | 普通 CPU 函数（默认） |
| `__host__ __device__` | 两者 | 两者 | 同时生成 CPU 和 GPU 版本 |

### 第一个 CUDA 程序

```cuda
#include <stdio.h>

// Kernel: 向量加法
__global__ void vec_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // 边界检查
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 20;  // 1M 元素
    size_t size = n * sizeof(float);
    
    // 1. 分配 Host 内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 初始化
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // 2. 分配 Device 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 3. 拷贝数据 Host → Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 4. 启动 Kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vec_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // 5. 拷贝结果 Device → Host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 6. 验证
    printf("c[0] = %f (expected 3.0)\n", h_c[0]);
    
    // 7. 释放内存
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}
```

编译运行：
```bash
nvcc -o vec_add vec_add.cu -O2
./vec_add
```

## 关键细节

### 线程索引计算

```
1D Grid, 1D Block:
  globalIdx = blockIdx.x * blockDim.x + threadIdx.x

2D Grid, 2D Block (矩阵操作常用):
  row = blockIdx.y * blockDim.y + threadIdx.y
  col = blockIdx.x * blockDim.x + threadIdx.x
  globalIdx = row * width + col

3D Grid, 3D Block (体素/3D卷积):
  x = blockIdx.x * blockDim.x + threadIdx.x
  y = blockIdx.y * blockDim.y + threadIdx.y
  z = blockIdx.z * blockDim.z + threadIdx.z
```

### 错误处理

CUDA API 调用都返回 `cudaError_t`，**必须检查**：

```cuda
// 宏：检查 CUDA API 错误
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 使用
CUDA_CHECK(cudaMalloc(&d_a, size));
CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

// Kernel 启动后检查错误（kernel 不返回 cudaError_t）
myKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());      // 检查 launch 错误
CUDA_CHECK(cudaDeviceSynchronize()); // 检查 execution 错误
```

### 内置变量

| 变量 | 类型 | 含义 |
|------|------|------|
| `threadIdx.x/y/z` | uint3 | 线程在 Block 内的索引 |
| `blockIdx.x/y/z` | uint3 | Block 在 Grid 内的索引 |
| `blockDim.x/y/z` | dim3 | Block 的维度（线程数） |
| `gridDim.x/y/z` | dim3 | Grid 的维度（Block 数） |
| `warpSize` | int | Warp 大小，目前始终为 32 |

### GPU 设备查询

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Device: %s\n", prop.name);
printf("SM count: %d\n", prop.multiProcessorCount);
printf("Max threads/block: %d\n", prop.maxThreadsPerBlock);
printf("Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
printf("Shared memory/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
printf("Registers/block: %d\n", prop.regsPerBlock);
printf("Warp size: %d\n", prop.warpSize);
printf("Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
printf("Memory bandwidth: %.0f GB/s\n", 
       prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9);
```

## 常见问题

**Q: `<<<gridDim, blockDim>>>` 里的数字怎么选？**

A: 经验法则：
- `blockDim`：通常 128 或 256（必须是 32 的倍数）
- `gridDim`：`(N + blockDim - 1) / blockDim`，确保覆盖所有数据
- 可以用 `cudaOccupancyMaxPotentialBlockSize` 自动选择

**Q: kernel 里能用 `printf` 吗？**

A: 可以，但有限制：
- 输出缓冲区大小有限（默认 1MB）
- 输出在 `cudaDeviceSynchronize()` 后才刷新
- 仅用于调试，会严重影响性能

**Q: `__syncthreads()` 和 `__syncwarp()` 的区别？**

A: `__syncthreads()` 同步整个 Block 的所有线程（barrier），`__syncwarp()` 只同步当前 warp 的 32 个线程。前者开销更大，但 Block 内跨 warp 通信必须用它。

## 延伸阅读

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — 官方文档，必读
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples) — 官方示例代码
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) — NVIDIA 入门博客

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Host** | 指 CPU 及其内存（系统内存），负责程序的主流程、数据准备和 kernel 启动 |
| **Device** | 指 GPU 及其显存，负责执行并行计算 |
| **Kernel** | 用 `__global__` 修饰的函数，由 Host 启动、在 Device 上并行执行 |
| **`__global__`** | CUDA 函数修饰符，表示这个函数是一个 kernel，由 CPU 调用、在 GPU 上执行 |
| **`__device__`** | CUDA 函数修饰符，表示只能在 GPU 上调用的函数 |
| **`cudaMalloc` / `cudaFree`** | 在 GPU 显存上分配/释放内存 |
| **`cudaMemcpy`** | 在 CPU 内存和 GPU 显存之间拷贝数据 |
| **`<<<gridDim, blockDim>>>`** | kernel 启动语法，指定 Grid 中有多少个 Block、每个 Block 有多少个线程 |
| **`threadIdx` / `blockIdx`** | CUDA 内置变量，用于获取当前线程在 Block 内 / Block 在 Grid 内的索引 |
| **`cudaDeviceSynchronize()`** | 阻塞 CPU 直到 GPU 上所有先前提交的工作完成（GPU 操作是异步的，不 sync 就测不准时间） |
