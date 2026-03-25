# Shared Memory 详解

> Shared Memory（共享内存）是 GPU SM 内部的高速存储，同一个 Block 内的所有线程都可以读写。它是 CUDA 优化的核心工具——用它作为手动管理的缓存，可以将对慢速 Global Memory（显存）的访问减少到最低限度。

## 用一个比喻理解 Shared Memory

回想一下 Ch02 的显存层级图。Global Memory（HBM）访问延迟 ~400 周期，Shared Memory 只要 ~20 周期——快了 20 倍。

把它想成一个团队协作的场景：32 个人（一个 Block）在同一间办公室里工作。Global Memory 就像远处的仓库，取一次资料要走 20 分钟。Shared Memory 就像办公室中间的共享工作台，拿东西只要 1 分钟。

所以明昺的做法是：**先派人去仓库把需要的资料批量搬到工作台上（加载到 Shared Memory），然后所有人在工作台上反复取用（从 Shared Memory 读），完事后再把结果一次性送回仓库（写回 Global Memory）**。这就是 Tiling（分块）的核心思想。

## 核心概念

### 为什么需要 Shared Memory？

```
问题：Global Memory (HBM) 延迟 ~400 cycles，带宽有限
方案：用 Shared Memory 做 Block 级别的手动缓存

                    ┌──────────────────┐
                    │   Global Memory   │  ~400 cycles, 3.35 TB/s
                    │     (HBM)         │
                    └────────┬─────────┘
                             │ 读一次
                    ┌────────▼─────────┐
                    │  Shared Memory    │  ~20 cycles, ~20 TB/s
                    │  (Block 内共享)   │
                    └────────┬─────────┘
                             │ 读多次
                    ┌────────▼─────────┐
                    │    Registers      │  ~0 cycles
                    │  (Thread 私有)    │
                    └──────────────────┘
```

### 声明方式

```cuda
// 方式一：静态分配（编译时大小已知）
__shared__ float smem[256];

// 方式二：动态分配（运行时指定大小）
extern __shared__ float smem[];
// 启动时通过第三个参数指定大小
myKernel<<<grid, block, sharedMemBytes>>>(args);
```

### 基本使用模式：Tiling

以矩阵乘法为例，Tiling 的核心思想：

```
目标: C = A × B，其中 A(M×K), B(K×N), C(M×N)

朴素做法: 每个线程计算 C 的一个元素，需要从 HBM 读取 A 的一行和 B 的一列
  → A 的每一行被读取 N 次，B 的每一列被读取 M 次
  → 总 HBM 读取量 = M*N*(K+K) = 2*M*N*K

Tiling: 将 A 和 B 分成小 Tile 加载到 Shared Memory，每个 Tile 被 Block 内所有线程共享
  → 每个 Tile 只从 HBM 读取一次
  → 总 HBM 读取量大幅减少
```

```
Tiling 示意（TILE_SIZE = 32）:

  A (M×K)                    B (K×N)
┌──┬──┬──┬──┐            ┌──┬──┬──┬──┐
│T0│T1│T2│..│            │T0│T1│T2│..│
├──┼──┼──┼──┤            ├──┼──┼──┼──┤
│  │  │  │  │    ──→     │  │  │  │  │
├──┼──┼──┼──┤            ├──┼──┼──┼──┤
│  │  │  │  │            │  │  │  │  │
└──┴──┴──┴──┘            └──┴──┴──┴──┘

每次迭代:
  1. Block 内所有线程协作，把 A 的一个 Tile 和 B 的一个 Tile 读入 Shared Memory
  2. __syncthreads()
  3. 每个线程从 Shared Memory 读取数据，计算部分和
  4. __syncthreads()
  5. 进入下一个 Tile
```

## 关键细节

### Bank Conflict

Shared Memory 被组织为 **32 个 Bank**（对应 warp 的 32 个线程），每个 bank 宽 4 bytes：

```
Bank:    0    1    2    3   ...   31
Addr:  [0]  [4]  [8]  [12] ... [124]   ← 第 0 行
       [128][132][136][140] ... [252]   ← 第 1 行
       ...
```

**同一 warp 中的线程同时访问不同 bank → 并行，无冲突**
**同一 warp 中的线程访问同一 bank 的不同地址 → 串行化（bank conflict）**

```
✅ 无 conflict：线程 i 访问 smem[i]
   Thread 0 → Bank 0, Thread 1 → Bank 1, ..., Thread 31 → Bank 31

✅ 无 conflict (broadcast)：所有线程访问同一 bank 的同一地址
   Thread 0-31 都访问 smem[0] → 硬件广播

❌ 2-way conflict：线程 0 和线程 16 访问同一 bank 的不同地址
   Thread 0 → smem[0] (Bank 0), Thread 16 → smem[128] (Bank 0)
   → 需要 2 个周期

❌ 32-way conflict（最坏）：所有线程访问同一 bank
   Thread i → smem[i * 32]  全部落在 Bank 0
   → 需要 32 个周期（完全串行化）
```

### 避免 Bank Conflict 的技巧

**技巧一：Padding**

```cuda
// ❌ 有 bank conflict 的矩阵转置
__shared__ float tile[32][32];
// 列方向访问: tile[0][0], tile[1][0], tile[2][0], ...
// 地址间隔 32×4=128 bytes → 全落在同一 Bank → 32-way conflict!

// ✅ Padding 后无 conflict
__shared__ float tile[32][33];  // 多一列 padding
// 列方向访问: tile[0][0], tile[1][0], tile[2][0], ...
// 地址间隔 33×4=132 bytes → Bank 0, Bank 1, Bank 2, ... 分散到不同 Bank
```

**技巧二：Swizzle**

更高级的地址重映射技巧，在 GEMM 优化中常用。核心思想是对 shared memory 的地址做 XOR 变换，使得相邻线程的访问自然分散到不同 bank。

### __syncthreads() — Block 级同步

```cuda
__shared__ float smem[256];

// Step 1: 所有线程写入 shared memory
smem[threadIdx.x] = global_data[idx];

// ⚠️ 必须同步！否则某些线程可能读到其他线程尚未写入的值
__syncthreads();

// Step 2: 线程读取其他线程写入的数据
float val = smem[threadIdx.x ^ 1];  // 读邻居的数据
```

**注意**：`__syncthreads()` 必须被 Block 中的**所有线程**执行到，不能放在条件分支中（除非所有线程都会进入该分支）。

## 代码示例

### Tiled Matrix Multiply

```cuda
#define TILE_SIZE 32

__global__ void matmul_tiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 遍历所有 Tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 协作加载 Tile 到 Shared Memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // 从 Shared Memory 计算部分和
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// 启动
dim3 block(TILE_SIZE, TILE_SIZE);  // 32×32 = 1024 threads
dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

### Shared Memory Reduction（归约求和）

```cuda
__global__ void reduce_sum(float *input, float *output, int N) {
    extern __shared__ float smem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载到 shared memory
    smem[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    
    // 树形归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    
    // Block 的结果写回 global memory
    if (tid == 0)
        output[blockIdx.x] = smem[0];
}
```

## 常见问题

**Q: Shared Memory 的大小限制是多少？**

A: 取决于 Compute Capability。H100（CC 9.0）每 SM 最大 228 KB（配置为最大 shared memory 时）。每个 block 请求的 shared memory 不能超过这个限制。用 `cudaFuncSetAttribute` 可以请求更多：

```cuda
cudaFuncSetAttribute(myKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 228 * 1024);
```

**Q: 动态 shared memory 和静态 shared memory 能混用吗？**

A: 可以。动态 shared memory 的起始地址紧跟在静态 shared memory 之后。但要注意总量限制。

**Q: 在 PyTorch 自定义 CUDA extension 中怎么用 Shared Memory？**

A: PyTorch 的 C++ extension 中可以直接写 CUDA kernel，使用 shared memory 和原生 CUDA 完全一样。通过 `torch.utils.cpp_extension` 编译。

## 延伸阅读

- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) — NVIDIA Blog
- [Matrix Multiplication with Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Bank Conflicts 详解](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Shared Memory** | SM 内部的高速 SRAM 存储，同一 Block 内的线程共享。需要程序员显式管理（分配/读写/同步） |
| **Tiling（分块）** | 将大矩阵切成小块（Tile），每次将一个 Tile 加载到 Shared Memory 中复用，减少 Global Memory 访问 |
| **Bank** | Shared Memory 被划分为 32 个 Bank，每个 Bank 宽 4 bytes。同一时刻不同线程访问不同 Bank 可以并行 |
| **Bank Conflict** | 同一 warp 中多个线程同时访问同一 Bank 的不同地址，访问会被串行化，严重时性能降至 1/32 |
| **Padding** | 在数组声明时多加一列（如 `tile[32][33]` 而非 `tile[32][32]`），使列方向访问分散到不同 Bank，避免 Bank Conflict |
| **`__shared__`** | CUDA 内存空间修饰符，声明变量位于 Shared Memory 中 |
| **`__syncthreads()`** | Block 内同步屏障，确保 Block 内所有线程都执行到此处后才继续，用于线程间数据交换前后 |
| **Reduction（归约）** | 将多个值合并为一个结果的操作，如求和、求最大值。树形归约是 GPU 上的经典并行算法 |
