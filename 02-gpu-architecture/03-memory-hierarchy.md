# GPU 显存层级

> GPU 内部有多层存储器（从最快的寄存器到最大的 HBM 显存），每层速度和容量不同。理解这个层次结构是写出高性能 CUDA kernel 的关键——大多数优化本质上都是在“减少慢存储的访问，增加快存储的复用”。
## 为什么需要这么多层存储？

在 Ch01 中我们讲了 CPU 的缓存层级——GPU 面临的问题是一样的，只是更极端。

GPU 的计算单元（Tensor Core）每秒能做近 2000 万亿次运算（H100 BF16），但 HBM 显存每秒只能搬运 3.35 TB 数据。做一次简单的除法：**2000 TFLOPS / 3.35 TB/s ≈ 每读一个字节需要做 590 次运算才能"喂饱"计算单元**。绝大多数操作达不到这个比值——也就是说，**GPU 的计算单元大部分时间在等数据**。

所以 GPU 在计算单元和 HBM 之间塞了多层高速存储（Register → Shared Memory → L1 → L2 → HBM），每层更快但更小。CUDA 优化的核心就是：**把数据尽量放在快的层级上反复使用，减少对慢层级（HBM）的访问**。这就是为什么 FlashAttention 比标准 Attention 快 2-4 倍——不是因为计算少了，而是因为减少了 HBM 读写。
## 核心概念

### 显存层级总览

```
速度快                                                    容量大
延迟低                                                    延迟高
←─────────────────────────────────────────────────────────────→

 Register    Shared      L1        L2 Cache      HBM
  File      Memory     Cache                   (Global)
┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐
│ 64K个   │ │ 164KB  │ │ 与SM   │ │  50 MB   │ │  80 GB   │
│ 32-bit  │ │ /SM    │ │ 共享   │ │ (H100)   │ │ (H100)   │
│ per SM  │ │        │ │ 配置   │ │          │ │          │
│         │ │        │ │        │ │          │ │          │
│ ~0 cyc  │ │ ~20cyc │ │~20cyc  │ │ ~200 cyc │ │~400 cyc  │
│ ~20TB/s │ │~20TB/s │ │        │ │ ~12 TB/s │ │ 3.35TB/s │
└────────┘ └────────┘ └────────┘ └──────────┘ └──────────┘
 线程私有   Block共享   SM级别     全局共享       全局共享
```

### 各层级详解

#### 1. Register File（寄存器堆）

- **范围**：每个线程私有
- **容量**：H100 每个 SM 有 256KB register file（64K × 32-bit）
- **分配**：编译器自动分配，每个线程可用的 register 数量取决于 block 中的线程数
- **性能**：零额外延迟，是最快的存储

```
关键约束：register 总量有限
- 每个线程用的 register 越多 → 同时驻留的线程越少 → occupancy 越低
- 需要在「每线程用更多寄存器」和「更高 occupancy」之间权衡
```

#### 2. Shared Memory（共享内存）

- **范围**：同一个 Block 内所有线程共享
- **容量**：H100 每 SM 最高 228 KB（与 L1 可配置划分）
- **用途**：线程间通信、手动管理的高速缓存
- **性能**：约 20 cycle 延迟，带宽接近 register

```
Shared Memory 组织为多个 Bank：

  Bank 0   Bank 1   Bank 2   ...  Bank 31
┌────────┬────────┬────────┬─────────────┐
│ addr 0 │ addr 4 │ addr 8 │    ...      │  ← 连续地址分布在不同 Bank
│ addr128│ addr132│ addr136│    ...      │  ← 每 128 字节一轮
│  ...   │  ...   │  ...   │    ...      │
└────────┴────────┴────────┴─────────────┘

⚠️ Bank Conflict: 同一 warp 中多个线程访问同一 Bank 的不同地址
   → 访问被串行化，性能急剧下降
```

#### 3. L1 Cache

- 与 Shared Memory 物理上共享同一块 SRAM
- 从 Volta 开始，L1 Cache 和 Shared Memory 的大小可配置
- L1 by default 缓存 global memory 的读取（自动管理）

#### 4. L2 Cache

- **范围**：所有 SM 共享
- **容量**：H100 为 50 MB
- **作用**：缓冲 HBM 访问，减少对 HBM 带宽的压力
- 你无法直接控制 L2 的行为，但可以通过访存模式优化 cache 命中率

#### 5. HBM / Global Memory

- **范围**：所有线程可见
- **容量**：H100 为 80 GB
- **带宽**：3.35 TB/s（HBM3）
- **延迟**：约 400-600 cycles

```
⚠️ HBM 访问是 GPU 中最大的性能瓶颈
   优化策略：
   1. Coalesced Access（合并访问）
   2. 尽量用 Shared Memory 做中间缓存
   3. 减少不必要的 global memory 读写
```

## 关键细节

### Coalesced Memory Access（合并访存）

GPU 以 **128 字节（32 × 4 bytes）** 的粒度从 HBM 读取数据。一个 warp（32 线程）同时发出的内存请求会被硬件合并：

```
✅ 理想情况（Coalesced）:
Thread 0 → addr[0], Thread 1 → addr[1], ..., Thread 31 → addr[31]
→ 一次 128B 事务搞定

❌ 最坏情况（Strided / Random）:
Thread 0 → addr[0], Thread 1 → addr[1000], Thread 2 → addr[2000], ...
→ 32 次独立事务，带宽利用率 1/32 = 3%
```

### 显存不同层级的编程接口

```c
// Register — 编译器自动分配，局部变量
int x = threadIdx.x;  // 在 register 中

// Shared Memory — 显式声明
__shared__ float smem[256];
smem[threadIdx.x] = global_data[idx];
__syncthreads();  // Block 内同步

// Global Memory — 通过指针访问
float* d_data;  // 指向 HBM 的指针
cudaMalloc(&d_data, size);

// Constant Memory — 只读，有专用 cache（64KB）
__constant__ float params[256];

// Texture Memory — 只读，对 2D 空间局部性优化（深度学习中较少用）
```

### ECC 的影响

数据中心 GPU（A100/H100）默认开启 ECC（Error-Correcting Code），这会：
- **减少约 6% 的可用显存**（用于存储 ECC 校验位）
- **略降低带宽**（每次读写都要计算校验）
- 但在长时间训练中是必要的——HBM 的 bit flip 率不可忽视

## 代码示例

比较 Global Memory 直接访问 vs Shared Memory 优化：

```cuda
// ❌ 朴素版：每次从 global memory 读
__global__ void naive_transpose(float *out, float *in, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N)
        out[x * N + y] = in[y * N + x];  // 写 coalesced，读 strided!
}

// ✅ Shared Memory 优化版
__global__ void smem_transpose(float *out, float *in, int N) {
    __shared__ float tile[32][33];  // 33 而非 32，避免 bank conflict!
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // 从 global 读到 shared（coalesced read）
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    
    // 从 shared 写到 global（coalesced write）
    x = blockIdx.y * 32 + threadIdx.x;  // 注意交换了 x/y
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}
```

> `tile[32][33]` 中的 33 是经典技巧：通过 padding 一列避免列方向访问时的 bank conflict。

## 常见问题

**Q: Shared Memory 和 L1 Cache 到底是什么关系？**

A: 从 Volta (V100) 开始，Shared Memory 和 L1 Cache **共享同一块物理 SRAM**。你可以配置它们的比例。例如 H100 每 SM 有 256 KB SRAM，可以配置为 128 KB shared + 128 KB L1，或 228 KB shared + 28KB L1 等。对于需要大量线程间通信的 kernel，多分配 shared memory；对于访存模式规则的 kernel，多留给 L1。

**Q: 为什么 GPU 不直接用更大的 cache 代替 HBM？**

A: SRAM（cache/shared memory 用的材料）比 DRAM/HBM **贵 10-100 倍**且面积大，在芯片上放不下 80GB 的 SRAM。GPU 的策略是用 HBM 提供大容量 + 高带宽，用少量 SRAM 做高速缓冲。

**Q: 什么时候该手动用 Shared Memory，什么时候靠 L1 Cache 就够了？**

A: 经验法则：
- 数据会被同一 block 内的多个线程**重复访问** → 用 Shared Memory
- 数据只被每个线程访问一次，且访存模式规则 → 靠 L1 Cache
- 需要线程间**通信/协作** → 必须用 Shared Memory

## 延伸阅读

- [CUDA C++ Programming Guide - Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [NVIDIA GPU Memory Tutorial](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Understanding GPU Memory Access Patterns](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Register（寄存器）** | GPU 中每个线程私有的最快存储，由编译器自动分配，容量有限 |
| **Shared Memory（共享内存）** | SM 内部的高速存储，同一个 Block 内的所有线程可以读写，通常用于手动缓存和线程间通信 |
| **L1/L2 Cache** | 硬件自动管理的缓存。L1 在 SM 内部（与 Shared Memory 共享同一块 SRAM），L2 全局共享 |
| **HBM / Global Memory** | GPU 的主显存（如 H100 的 80 GB HBM3）。容量最大但访问最慢，所有线程可见 |
| **SRAM** | Static RAM，静态随机存取存储器。速度快但单位成本高，GPU 内的 Shared Memory 和 Cache 都用 SRAM 实现 |
| **Coalesced Access（合并访存）** | 同一个 warp 中的 32 个线程同时访问连续的内存地址，硬件可以合并为一次内存事务，效率最高 |
| **Bank Conflict（Bank 冲突）** | Shared Memory 分为 32 个 Bank，当同一 warp 中多个线程同时访问同一 Bank 的不同地址时，访问会被串行化 |
| **Constant Memory** | 64KB 的只读内存，有专用 Cache，适合存放所有线程都读取的相同参数 |
| **`__syncthreads()`** | CUDA 中的 Block 级别同步原语，确保 Block 内所有线程都执行到这一点后才继续 |
