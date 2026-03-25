# GPU 执行模型

> CUDA 程序的线程按 Grid → Block → Warp → Thread 四层结构组织。理解这个层级和硬件调度机制，是写出高效 kernel 的前提。

## 为什么 GPU 的线程模型和 CPU 不一样？

在 CPU 编程中，你可能用过 `pthread` 或 `std::thread` 开几十个线程。CPU 线程很"重"——每个线程有独立的栈、由操作系统调度、上下文切换成本高。所以 CPU 的最佳线程数通常等于核心数（几十个）。

GPU 的线程完全不同：**它们"极轻"——创建零开销、切换零开销、数量可以达到百万级**。GPU 的策略不是"少量线程各自干大活"，而是"海量线程各自干一丁点小活，靠数量取胜"。

但海量线程需要组织结构。如果百万个线程都是一盘散沙，硬件没法调度。所以 CUDA 设计了一个清晰的四层层级：Grid → Block → Warp → Thread。每一层都有明确的角色：

- **Grid**：一次 kernel 启动的全部线程（可以有几百万个）
- **Block**：一组可以互相协作的线程（最多 1024 个），共享 Shared Memory
- **Warp**：32 个线程的执行组——这是 GPU 实际调度的最小单位
- **Thread**：最小的逻辑执行单元

理解这个层级不只是记术语——它直接决定了你写 kernel 时的三个核心问题：blockDim 怎么选？Shared Memory 怎么用？Warp Divergence 怎么避免？

## 核心概念

### 线程层级结构

当你用 `kernel<<<gridDim, blockDim>>>()` 启动一个 CUDA kernel 时，GPU 会创建如下层级的线程结构：

```
┌─────────────────────────── Grid ──────────────────────────────┐
│                                                                │
│  ┌──── Block(0,0) ───┐  ┌──── Block(1,0) ───┐  ┌─── ... ──┐ │
│  │ ┌───┬───┬───┬───┐ │  │ ┌───┬───┬───┬───┐ │  │          │ │
│  │ │ W0│ W1│ W2│ W3│ │  │ │ W0│ W1│ W2│ W3│ │  │          │ │
│  │ └───┴───┴───┴───┘ │  │ └───┴───┴───┴───┘ │  │          │ │
│  │ ┌───┬───┬───┬───┐ │  │ ┌───┬───┬───┬───┐ │  │          │ │
│  │ │ W4│ W5│ W6│ W7│ │  │ │ W4│ W5│ W6│ W7│ │  │          │ │
│  │ └───┴───┴───┴───┘ │  │ └───┴───┴───┴───┘ │  │          │ │
│  │  Shared Memory     │  │  Shared Memory     │  │          │ │
│  └────────────────────┘  └────────────────────┘  └──────────┘ │
│  Block(0,1)               Block(1,1)              ...         │
│  ...                      ...                                 │
└────────────────────────────────────────────────────────────────┘

Grid: 所有 Block 的集合（对应一次 kernel 启动）
Block: 一组线程（最多 1024 个），共享 Shared Memory
Warp: 32 个连续线程，是**实际执行的最小调度单位**
Thread: 最小的逻辑执行单元
```

### Warp — GPU 的灵魂

**Warp 是 GPU 执行的核心概念。** 一个 warp = 32 个线程，它们：

1. **同时执行同一条指令**（SIMT — Single Instruction Multiple Threads）
2. 共享一个 Program Counter（PC）
3. 由一个 Warp Scheduler 调度

```
Warp 0（32 线程同步执行同一条指令）:
  Thread 0:  FADD R1, R2, R3
  Thread 1:  FADD R1, R2, R3
  Thread 2:  FADD R1, R2, R3
  ...
  Thread 31: FADD R1, R2, R3
  ← 同一时钟周期
```

### Warp Scheduling（Warp 调度）

每个 SM 的 sub-core 有一个 **Warp Scheduler**，管理多个 warp：

```
SM Sub-core Warp Scheduler:

时钟周期 1: Warp 0 执行 → 发出内存请求，需要等待 400 cycles
时钟周期 2: Warp 1 执行 → 正常执行
时钟周期 3: Warp 2 执行 → 正常执行
时钟周期 4: Warp 3 执行 → 发出内存请求
...
时钟周期 N: Warp 0 的内存请求返回 → 继续执行

→ 通过快速切换 warp 来隐藏内存延迟（Latency Hiding）
→ 所以 GPU 需要大量线程！驻留的 warp 越多，延迟隐藏越充分
```

## 关键细节

### Warp Divergence

当同一个 warp 中的线程走了**不同的分支**，就会发生 divergence：

```c
if (threadIdx.x < 16) {
    // Path A — 线程 0-15
    do_something_a();
} else {
    // Path B — 线程 16-31
    do_something_b();
}
```

硬件处理方式（Volta 之前）：
```
Step 1: 所有 32 线程执行 Path A，线程 16-31 被 mask（不写结果）
Step 2: 所有 32 线程执行 Path B，线程 0-15 被 mask

→ 总时间 = Path A + Path B（而非 max(A, B)）
→ 性能最坏可能降至 1/32
```

Volta+ 引入了 **Independent Thread Scheduling**，允许不同分支的线程更灵活地执行，但 divergence 仍然有性能代价。

**最佳实践**：让同一 warp 中的线程尽量走相同的分支。

### Block 到 SM 的映射

```
Kernel Launch:
  grid = (gridDim.x, gridDim.y, gridDim.z)
  block = (blockDim.x, blockDim.y, blockDim.z)

Block 分配到 SM:
  ┌────┐ ┌────┐ ┌────┐ ┌────┐
  │SM 0│ │SM 1│ │SM 2│ │SM 3│ ...
  └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘
     │      │      │      │
  Block0  Block1  Block2  Block3
  Block4  Block5  Block6  Block7
  ...

规则：
  - 一个 Block 只在一个 SM 上执行（不会跨 SM）
  - 一个 SM 可以同时运行多个 Block（受资源限制）
  - Block 之间没有执行顺序保证
  - Block 间同步只能通过 kernel 结束（global barrier）
```

### Occupancy（占用率）

$$\text{Occupancy} = \frac{\text{SM 上实际驻留的 warp 数}}{\text{SM 支持的最大 warp 数}}$$

影响 Occupancy 的三个资源：

| 资源 | 限制说明 |
|------|---------|
| **Threads per Block** | 每 SM 最多 2048 线程（H100），block 越大 → 可驻留的 block 越少 |
| **Registers per Thread** | 寄存器总量有限，每线程用的越多 → 能驻留的线程越少 |
| **Shared Memory per Block** | Shared Memory 总量有限，block 用的越多 → 能驻留的 block 越少 |

示例（H100 SM）：
```
最大：2048 threads/SM = 64 warps/SM
场景 A: blockDim=256, 每线程 32 regs, shared=0
  → 每 SM 可放 8 blocks × 256 = 2048 threads → occupancy = 100%
场景 B: blockDim=256, 每线程 128 regs, shared=0
  → register 限制：256KB / (128×4B) = 512 threads/SM → occupancy = 25%
场景 C: blockDim=256, 每线程 32 regs, shared=100KB
  → shared 限制：228KB / 100KB = 2 blocks → 512 threads → occupancy = 25%
```

> **Occupancy 不是越高越好！** 有时降低 occupancy 但每线程用更多 register/shared memory 反而更快。关键是看是否有足够的 warp 来隐藏延迟。

### Kernel Launch 配置

```c
// kernel<<<gridDim, blockDim, sharedMem, stream>>>()
dim3 blockDim(256);  // 256 threads per block
dim3 gridDim((N + 255) / 256);  // 向上取整，确保覆盖所有数据
myKernel<<<gridDim, blockDim>>>(data, N);
```

选择 Block Size 的经验法则：
- 必须是 32 的倍数（warp 大小）
- 通常 128 或 256 是好的起点
- 用 `cudaOccupancyMaxPotentialBlockSize` 让驱动自动选择

```c
int blockSize, minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
myKernel<<<minGridSize, blockSize>>>(data, N);
```

## 代码示例

观察 Warp Divergence 的影响：

```cuda
// ❌ 坏的 divergence 模式：同一 warp 内交替分支
__global__ void bad_divergence(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (threadIdx.x % 2 == 0)   // warp 内奇偶线程走不同分支!
            data[idx] *= 2.0f;
        else
            data[idx] += 1.0f;
    }
}

// ✅ 好的模式：分支以 warp 为粒度
__global__ void good_divergence(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int warp_id = threadIdx.x / 32;
        if (warp_id % 2 == 0)       // 同一 warp 内所有线程走同一分支
            data[idx] *= 2.0f;
        else
            data[idx] += 1.0f;
    }
}
```

## 常见问题

**Q: blockDim 最大只能 1024，但每 SM 支持 2048 线程，为什么？**

A: 这允许一个 SM 同时运行多个 block（比如 2 个 1024-thread block 或 8 个 256-thread block）。多个 block 驻留在同一 SM 上能提供更多 warp 来隐藏延迟。

**Q: Grid 中的 Block 是按什么顺序被调度到 SM 上的？**

A: NVIDIA 没有公开具体调度策略，你**不能依赖任何特定顺序**。Block 之间的唯一同步点是 kernel 结束。如果需要 block 间通信，要么用 atomic operation + global memory，要么启动新 kernel。

**Q: 什么是 Cooperative Groups？**

A: CUDA 9.0 引入的 API，允许更灵活的线程分组和同步，包括 warp 级同步（`__syncwarp()`）、block 级之外的 grid 级同步（需硬件支持）。这在需要 block 间协作的算法中很有用。

## 延伸阅读

- [CUDA C++ Programming Guide - Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/) — 在线工具
- [Life of a CUDA Kernel](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) — NVIDIA Blog

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Grid** | 一次 kernel 启动创建的所有 Block 的集合，是最外层的线程组织单元 |
| **Block（线程块）** | 一组线程的集合（最多 1024 个），同一 Block 内的线程可以共享 Shared Memory 并同步 |
| **Warp** | 32 个连续线程的执行组，是 GPU 实际调度和执行的最小单位。同一 warp 的 32 个线程同步执行同一条指令 |
| **Thread（线程）** | 最小的逻辑执行单元，每个线程有自己的寄存器和程序计数器 |
| **Warp Divergence（分支发散）** | 同一 warp 中的线程走了不同的 if/else 分支，硬件必须串行执行两个分支，导致性能下降 |
| **Occupancy（占用率）** | SM 上实际驻留的 warp 数 / SM 支持的最大 warp 数。越高意味着有更多 warp 可以切换以隐藏延迟 |
| **Warp Scheduler** | SM 内部的硬件模块，负责从多个就绪的 warp 中选择一个发射指令执行 |
| **Latency Hiding（延迟隐藏）** | 当一个 warp 等待内存数据时，Warp Scheduler 立即切换到另一个就绪的 warp 执行，从而“隐藏”等待时间 |
| **`__launch_bounds__`** | CUDA 修饰符，告诉编译器每个 Block 的最大线程数和最少驻留 Block 数，帮助编译器优化寄存器分配 |
| **Cooperative Groups** | CUDA 9.0 引入的 API，提供比 `__syncthreads()` 更灵活的线程分组和同步机制 |
