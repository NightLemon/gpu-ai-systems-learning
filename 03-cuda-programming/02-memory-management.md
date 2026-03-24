# CUDA 内存管理

> 数据在 Host（CPU）和 Device（GPU）之间的搬运往往是性能瓶颈。揌握高效的 GPU 内存分配、传输和异步机制至关重要。

## 核心概念

### 内存分配与传输 API

```
Host Memory                          Device Memory (HBM)
┌──────────┐    cudaMemcpy (H2D)     ┌──────────┐
│  malloc   │ ──────────────────────→ │cudaMalloc│
│  h_data   │                         │  d_data  │
│           │ ←────────────────────── │          │
└──────────┘    cudaMemcpy (D2H)     └──────────┘
     ↑                                     ↑
   free()                            cudaFree()
```

### 基础 API

```cuda
// 分配 Device 内存
float *d_data;
cudaMalloc(&d_data, N * sizeof(float));

// Host → Device
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

// Device → Host
cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

// Device → Device
cudaMemcpy(d_dst, d_src, N * sizeof(float), cudaMemcpyDeviceToDevice);

// 释放
cudaFree(d_data);

// Device 内存清零
cudaMemset(d_data, 0, N * sizeof(float));
```

### Pinned Memory（页锁定内存）

普通 `malloc` 分配的内存可能被 OS 交换到磁盘（pageable），GPU 的 DMA 引擎无法直接访问，必须先拷贝到一块 pinned buffer：

```
Pageable Memory 传输:
  h_data (pageable) → staging buffer (pinned) → DMA → d_data (HBM)
                       ^^ 额外拷贝

Pinned Memory 传输:
  h_data (pinned) → DMA → d_data (HBM)
                    ^^ 直接传输，带宽可提升 2-3x
```

```cuda
// 分配 Pinned Memory
float *h_pinned;
cudaMallocHost(&h_pinned, size);  // 或 cudaHostAlloc

// 使用（和普通内存一样）
for (int i = 0; i < N; i++) h_pinned[i] = i;
cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);

// 释放
cudaFreeHost(h_pinned);
```

⚠️ Pinned Memory 注意事项：
- 分配在物理内存中，不会被换页 → 分配太多会挤占系统可用内存
- 分配/释放比 `malloc` 慢
- 适合需要频繁 CPU↔GPU 传输的场景

### Unified Memory（统一内存）

```cuda
float *data;
cudaMallocManaged(&data, size);

// CPU 和 GPU 都能用同一个指针
for (int i = 0; i < N; i++) data[i] = i;  // CPU 写
myKernel<<<grid, block>>>(data, N);          // GPU 读写
cudaDeviceSynchronize();
printf("data[0] = %f\n", data[0]);          // CPU 读

cudaFree(data);
```

**工作原理**：Driver 通过 page fault 机制自动在 CPU 和 GPU 之间迁移数据。

| 优点 | 缺点 |
|------|------|
| 编程简单，不需要手动管理传输 | Page fault 导致额外延迟 |
| 按需迁移，不需要一次性拷贝全部数据 | 性能不如手动管理的 pinned + async |
| 支持 oversubscription（数据总量 > 显存） | 迁移粒度较大（4KB 页或更大） |

> **建议**：原型验证用 Unified Memory（简单快速），性能敏感场景用手动管理 + async。

### 异步内存操作与 Stream

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);

// 异步拷贝（必须使用 pinned memory）
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
myKernel<<<grid, block, 0, stream>>>(d_a, d_b, N);
cudaMemcpyAsync(h_b, d_b, size, cudaMemcpyDeviceToHost, stream);

cudaStreamSynchronize(stream);  // 等待该 stream 上所有操作完成
cudaStreamDestroy(stream);
```

## 关键细节

### PCIe 带宽 — 隐藏的瓶颈

```
PCIe Gen4 x16: ~32 GB/s（双向各 32 GB/s）
PCIe Gen5 x16: ~64 GB/s

VS.

HBM3 (H100):   ~3,350 GB/s

→ CPU↔GPU 传输速度只有 GPU 内部带宽的 1/50!
→ 尽量减少 CPU↔GPU 数据传输
```

优化策略：
1. **一次性传输所有数据到 GPU**，在 GPU 上完成所有计算后再传回
2. **用 Stream 重叠传输和计算**（pipeline）
3. 避免频繁的小数据传输（batch 起来）

### Stream 实现 Overlap（并行流水线）

```cuda
// 将数据分成 nStreams 份，用多个 stream 并行处理
const int nStreams = 4;
cudaStream_t streams[nStreams];
for (int i = 0; i < nStreams; i++)
    cudaStreamCreate(&streams[i]);

int chunkSize = N / nStreams;
for (int i = 0; i < nStreams; i++) {
    int offset = i * chunkSize;
    // 各 stream 的操作互相独立，可以并行
    cudaMemcpyAsync(d_a + offset, h_a + offset, 
                     chunkSize * sizeof(float),
                     cudaMemcpyHostToDevice, streams[i]);
    myKernel<<<grid, block, 0, streams[i]>>>(d_a + offset, chunkSize);
    cudaMemcpyAsync(h_result + offset, d_a + offset,
                     chunkSize * sizeof(float),
                     cudaMemcpyDeviceToHost, streams[i]);
}

// 等待所有 stream 完成
cudaDeviceSynchronize();
```

时间线可视化：
```
Stream 0: [H2D chunk0] [Compute chunk0] [D2H chunk0]
Stream 1:              [H2D chunk1] [Compute chunk1] [D2H chunk1]
Stream 2:                          [H2D chunk2] [Compute chunk2] [D2H chunk2]
Stream 3:                                      [H2D chunk3] [Compute chunk3] [D2H chunk3]
                                                                        ↑
                                                        总时间显著缩短！
```

### cudaMemcpy 的同步行为

| API | 对 Host 同步？ | 对 Device 同步？ |
|-----|--------------|----------------|
| `cudaMemcpy` | 是（阻塞到完成） | 是 |
| `cudaMemcpyAsync` | 否（立即返回） | 仅在同一 stream 内有序 |
| `cudaDeviceSynchronize` | 是（等待所有 GPU 操作） | 是 |
| `cudaStreamSynchronize` | 是（等待指定 stream） | 指定 stream |

## 常见问题

**Q: 什么时候用 `cudaMallocManaged` vs `cudaMalloc`？**

A: 快速原型 → `cudaMallocManaged`；性能敏感 → `cudaMalloc` + `cudaMemcpyAsync`。在 PyTorch 等框架中，内存管理已经被封装好了（`tensor.to('cuda')`），通常不需要手动调用这些 API。

**Q: PyTorch 的 CUDA memory 管理和原生 CUDA 有什么关系？**

A: PyTorch 内部有一个 **CUDA Memory Allocator（caching allocator）**：
- 第一次分配时调用 `cudaMalloc` 获取大块内存
- 后续的 tensor 分配从这个缓存池中切割，避免频繁调用 `cudaMalloc`（很慢）
- `torch.cuda.empty_cache()` 释放缓存池中未使用的内存块
- `torch.cuda.memory_allocated()` 查看实际使用量

**Q: `cudaMalloc` 和 `cudaMallocAsync`（CUDA 11.2+）的区别？**

A: `cudaMallocAsync` 使用 CUDA 的 stream-ordered memory allocator，分配和释放都是 stream 中的操作：
- 更好地重用内存，减少碎片
- 分配/释放更快（不需要全局同步）
- PyTorch 2.0+ 可以选择使用它作为后端 allocator

## 延伸阅读

- [CUDA C++ Programming Guide - Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)
- [CUDA Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [CUDA Streams Best Practices](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Pinned Memory（页锁定内存）** | 用 `cudaMallocHost()` 分配的 CPU 内存，不会被操作系统换出到磁盘，GPU 的 DMA 引擎可以直接访问，传输更快 |
| **Unified Memory（统一内存）** | 用 `cudaMallocManaged()` 分配，CPU 和 GPU 可以用同一个指针访问，驱动自动在两者之间迁移数据 |
| **DMA（Direct Memory Access）** | GPU 上的硬件引擎，可以在不占用 CPU 的情况下在内存之间搉数据 |
| **Stream（流）** | CUDA 中的任务队列，同一 Stream 内的操作按顺序执行，不同 Stream 的操作可以并行 |
| **`cudaMemcpyAsync`** | 异步内存拷贝，不阻塞 CPU，但要求使用 Pinned Memory 并指定 Stream |
| **Overlap（重叠）** | 通过多 Stream 让数据传输和 kernel 计算同时进行，缩短总时间 |
| **Caching Allocator** | PyTorch 的 GPU 内存管理器，预分配大块显存并在内部切割复用，避免频繁调用慢速的 `cudaMalloc` |
