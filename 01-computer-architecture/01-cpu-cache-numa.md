# CPU Cache 与 NUMA

> HPC（高性能计算）视角下 CPU Cache（缓存）和 NUMA（非一致性内存访问）的关键优化点。本节假设你了解基本的 CPU 架构，重点补充与高性能计算和 AI 训练相关的细节。

## 为什么后端工程师也需要关心 CPU 缓存？

你可能觉得"AI 训练不是都跑在 GPU 上吗，CPU 的事跟我有什么关系？"

现实是：**GPU 训练系统里的 CPU 一点都不轻松**。PyTorch 的 DataLoader 用多个 CPU 进程在后台读磁盘、做 tokenize、组 batch、传给 GPU。如果这些 CPU 进程跑得不够快，GPU 就会空转等数据——你花大价钱买的 A100/H100 利用率可能只有 60%，而瓶颈竟然在 CPU 上。

更隐蔽的是，在多路服务器（多个 CPU 插槽共享内存）上，如果你的进程和它访问的内存不在同一个 NUMA 节点上，单纯的一次内存读取可能就慢了 50%。这种性能损失不会报错，只会默默拖慢吞吐。

所以，理解 CPU 缓存和 NUMA 不是为了让你优化 CPU 代码——而是让你**在排查 GPU 训练瓶颈时，不会忽略 CPU 这一侧的问题**。

## 核心概念

### Cache 层级：为什么 CPU 不直接读内存？

想象你在图书馆写论文。图书馆有上百万本书（= 主内存 DRAM），但你的书桌只能放几本（= L1 Cache）。每次需要一个数据，你是希望从桌上拿（1 纳秒），还是每次都跑去书架取（100 纳秒）？

这就是 CPU 缓存存在的原因。现代 CPU 用三级缓存来弥补处理器速度（~0.3ns/周期）和主内存速度（~100ns）之间 **300 倍**的差距：

```
CPU Core → L1 (64KB, ~1ns) → L2 (1MB, ~3ns) → L3 (多核共享, 30MB+, ~10ns) → DRAM (~100ns)
           ↑ 最快最小        ↑ 较快较大       ↑ 所有核心共享               ↑ 容量最大但最慢
```

数据第一次从 DRAM 加载时很慢（~100ns），但之后它会留在缓存中。只要你接下来还会用到同一块数据（**时间局部性**）或相邻的数据（**空间局部性**），后续访问就能命中缓存，快 10-100 倍。

### Cache Line：CPU 搬数据的"最小包裹"

CPU 不是一个字节一个字节地从内存读数据的——它每次搬运的最小单位是一条 **Cache Line（缓存行），通常是 64 字节**。

这意味着什么？假设你有一个 int 数组 `arr[100]`，你读了 `arr[0]`（4 bytes），CPU 会把 `arr[0]` 到 `arr[15]` 这 64 字节一起搬进缓存。如果你接下来依次读 `arr[1]`、`arr[2]`...，它们已经在缓存里了——几乎零延迟。

但如果你的访问模式是跳跃的（比如每隔 100 个元素读一个），每次访问都要搬一条新的 Cache Line，缓存中已有的数据又用不上——这叫**缓存未命中（cache miss）**，性能会急剧下降。

**工程意义**：设计数据结构时，**让连续访问的数据在内存中也是连续排列的**。这就是为什么在 GPU 编程中也反复强调"coalesced access"（合并访问）——本质上是同一个道理：让硬件搬运数据的粒度和你实际使用数据的模式对齐。

### False Sharing：一个隐蔽的多线程性能杀手

假设你有 4 个线程各自维护一个独立的计数器。逻辑上它们完全不干涉对方。但如果这 4 个计数器在内存中紧挨着（比如放在一个数组里），它们可能落在同一条 64 字节的 Cache Line 中。

此时问题来了：当线程 0 写了 `counter[0]`，CPU 的**缓存一致性协议**（cache coherency protocol）会通知其他所有核心："你们缓存的那条 line 过期了，得重新从内存/上级缓存取"。线程 1 写 `counter[1]` 时，又触发同样的失效通知。结果：4 个线程互相让对方的缓存失效，实际性能可能比单线程还差。

这就是 **False Sharing（伪共享）**——逻辑上无关的数据在物理上共享了缓存行。

```c
// ❌ False Sharing: 多个线程的 counter 紧挨着，落在同一条 cache line 中
struct { int counter[NUM_THREADS]; } shared;

// ✅ Padding 避免: 用 alignas(64) 让每个 counter 独占一条 cache line
struct { alignas(64) int counter; } per_thread[NUM_THREADS];
```

你不太可能在 AI 训练的 Python 代码中直接遇到 False Sharing（PyTorch 框架层面已经处理了），但 NCCL 通信库、自定义 CUDA extension 里如果涉及 CPU 端多线程，这是要注意的点。

### NUMA：为什么同一台机器里内存速度还不一样？

在你的笔记本或台式机上，所有 CPU 核心访问内存的速度是一样的——因为只有一个 CPU 插槽。但在训练服务器上（比如双路 Xeon），有两个 CPU 插槽，每个插槽有自己直连的内存：

```
                  ┌────────────────────────┐
                  │     双路 Xeon 服务器      │
                  │                          │
  ┌─ Socket 0 ──┐│                 ┌─ Socket 1 ──┐
  │ CPU 核 0-31  ││                 │ CPU 核 32-63 │
  │ 本地 DRAM    ││  ←— QPI/UPI —→ │ 本地 DRAM    │
  │ GPU 0-3      ││                 │ GPU 4-7      │
  └──────────────┘│                 └──────────────┘
                  └────────────────────────┘

  Socket 0 的核 访问 Socket 0 的 DRAM: ~100ns（本地）
  Socket 0 的核 访问 Socket 1 的 DRAM: ~150ns（跨插槽，慢 50%）
```

这就是 NUMA——Non-Uniform Memory Access，访问不同位置的内存速度不同。

**在 AI 训练中的实际影响**：假设 GPU 0 挂在 Socket 0 的 PCIe 上，但负责给 GPU 0 喂数据的 CPU 进程跑在了 Socket 1 的核心上。这个进程要先读取数据（可能跨 NUMA），然后通过 Socket 1 → Socket 0 的互联把数据传给 GPU。**一来一回，延迟和带宽都打了折扣**。

```bash
# 查看服务器的 NUMA 拓扑（哪些核心、哪些内存、哪些 PCIe 设备在哪个节点）
numactl --hardware

# 将训练脚本绑定到 NUMA 节点 0（确保 CPU 和 GPU 在同侧）
numactl --cpunodebind=0 --membind=0 ./my_training_script

# 很多训练框架（如 DeepSpeed、Megatron-LM）会自动做 NUMA 绑定
# 但如果你发现 DataLoader 吞吐异常低，值得检查这一点
```

## 什么时候你会需要用到这些知识？

| 场景 | 具体表现 | 排查方向 |
|------|---------|---------|
| GPU 利用率低，nvidia-smi 显示利用率只有 50-70% | DataLoader 来不及喂数据 | 检查 num_workers、pin_memory；检查 NUMA 绑定 |
| 多机训练比单机慢很多（排除网络后） | CPU 预处理是瓶颈 | Profile DataLoader，考虑预处理数据为 tokenized 格式 |
| 自定义 CUDA extension 中 CPU 端多线程性能差 | False Sharing 或 NUMA 不亲和 | 用 `perf` 分析 cache miss rate |

## 常见问题

**Q: 这些 CPU 层面的优化和 GPU/AI 训练有什么关系？**

A: 虽然 AI 训练的核心计算发生在 GPU 上，但数据加载（PyTorch 的 DataLoader workers）全部运行在 CPU 上，CPU 和 GPU 之间的数据传输需要经过 PCIe 总线。如果负责给某张 GPU 喂数据的 CPU 进程和该 GPU 不在同一个 NUMA 节点上，数据传输会因为跨 NUMA 访问而变慢。在大规模训练中，这类看似微小的延迟会累积成明显的吞吐瓶颈。

## 延伸阅读

- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) — Ulrich Drepper
- [NUMA-aware Programming](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Cache（缓存）** | CPU 内部的高速存储，用于暂存常用数据，减少反复访问主内存的延迟 |
| **Cache Line（缓存行）** | 缓存与主内存之间数据搬运的最小单位，通常为 64 字节 |
| **L1/L2/L3** | 三级缓存层级。L1 最快最小（每核私有），L3 最大但较慢（多核共享） |
| **False Sharing（伪共享）** | 多线程写不同变量但共享同一缓存行，触发不必要的缓存一致性开销 |
| **NUMA** | Non-Uniform Memory Access，多路服务器中 CPU 访问本地内存快、远端内存慢的架构特性 |
| **Socket** | 物理上安装一颗 CPU 芯片的插槽；多路服务器有多个 Socket |
| **QPI/UPI** | Intel CPU 插槽之间的高速互联总线 |
| **PCIe** | Peripheral Component Interconnect Express，CPU 与外设（GPU、网卡等）之间的标准高速总线 |
| **DRAM** | Dynamic Random-Access Memory，即主内存/系统内存 |
| **DataLoader** | PyTorch 中负责从磁盘读取训练数据并预处理的组件，运行在 CPU 上 |
