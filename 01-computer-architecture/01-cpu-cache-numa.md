# CPU Cache 与 NUMA

> HPC（高性能计算）视角下 CPU Cache（缓存）和 NUMA（非一致性内存访问）的关键优化点。本节假设你了解基本的 CPU 架构，重点补充与高性能计算和 AI 训练相关的细节。

## 核心概念

### Cache 层级回顾

现代 CPU 通过多级缓存来弥补处理器速度与主内存（DRAM）速度之间的差距。数据从内存加载后，会被逐级缓存在离 CPU 核心更近的存储中，以加快后续访问：

```
CPU Core → L1 (64KB, ~1ns) → L2 (1MB, ~3ns) → L3 (多核共享, 30MB+, ~10ns) → DRAM (~100ns)
           ↑ 最快最小        ↑ 较快较大       ↑ 所有核心共享               ↑ 容量最大但最慢
```

### HPC 关键点

**Cache Line（缓存行，64 bytes）** 是 CPU 与内存之间数据搬运的最小单位。即使你只读 1 个 int（4 bytes），CPU 也会把它所在的整行 64 bytes 一起加载。因此，**让连续使用的数据排列在相邻的内存地址上**（即良好的"空间局部性"），可以大幅提升缓存命中率。

**False Sharing（伪共享）**：当多个线程各自写不同的变量，但这些变量恰好位于同一条 Cache Line 中时，每次写操作都会触发缓存一致性协议（让其他核心的对应 Cache Line 失效），导致性能严重下降——即使各线程在逻辑上完全独立。

```c
// ❌ False Sharing: 多个线程的 counter 紧挨着，落在同一条 cache line 中
struct { int counter[NUM_THREADS]; } shared;

// ✅ Padding 避免: 用 alignas(64) 让每个 counter 独占一条 cache line
struct { alignas(64) int counter; } per_thread[NUM_THREADS];
```

### NUMA (Non-Uniform Memory Access，非一致性内存访问)

在多路服务器（有多个 CPU 插槽/Socket）中，每个 Socket 有自己直连的本地内存。CPU 访问本地内存最快；访问另一个 Socket 的远端内存需要经过插槽间互联总线（如 Intel 的 QPI/UPI），延迟更高：

```
多 socket 服务器:
  Socket 0 ←→ Local DRAM 0     (~100ns 本地访问)
       ↕ QPI/UPI 互联 (~150ns 跨插槽访问)
  Socket 1 ←→ Local DRAM 1     (~100ns 本地访问)

访问远端内存 = 本地延迟 + 互联延迟 → 比本地慢约 1.5-2x
```

HPC/AI 训练服务器通常是双路或四路配置。GPU 通过 PCIe 总线连接到特定的 CPU 插槽，因此也属于特定的 NUMA 节点：

```bash
# 查看服务器的 NUMA 拓扑（哪些 CPU 核心和内存属于哪个节点）
numactl --hardware
# 将进程绑定到 NUMA 节点 0（让 CPU 进程和它服务的 GPU 在同一 NUMA 域）
numactl --cpunodebind=0 --membind=0 ./my_training_script
```

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
