# CPU Cache 与 NUMA

> HPC 视角下 CPU Cache 和 NUMA 的关键优化点。

## 核心概念

### Cache 层级回顾

```
CPU Core → L1 (64KB, ~1ns) → L2 (1MB, ~3ns) → L3 (共享, 30MB+, ~10ns) → DRAM (~100ns)
```

### HPC 关键点

**Cache Line（64 bytes）** 是数据搬运的最小单位。优化数据布局使得同一 Cache Line 中的数据会一起被使用。

**False Sharing**：多线程写不同变量但它们落在同一 Cache Line → 触发 Cache 一致性协议 → 性能暴跌。

```c
// ❌ False sharing
struct { int counter[NUM_THREADS]; } shared;  // 相邻 int 在同一 cache line

// ✅ Padding 避免
struct { alignas(64) int counter; } per_thread[NUM_THREADS];
```

### NUMA (Non-Uniform Memory Access)

```
多 socket 服务器:
  Socket 0 ←→ Local DRAM 0     (~100ns)
       ↕ QPI/UPI (~150ns)
  Socket 1 ←→ Local DRAM 1     (~100ns)

访问远端内存 = 本地延迟 + 互联延迟 → ~1.5-2x 慢
```

HPC/AI 训练机器通常是双 socket 或四 socket。GPU 也挂在特定 NUMA 节点上：

```bash
# 查看 NUMA 拓扑
numactl --hardware
# 绑定进程到特定 NUMA 节点（让 CPU 和 GPU 在同一 NUMA 域）
numactl --cpunodebind=0 --membind=0 ./my_training_script
```

## 常见问题

**Q: 这些 CPU 优化和 GPU/AI 训练有什么关系？**

A: 数据加载（DataLoader workers）在 CPU 上运行，CPU↔GPU 的数据传输路径经过 NUMA。如果 DataLoader 的 CPU 进程绑定在远端 NUMA 节点，PCIe 传输会变慢。大规模训练中这些细节影响整体吞吐。

## 延伸阅读

- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) — Ulrich Drepper
- [NUMA-aware Programming](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html)
