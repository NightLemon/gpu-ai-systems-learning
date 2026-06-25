# 01 - 计算机体系结构（HPC 视角）

> 这一章不是把大学体系结构课重讲一遍，而是补足后端工程师在做 AI/HPC 时容易跳过的几个视角：**带宽往往比算力先成瓶颈、NUMA 不感知会让性能莫名其妙地差一倍、SIMD 是 GPU SIMT 的精神祖先**。下面这些概念会在 Ch02（GPU 架构）和后续章节里反复出现，建议先打好基础再往下走。

## 本章内容

| 文件 | 主题 | 要点 |
|------|------|------|
| [01-cpu-cache-numa.md](01-cpu-cache-numa.md) | CPU Cache 与 NUMA | Cache Line、False Sharing、NUMA-aware 编程 |
| [02-memory-bandwidth.md](02-memory-bandwidth.md) | 内存带宽 | Bandwidth vs Latency、Memory-bound 概念、Roofline 模型 |
| [03-simd-vectorization.md](03-simd-vectorization.md) | SIMD 向量化 | AVX/SSE、自动向量化、与 GPU SIMT 的对比 |

## 前置知识

- 用过 C/C++，知道指针、数组、栈/堆的基本概念
- 有 Linux 系统编程基础（进程/线程/CPU 亲和性）
- **不需要**学过《计算机体系结构》课程；本章按"工程视角"组织

## 学完本章你能回答

1. 为什么"算力够大"经常解决不了真实的性能问题？什么叫 Memory-bound？
2. False Sharing 是什么？为什么它能让多线程程序比单线程还慢？
3. 一台双路服务器上，跨 NUMA 节点访问内存的代价大概是同节点的几倍？怎么用 `numactl` 避开？
4. AVX-512 一条指令能并行处理多少个 float？这种 SIMD 思想和 GPU 的 SIMT 有什么本质区别？
5. Roofline 模型是怎么用 arithmetic intensity 把 kernel 分成 compute-bound 和 memory-bound 两类的？

## 与前后章的关系

- **承接**：默认你有后端工程经验，所以跳过了基础的 ISA、流水线、分支预测等内容
- **铺垫 Ch02**：本章的 cache 层级、SIMD、Memory-bound 概念会在 Ch02 GPU 架构里直接对应到 shared memory / SIMT / HBM 带宽
- **铺垫 Ch07**：Memory-bound 是后面所有推理优化（量化、KV-Cache、FlashAttention）的出发点
