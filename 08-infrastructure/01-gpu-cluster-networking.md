# GPU 集群网络

> 大规模训练中，网络是仅次于 GPU 的第二大瓶颈。

## 核心概念

### 网络技术栈

```
应用层:    NCCL / MPI
传输层:    RDMA (Remote Direct Memory Access)
链路层:    InfiniBand / RoCE (RDMA over Converged Ethernet)
物理层:    光纤 / 铜缆

RDMA 的核心价值: 绕过 CPU 和 OS 内核，GPU 显存直接读写远端 GPU 显存
  → 延迟: ~1-2 μs (vs TCP 的 ~50-100 μs)
  → 带宽: 接近线速
```

### InfiniBand vs RoCE

| 方面 | InfiniBand | RoCE v2 |
|------|-----------|---------|
| 协议 | 专有 | 基于以太网 |
| 延迟 | ~1 μs | ~2-5 μs |
| 拥塞控制 | 硬件信用机制 | ECN + PFC |
| 带宽 | HDR 200Gb, NDR 400Gb | 100/200/400 GbE |
| 成本 | 高（专用交换机） | 中（复用以太网基础设施） |
| 适用 | 大规模 AI 训练 | 中等规模 / 混合集群 |

### GPUDirect RDMA

```
传统路径:
  GPU0 → CPU 内存 → 网卡 → 网络 → 网卡 → CPU 内存 → GPU1

GPUDirect RDMA:
  GPU0 → 网卡 → 网络 → 网卡 → GPU1
  → 完全绕过 CPU，延迟和带宽都大幅改善

GPUDirect P2P (同一节点):
  GPU0 → NVLink → GPU1  (不走 PCIe)
```

### 网络拓扑

```
典型 AI 集群拓扑 (Fat-tree):

         ┌──── Core Switch ────┐
         │                      │
    ┌── Spine ──┐         ┌── Spine ──┐
    │           │         │           │
  Leaf        Leaf      Leaf        Leaf
   │           │         │           │
 ┌─┤─┐     ┌─┤─┐     ┌─┤─┐     ┌─┤─┐
 N0 N1 N2  N3 N4 N5  N6 N7 N8  N9 ...

N = 节点 (每节点 8 GPU)
Leaf Switch: 连接同一机架的节点
Spine Switch: 连接不同机架

机架内: NVLink (900 GB/s) + InfiniBand
跨机架: InfiniBand (~25-50 GB/s per link)

关键: 尽量将通信密集的操作（如 TP）放在机内
      跨机架只做 DP gradient sync 和 PP 激活传递
```

## 常见问题

**Q: 为什么不直接用 TCP/IP？**

A: TCP/IP 涉及内核协议栈、多次数据拷贝、中断处理，延迟高 (~50μs+)、CPU 占用大。RDMA 是零拷贝、内核旁路的，延迟低 (~1μs)、不消耗 CPU。大规模训练中这个差距被放大。

**Q: 网络对训练速度的影响有多大？**

A: 在 DP 中，梯度 AllReduce 可以和计算 overlap，影响较小。在 TP 中，通信在关键路径上，带宽和延迟直接影响训练速度。一个 7B 模型用 TP=8 机内训练，NVLink 切换到 PCIe 可能慢 30-50%。

## 延伸阅读

- [NVIDIA DGX SuperPOD Architecture](https://docs.nvidia.com/dgx-superpod/)
- [InfiniBand vs RoCE](https://www.nvidia.com/en-us/networking/ethernet-switching/)
- [NCCL Network Design](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
