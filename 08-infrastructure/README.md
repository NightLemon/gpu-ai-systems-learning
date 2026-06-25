# 08 - 基础设施

> 走到这一章，前面的训练/推理优化都假设了"集群已经能正常工作"。但现实中训练任务挂掉、性能莫名变差、节点排障，**真正的成本和事故都集中在基础设施层**。这一章的视角是"AI 工程师必须懂的最小基础设施集"——你的后端经验在这里最有用：网络、存储、调度、监控、排障，都是熟悉的话题，只是 GPU 集群的工程权衡不一样。本章配套了一份"踩坑案例集"（06）和一份"高级调试手册"（05），都是真实生产中代价不菲的教训。

## 本章内容

| 文件 | 主题 | 要点 |
|------|------|------|
| [01-gpu-cluster-networking.md](01-gpu-cluster-networking.md) | 集群网络 | IB vs RoCE、拓扑映射 TP/PP/DP/EP、PFC/ECN、NCCL 调优 |
| [02-storage-systems.md](02-storage-systems.md) | 存储系统 | 数据路径 vs Checkpoint 设计、异步保存、一致性、恢复 |
| [03-kubernetes-gpu.md](03-kubernetes-gpu.md) | K8s GPU 调度 | GPU Operator、Gang Scheduling、拓扑感知、K8s vs Slurm |
| [04-monitoring.md](04-monitoring.md) | 监控与排障 | 系统/训练指标分层、DCGM、排障流程图、故障案例 |
| [05-debugging-runbook.md](05-debugging-runbook.md) | 高级调试手册 | NCCL 调试、训练挂起诊断、CUDA 调试工具、故障模式 |
| [06-production-pitfalls.md](06-production-pitfalls.md) | 实战踩坑案例 | NGC 镜像版本、驱动/CUDA、CDI、MIG、RDMA、NCCL、Operator 升级 |

## 前置知识

- 后端工程经验：用过 K8s / Docker / Linux 网络栈 / 分布式存储
- 看过 [Ch06](../06-distributed-training/README.md)，知道训练为什么需要 AllReduce/All-to-All、为什么对网络敏感
- **不需要**懂 RDMA verbs API 等底层细节；本章关注工程权衡和排障思路

## 学完本章你能回答

1. 64 卡或 256 卡集群里，TP/PP/DP/EP 怎么映射到机内 NVLink、机架内 IB、跨机架三层网络？
2. Sharded checkpoint / full checkpoint / async checkpoint 有什么区别？大模型恢复时瓶颈在哪？为什么 `dcp.save()` 不等于异步？
3. K8s GPU 调度为什么必须 gang scheduling？为什么 topology-aware 不只是锦上添花？K8s 和 Slurm 在训练场景下怎么取舍？
4. 系统指标（GPU util、NVLink、IB 带宽）和训练指标（step time、MFU、loss、tokens/s）该怎么分层监控？训练慢了的排障流程是什么？
5. 为什么不应该把 `nvcr.io/nvidia/pytorch:24.01-py3` 当默认模板？版本组合（驱动 / CUDA / 镜像 / 框架）出问题时怎么定位？
6. NCCL 卡住、训练突然挂、IB 带宽掉一半——拿到这种现象时第一步该看什么？

## 与前后章的关系

- **承接 Ch06**：Ch06 讲了为什么分布式训练需要这些通信原语；本章讲承载这些通信的硬件和系统怎么落地
- **承接 Ch07**：Ch07 讲了推理框架，本章给出推理服务的 K8s 部署和监控视角
- **铺垫 Ch09**：本章的工具链（DCGM、Nsight、NCCL trace）是 Ch09 实战项目里测性能、做对比实验的基础
