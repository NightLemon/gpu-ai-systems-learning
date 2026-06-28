# GPU & AI Systems 学习指南

> 面向有后端经验的工程师，系统性掌握大模型训练与推理优化的核心知识。

## 📋 目录

| 章节 | 主题 | 难度 | 说明 |
|------|------|------|------|
| [00-industry-overview](00-industry-overview/README.md) | LLM 行业全景 | ⭐ | 数据规模、团队结构、成本与商业模型 |
| [01-computer-architecture](01-computer-architecture/README.md) | 计算机体系结构 | ⭐⭐ | 精简回顾 + HPC 视角补充 |
| [02-gpu-architecture](02-gpu-architecture/README.md) | GPU 架构 | ⭐⭐⭐ | **重点** — 理解 GPU 硬件是一切优化的基础 |
| [03-cuda-programming](03-cuda-programming/README.md) | CUDA 编程 | ⭐⭐⭐ | **重点** — 从入门到 GEMM 优化实战 |
| [04-transformer-deep-dive](04-transformer-deep-dive/README.md) | Transformer 深入 | ⭐⭐ | 模型架构 + 计算/显存分析 |
| [05-training-fundamentals](05-training-fundamentals/README.md) | 训练基础 | ⭐⭐ | 混合精度、Checkpointing、Profiling |
| [06-distributed-training](06-distributed-training/README.md) | 分布式训练 | ⭐⭐⭐ | **重点** — 通信原语到多机训练框架 |
| [07-inference-optimization](07-inference-optimization/README.md) | 推理优化 | ⭐⭐⭐ | **重点** — KV-cache 到 vLLM/TensorRT-LLM |
| [08-infrastructure](08-infrastructure/README.md) | 基础设施 | ⭐⭐ | 集群网络、存储、K8s GPU 调度 |
| [09-hands-on-projects](09-hands-on-projects/README.md) | 实战项目 | ⭐⭐⭐ | 动手练习指南 |
| [resources](resources/version-baseline.md) | 资源汇总 | — | 版本基线、书籍、论文、课程、工具 |

> 版本说明：容易过期的框架接口、CLI 和性能结论以 [版本基线](resources/version-baseline.md) 为准；部署前请再次核对官方文档和当前版本的 `--help` 输出。

## 🗺️ 学习路线图

```mermaid
graph LR
    Z[00 行业全景] --> A[01 体系结构]
    A --> B[02 GPU 架构]
    B --> C[03 CUDA 编程]
    B --> D[04 Transformer]
    D --> E[05 训练基础]
    C --> F[06 分布式训练]
    C --> G[07 推理优化]
    E --> F
    D --> G
    F --> H[08 基础设施]
    G --> H
    H --> I[09 实战项目]
```

## 📅 按周学习计划（建议 12 周）

### Phase 1：基础（Week 1-3）

- [ ] **Week 1**：Ch01 体系结构回顾 + Ch02 GPU 架构（GPU vs CPU、SM 架构、显存层级）
- [ ] **Week 2**：Ch03 CUDA 编程基础（kernel、内存管理、shared memory）
- [ ] **Week 3**：Ch03 CUDA 优化技巧 + GEMM Case Study + Triton 入门

### Phase 2：模型与训练（Week 4-6）

- [ ] **Week 4**：Ch04 Transformer 深入（Attention、FLOPs/显存分析、常见架构）
- [ ] **Week 5**：Ch05 训练基础（混合精度、Gradient Checkpointing、Profiling）
- [ ] **Week 6**：Ch06 分布式训练（通信原语、Data Parallel — DDP/FSDP/ZeRO）

### Phase 3：分布式与推理（Week 7-9）

- [ ] **Week 7**：Ch06 分布式训练（Tensor/Pipeline/Expert Parallel）
- [ ] **Week 8**：Ch06 DeepSpeed & Megatron-LM 实战
- [ ] **Week 9**：Ch07 推理优化（KV-cache、量化、Continuous Batching）

### Phase 4：高级推理与实战（Week 10-12）

- [ ] **Week 10**：Ch07 高级推理（Speculative Decoding、FlashAttention、vLLM、TensorRT-LLM、框架选型）
- [ ] **Week 11**：Ch08 基础设施（集群网络、存储、K8s GPU、监控、排障与生产踩坑）
- [ ] **Week 12**：Ch09 实战项目（选择 1-2 个动手完成）

> Phase 3-4 边界说明：Week 9 把 Ch07 前半段（KV-cache、量化、Continuous Batching）吃掉，Week 10 集中收尾 Ch07 后半段并做框架选型。Ch07/Ch08 都偏厚，可以按需精读：先把对工作直接相关的小节看透，回到剩下的留作 reference。

## 🔍 我想了解 X，该看哪里？

按主题反向索引，不必从头读完整章。

| 你想搞清楚的问题 | 直接看这里 |
|------------------|-----------|
| 训练一个 7B/70B 模型大概要烧多少卡天/美元 | [00/03-economics.md](00-industry-overview/03-economics.md)、[04/02-computation-analysis.md](04-transformer-deep-dive/02-computation-analysis.md) |
| 训练显存账（参数 + 梯度 + 优化器 + 激活）怎么算 | [04/02-computation-analysis.md](04-transformer-deep-dive/02-computation-analysis.md) |
| KV-Cache 大小怎么算、GQA 为什么省 | [04/01-attention-mechanism.md](04-transformer-deep-dive/01-attention-mechanism.md)、[07/01-kv-cache.md](07-inference-optimization/01-kv-cache.md) |
| BF16 vs FP16 vs FP8，混合精度怎么选 | [05/01-mixed-precision.md](05-training-fundamentals/01-mixed-precision.md) |
| GEMM 优化是怎么从朴素一步步做到接近 cuBLAS 的 | [03/05-gemm-case-study.md](03-cuda-programming/05-gemm-case-study.md) |
| FlashAttention 为什么同时省显存又快 | [07/05-flash-attention.md](07-inference-optimization/05-flash-attention.md) |
| DDP / FSDP / ZeRO-1/2/3 啥关系，怎么选 | [06/02-data-parallel.md](06-distributed-training/02-data-parallel.md)、[06/08-training-strategy-guide.md](06-distributed-training/08-training-strategy-guide.md) |
| TP / PP / EP 通信量公式 | [06/03-tensor-parallel.md](06-distributed-training/03-tensor-parallel.md)、[06/04-pipeline-parallel.md](06-distributed-training/04-pipeline-parallel.md)、[06/05-expert-parallel.md](06-distributed-training/05-expert-parallel.md) |
| 64/256 卡集群里 TP/PP/DP/EP 怎么映射到机内/机架/跨机架 | [08/01-gpu-cluster-networking.md](08-infrastructure/01-gpu-cluster-networking.md) |
| vLLM / TensorRT-LLM / SGLang / TGI 怎么选 | [07/08-inference-framework-guide.md](07-inference-optimization/08-inference-framework-guide.md) |
| INT8 / INT4 / FP8 量化能加速多少、精度损失 | [07/02-quantization.md](07-inference-optimization/02-quantization.md) |
| Continuous Batching、PagedAttention、Speculative Decoding 原理 | [07/03-continuous-batching.md](07-inference-optimization/03-continuous-batching.md)、[07/06-vllm-architecture.md](07-inference-optimization/06-vllm-architecture.md)、[07/04-speculative-decoding.md](07-inference-optimization/04-speculative-decoding.md) |
| 训练突然挂、NCCL 卡死、IB 带宽掉一半，先看什么 | [08/04-monitoring.md](08-infrastructure/04-monitoring.md)、[08/05-debugging-runbook.md](08-infrastructure/05-debugging-runbook.md)、[08/06-production-pitfalls.md](08-infrastructure/06-production-pitfalls.md) |
| Checkpoint 怎么省时间 / async / 一致性 | [08/02-storage-systems.md](08-infrastructure/02-storage-systems.md) |
| MFU 怎么算、为什么用 dense 而非 sparse 峰值 | [05/04-profiling.md](05-training-fundamentals/04-profiling.md)、[08/04-monitoring.md](08-infrastructure/04-monitoring.md) |
| 该用哪个版本的 PyTorch / vLLM / NGC 镜像 | [resources/version-baseline.md](resources/version-baseline.md) |

## 🎯 前置假设

本仓库假设你已具备：

- **Linux 系统编程**：熟悉进程/线程模型、内存管理、`perf` 等工具
- **C/C++ 编程**：能读写中等复杂度的 C++ 代码
- **Python & PyTorch 基础**：能用 PyTorch 搭建简单模型并训练
- **后端工程经验**：了解分布式系统、网络、存储、容器化部署

## 📖 如何使用

1. **按章节顺序阅读**：每个章节内的文件有序号，建议按序阅读
2. **关注 Mermaid 图**：架构图和流程图用 Mermaid 绘制，GitHub 原生支持渲染
3. **动手实践**：关键章节附有代码或伪代码示例，建议边读边试
4. **打勾跟踪**：用上方的学习计划 checkbox 跟踪进度（需要 fork 后在自己的 repo 打勾）
5. **手机友好**：所有内容针对 GitHub 移动端阅读优化

## License

[MIT](https://github.com/NightLemon/gpu-ai-systems-learning/blob/master/LICENSE)
