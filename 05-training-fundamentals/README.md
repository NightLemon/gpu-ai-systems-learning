# 05 - 训练基础

> 写到这一章，你已经知道 GPU 长什么样、能跑多快，也知道 Transformer 的算力和显存账。本章把这些知识落到**单机/单卡训练工程的基本功上**——混合精度怎么选、显存不够时如何用计算换、数据加载为什么经常成为隐形瓶颈、profiling 怎么帮你定位真正的问题。这些技能是后面分布式训练和推理优化的基础——所有大规模训练都是单卡训练流程被扩到 N 卡，但单卡上的低效会被并行系统线性放大。

## 本章内容

| 文件 | 主题 | 要点 |
|------|------|------|
| [01-mixed-precision.md](01-mixed-precision.md) | 混合精度训练 | AMP、FP16/BF16/FP8、Loss Scaling、什么时候用哪种 |
| [02-gradient-checkpointing.md](02-gradient-checkpointing.md) | 梯度检查点 | 时间换空间、O(√n) 激活显存、Selective Checkpoint |
| [03-data-loading.md](03-data-loading.md) | 数据加载 | DataLoader 优化、Streaming Dataset、num_workers 选择 |
| [04-profiling.md](04-profiling.md) | 性能分析 | PyTorch Profiler、Nsight Systems、MFU 估算 |
| [05-pytorch-internals.md](05-pytorch-internals.md) | PyTorch 内部机制 | torch.compile、Autograd、Dispatcher、显存管理 |

## 前置知识

- 用 PyTorch 写过完整的训练 loop（forward / loss / backward / optimizer.step）
- 看完 [Ch02](../02-gpu-architecture/README.md) 和 [Ch04](../04-transformer-deep-dive/README.md)，知道 Tensor Core 算力和训练显存账（16P）
- 知道 GPU 上 BF16/FP16 Tensor Core 远快于 FP32，但不一定理解为什么默认还经常用 FP32

## 学完本章你能回答

1. BF16 和 FP16 哪个更适合训练，为什么 BF16 几乎不需要 Loss Scaling？
2. 单卡 OOM 又不能换卡时，Gradient Checkpointing 能省多少显存、代价多大？怎么 Selective 选关键层 checkpoint？
3. 训练时 GPU 利用率忽高忽低，怎么用 PyTorch Profiler / Nsight 区分是数据加载、kernel 启动、还是 H2D 拷贝是瓶颈？
4. MFU 该怎么算？为什么用 H100 dense FP16 989 TFLOPS（而不是 sparse 1979）作分母才合理？
5. `torch.compile` 在什么场景下能给训练带来收益，又有哪些坑（动态 shape、graph break）？

## 与前后章的关系

- **承接 Ch04**：本章的所有显存优化都建立在 Ch04 的 16P bytes 显存账之上
- **铺垫 Ch06**：本章是单卡视角，Ch06 把这套技能扩展到多卡（DDP/FSDP/ZeRO）；单卡上 profiling 不到位，并行系统会把问题放大 N 倍
- **铺垫 Ch07**：本章的混合精度、profiling 工具也是推理优化分析的基础
