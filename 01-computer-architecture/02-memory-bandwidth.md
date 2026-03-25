# 内存带宽

> 在高性能计算和 AI 系统中，**内存带宽**（单位时间内能传输的数据总量）往往比算力更先成为性能瓶颈。

## 一个反直觉的事实

很多人第一反应是"GPU 算力不够导致推理慢"。但实际上，当你部署一个大语言模型做在线服务时，**GPU 的计算单元大部分时间都在闲着——它们在等数据从显存搬过来**。

为什么？因为大模型推理的核心操作是"读取模型权重 → 做一次矩阵向量乘 → 输出一个 token"。权重有几十 GB，而每个 token 的计算量相对不大。GPU 的计算速度远快于显存的搬运速度，所以计算单元总是在等数据。

这就是"Memory-bound"的含义——**瓶颈不在算力，而在带宽**。理解这一点，是理解后面所有推理优化（量化、KV-Cache、FlashAttention）的基础。

## 核心概念

### Bandwidth（带宽）vs Latency（延迟）

这两个概念经常被混淆，但它们描述的是内存系统的两个不同维度：

```
Latency（延迟）: 你向内存"要一个数据"，从发出请求到拿到第一个字节的等待时间
  → 衡量的是"响应有多快"

Bandwidth（带宽）: 数据通道每秒能传输的总量
  → 衡量的是"通道有多宽"

类比——网购快递:
  Latency = 下单到收到第一个包裹的时间
  Bandwidth = 快递公司每天能送几吨货
```

对于 LLM 推理（逐 token 生成），瓶颈通常在带宽而非延迟。因为每个 token 需要读取整个模型权重（GB 级别），数据量大但计算少。

### 为什么带宽对 AI 如此重要：一笔帐

以一个 70 亿参数（7B）的模型为例，在 FP16 精度下做推理：

```
模型权重大小: 7B 参数 × 2 bytes/参数 = 14 GB
自回归生成时，每生成 1 个 token 都需要将整个模型权重从显存读到计算单元

如果用 CPU 推理:
  DDR5 带宽 ~100 GB/s  → 读 14 GB 需要 140 ms → 每秒只能生成 ~7 tokens

如果用 GPU 推理:
  H100 HBM3 带宽 ~3350 GB/s → 读 14 GB 需要 4.2 ms → 每秒可生成 ~240 tokens

→ 这还只是理论上限——实际会更低（因为 KV-Cache 读取、kernel launch 等额外开销）
→ 带宽直接决定了"这个模型在这张卡上最快能跑多快"
```

这也解释了为什么量化（FP16→INT4）能让推理速度翻 2-3 倍：权重体积从 14GB 降到 3.5GB，需要搬运的数据少了 4 倍，带宽消耗也少了 4 倍。

### 衡量与测试

你拿到一台服务器后，首先应该做的事之一就是**确认实际带宽是否达到理论值**。如果达不到，可能是硬件配置问题（ECC 设置、BIOS 选项等）。

```bash
# CPU 内存带宽测试：STREAM benchmark
# 这是 HPC 领域测内存带宽的标准工具
# git clone https://github.com/jeffhammond/STREAM && cd STREAM
gcc -O3 -march=native -fopenmp stream.c -o stream
OMP_NUM_THREADS=16 ./stream
# 看 "Triad" 结果，DDR5 双路服务器应该在 200+ GB/s

# GPU 显存带宽测试：NVIDIA bandwidthTest (CUDA Samples 中自带)
./bandwidthTest
# 看 "Device to Device Bandwidth"，H100 应该在 3000+ GB/s
```

### Arithmetic Intensity：判断瓶颈在哪的关键指标

当你发现一个 kernel 跑得慢时，第一个要问的问题是："它是算力不够，还是带宽不够？"

**Arithmetic Intensity（算术强度）** 就是用来回答这个问题的。它衡量的是"每搬运 1 字节数据，能做多少次浮点运算"：

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs（浮点运算次数）}}{\text{Bytes Accessed（访问的字节数）}}$$

- **AI 低 → Memory-bound（带宽瓶颈）**：每搬一堆数据只做了很少的运算。GPU 计算单元大部分时间在等数据。大部分 LLM 推理操作（LayerNorm、Softmax、逐元素运算、decode 阶段的 Attention）都属于此类。
- **AI 高 → Compute-bound（算力瓶颈）**：数据搬进来后被反复使用，计算单元满负荷运转。训练中的大矩阵乘法（GEMM）通常属于此类。

**H100 的临界点**：H100 的 BF16 算力 ~1979 TFLOPS / HBM 带宽 3350 GB/s ≈ **590 FLOPs/Byte**。这意味着如果你的 kernel 每读 1 byte 做不到 590 次运算，它就是 memory-bound 的。实际上大部分操作都远低于这个数字——所以**优化 AI 系统在很大程度上就是在优化内存访问**。

### Roofline Model：一张图看清瓶颈

Roofline Model（屋顶线模型）把上面的概念可视化了——在一张图上同时画出带宽上限和算力上限：

```
         性能 (FLOPS)
           │
  算力上限 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
           │                 ╱─────────── Compute-bound 区域
           │               ╱              （优化方向：提高计算效率）
           │             ╱
           │           ╱  ← 带宽上限斜线（斜率 = 内存带宽）
           │         ╱
           │       ╱    Memory-bound 区域
           │     ╱      （优化方向：减少内存访问）
           │   ╱
           │ ╱
           └──────────────────────── Arithmetic Intensity
```

你的 kernel 落在图的哪个区域，就决定了优化的方向：
- 在斜线左侧（Memory-bound）→ 减少内存访问：用 Shared Memory 缓存、合并访存、算子融合
- 在水平线下方（Compute-bound）→ 提高计算效率：用 Tensor Core、提高 occupancy

Nsight Compute（ncu）可以自动为你画 Roofline 图，帮你定位 kernel 瓶颈。

## 延伸阅读

- [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)
- [STREAM Benchmark](https://www.cs.virginia.edu/stream/)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Bandwidth（带宽）** | 单位时间内能传输的数据总量，通常以 GB/s 为单位 |
| **Latency（延迟）** | 从发出请求到收到第一个数据的等待时间，通常以 ns 或 ms 为单位 |
| **DDR5** | 第五代双倍数据率同步动态随机存取存储器，即当前主流的 CPU 主内存技术 |
| **HBM（High Bandwidth Memory）** | 高带宽显存，通过 3D 堆叠和超宽总线实现极高带宽，用于数据中心 GPU（如 A100/H100） |
| **FP16** | 16 位浮点数格式，每个数占 2 字节，是 AI 推理中常用的精度 |
| **FLOPS** | Floating Point Operations Per Second，每秒浮点运算次数，衡量算力的单位 |
| **Memory-bound** | 性能瓶颈在内存/显存带宽上，CPU/GPU 的计算单元在等数据 |
| **Compute-bound** | 性能瓶颈在计算能力上，数据已经准备好但计算单元处理不过来 |
| **Arithmetic Intensity（算术强度）** | 每字节数据访问对应的浮点运算次数，用于判断一段计算是带宽瓶颈还是算力瓶颈 |
| **Roofline Model（屋顶线模型）** | 一种性能分析框架，将带宽上限和算力上限可视化，帮助定位 kernel 瓶颈 |
| **GEMM** | General Matrix Multiply，通用矩阵乘法，深度学习中最核心的计算操作 |
| **LLM** | Large Language Model，大语言模型，如 GPT、LLaMA 等 |
