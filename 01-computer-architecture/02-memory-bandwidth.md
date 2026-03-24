# 内存带宽

> 在高性能计算和 AI 系统中，**内存带宽**（单位时间内能传输的数据总量）往往比算力更先成为性能瓶颈。

## 核心概念

### Bandwidth（带宽）vs Latency（延迟）

```
Latency（延迟）: 从发出数据请求到收到第一个字节所需的等待时间
Bandwidth（带宽）: 单位时间内能传输的数据总量

类比——水管:
  Latency = 打开水龙头后，水流出来需要等多久
  Bandwidth = 水管的粗细，决定了单位时间能流过多少水
```

两者都重要但含义不同。AI 推理中的主要瓶颈通常是带宽，而非延迟。

### 为什么带宽对 AI 如此重要

以一个 70 亿参数（7B）的模型为例，在 FP16 精度下推理：

```
模型权重大小: 7B × 2 bytes = 14 GB
在自回归生成（逐 token 输出）中，每生成 1 个 token 都需要读取整个模型的权重

CPU 主内存带宽 (DDR5):  ~100 GB/s → 读 14 GB 需要 140 ms/token
GPU 显存带宽 (HBM3):    ~3350 GB/s → 读 14 GB 需要 4.2 ms/token

→ 带宽直接决定了推理每生成一个 token 的速度上限
```

### 衡量与测试

```bash
# 用 STREAM benchmark 测量 CPU 内存带宽
# git clone https://github.com/jeffhammond/STREAM && cd STREAM
gcc -O3 -march=native -fopenmp stream.c -o stream
OMP_NUM_THREADS=16 ./stream

# 用 NVIDIA bandwidthTest 测量 GPU 显存带宽和 CPU↔GPU 传输带宽
# （包含在 CUDA Samples 中）
./bandwidthTest
```

### Arithmetic Intensity（算术强度）与 Roofline Model（屋顶线模型）

Arithmetic Intensity 衡量的是一段计算中"做了多少运算"与"搬了多少数据"的比值：

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs（浮点运算次数）}}{\text{Bytes Accessed（访问的字节数）}}$$

- **AI 低 → Memory-bound（带宽瓶颈）**：计算量不大但需要搬运大量数据，性能受带宽限制。大部分 LLM 推理操作（如 LayerNorm、Softmax、逐元素运算）属于此类。
- **AI 高 → Compute-bound（算力瓶颈）**：计算密集但数据复用率高，性能受算力限制。训练中的矩阵乘法（GEMM）通常属于此类。

Roofline Model 是一种可视化工具，把带宽上限和算力上限画在同一张图上，帮助判断某个 kernel 的瓶颈在哪里（详见 Ch02 GPU vs CPU 章节）。

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
