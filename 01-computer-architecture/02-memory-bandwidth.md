# 内存带宽

> 在 HPC 和 AI 中，**内存带宽**往往比算力更先成为瓶颈。

## 核心概念

### Bandwidth vs Latency

```
Latency: 发出请求到收到第一个字节的时间
Bandwidth: 单位时间内传输的数据总量

类比: 水管
  Latency = 打开龙头到水流出的时间
  Bandwidth = 水管粗细（单位时间水量）
```

### 为什么带宽重要

```
7B 模型推理 (FP16):
  模型权重: 14 GB
  生成 1 个 token: 必须读 14 GB 权重
  
  DDR5 (CPU): ~100 GB/s → 14/100 = 140 ms/token
  HBM3 (GPU): ~3350 GB/s → 14/3350 = 4.2 ms/token
  
  → 带宽决定了推理速度的上限
```

### 衡量与测试

```bash
# Linux 上用 STREAM benchmark 测 CPU 内存带宽
# git clone https://github.com/jeffhammond/STREAM && cd STREAM
gcc -O3 -march=native -fopenmp stream.c -o stream
OMP_NUM_THREADS=16 ./stream

# GPU 带宽用 NVIDIA bandwidthTest
# (CUDA Samples 中自带)
./bandwidthTest
```

### Arithmetic Intensity 与 Roofline

$$\text{Arithmetic Intensity (AI)} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

- AI 低 → Memory-bound（优化带宽利用）
- AI 高 → Compute-bound（优化计算效率）

大多数 LLM 推理操作（LayerNorm, Softmax, Element-wise）都是 memory-bound。训练中的 GEMM 通常是 compute-bound。

## 延伸阅读

- [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)
- [STREAM Benchmark](https://www.cs.virginia.edu/stream/)
