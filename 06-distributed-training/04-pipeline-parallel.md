# 流水线并行（Pipeline Parallelism）

> 将模型按层切分到不同的设备（或设备组）上。每个设备只负责若干层的计算，数据像流水线一样依次经过各个设备。代价是存在“bubble”（流水线空泡，部分设备空闲等待），但通信量小，适合跨机场景。

## 与 TP 的关键区别

TP 是在“同一层内部”切分计算，每层都要通信，所以需要极高带宽的 NVLink。而 PP 是在“层之间”切分，只在相邻 stage 的边界传递一次激活值/梯度，通信量小得多。所以 PP 可以放在 InfiniBand 连接的跨机场景。

PP 的主要代价是 **bubble**：当第一个 stage 还在做 forward 时，最后一个 stage 无事可做。通过将 batch 拆成多个 microbatch 的流水线可以显著减少 bubble——microbatch 越多，流水线各 stage 的重叠越充分，bubble 越小。

## 核心概念

### 基本思想

```
模型有 L 层，切分到 P 个 stage（设备/设备组）:

Stage 0 (GPU 0): Layer 0-7    → forward → 发送激活值 →
Stage 1 (GPU 1): Layer 8-15   → forward → 发送激活值 →
Stage 2 (GPU 2): Layer 16-23  → forward → 发送激活值 →
Stage 3 (GPU 3): Layer 24-31  → forward → 计算 loss → backward →

每个 stage 之间只需要传递激活值/梯度 → 通信量 = batch_size × seq_len × hidden_size
比 AllReduce 整个模型的梯度少得多
```

### Bubble 问题

朴素流水线（一个 microbatch）有严重的 bubble：

```
Time →
Stage 0: [F] . . . [B] . . .
Stage 1: . [F] . . . [B] . .
Stage 2: . . [F] . . . [B] .
Stage 3: . . . [F] [B] . . .

F = Forward, B = Backward
. = 空闲（Bubble）

Bubble 比例 = (P-1) / (P-1+1) = 很高!  (P=4 时 75% 空闲)
```

### GPipe：用 Microbatch 减少 Bubble

将一个 batch 分成 M 个 microbatch：

```
M=4 microbatches, P=4 stages:

Time →
Stage 0: [F1][F2][F3][F4] . . . . . . [B4][B3][B2][B1]
Stage 1: . [F1][F2][F3][F4] . . . . . [B4][B3][B2][B1] .
Stage 2: . . [F1][F2][F3][F4] . . . [B4][B3][B2][B1] . .
Stage 3: . . . [F1][F2][F3][F4] [B4][B3][B2][B1] . . .

Bubble 比例 = (P-1) / (P-1+M)
当 M >> P 时，Bubble → 0
```

$$\text{Bubble ratio} = \frac{P - 1}{P - 1 + M}$$

例：P=8, M=32 → Bubble = 7/39 ≈ 18%
例：P=8, M=64 → Bubble = 7/71 ≈ 10%

**GPipe 的问题**：需要缓存所有 microbatch 的激活值（forward 到 backward 之间），显存压力大。

### 1F1B (One Forward One Backward)

```
交替执行 forward 和 backward，减少激活值缓存:

P=4, M=8:
Stage 0: [F1][F2][F3][F4] [B1][F5][B2][F6][B3][F7][B4][F8] [B5][B6][B7][B8]
Stage 1: . [F1][F2][F3] [F4][B1][F5][B2][F6][B3][F7][B4] [F8][B5][B6][B7][B8]
...

稳态阶段: 一次 forward + 一次 backward 交替
→ 最多只需缓存 P 个 microbatch 的激活值（vs GPipe 的 M 个）
→ 显存节省: M/P 倍
```

### Interleaved Pipeline (Megatron-LM)

将每个 stage 分配**不连续**的层，进一步减少 bubble：

```
标准：每个 stage 分配连续的层
  Stage 0: Layer [0,1,2,3]
  Stage 1: Layer [4,5,6,7]

交错：每个 stage 分配不连续的层（虚拟 stage）
  Stage 0: Layer [0,1] + Layer [8,9]   ← 2 个虚拟 stage
  Stage 1: Layer [2,3] + Layer [10,11]
  Stage 2: Layer [4,5] + Layer [12,13]
  Stage 3: Layer [6,7] + Layer [14,15]
```

$$\text{Interleaved Bubble ratio} = \frac{P - 1}{(P - 1 + M) \times V}$$

其中 V 是虚拟 stage 数。Bubble 减少 V 倍，但通信量增加 V 倍。

## 关键细节

### 通信量分析

```
Pipeline Parallel 的通信 = Stage 之间传递 激活值/梯度

每次: batch_size × seq_len × hidden_size × dtype_size
  - 7B 模型 (h=4096): 32 × 2048 × 4096 × 2 = 512 MB per microbatch

对比 TP 的 AllReduce:
  - TP 每层 2 次 AllReduce
  - PP 每个 stage 边界 1 次点对点通信（P2P）

PP 通信量 << TP 通信量 → PP 适合跨机 (高延迟但低带宽需求)
```

### Pipeline Parallel vs Tensor Parallel

| 方面 | Tensor Parallel | Pipeline Parallel |
|------|----------------|-------------------|
| 切分粒度 | 算子内（矩阵列/行） | 层间 |
| 通信模式 | AllReduce（全对全） | P2P（点对点） |
| 通信频率 | 每层 2 次 | 仅在 stage 边界 |
| 通信量 | 大（和 hidden_size 成正比） | 中等 |
| 延迟敏感 | 高（在计算关键路径上） | 低（可用 microbatch 隐藏） |
| Bubble | 无 | 有（P-1 个 microbatch） |
| 适合场景 | **机内** NVLink | **跨机** InfiniBand |

### 显存分析

```
PP 的显存优势:
  每个 stage 只存 L/P 层的参数、梯度、优化器状态
  + 需要缓存的激活值（1F1B: P 个 microbatch 的激活值）

显存/stage ≈ (模型状态) / P + (激活值缓存)
```

## 常见问题

**Q: PP 和 TP 能组合使用吗？怎么组合？**

A: 标准做法是 **3D 并行**（Megatron-LM 的核心策略）：
- TP 放在机内（需要 NVLink）
- PP 放在跨机（容忍 InfiniBand 延迟）
- DP 放在最外层

例：128 卡 = 8 机 × 16 卡/机，TP=8 × PP=2 × DP=8

**Q: Bubble 能完全消除吗？**

A: 理论上不能完全消除（除非 M → ∞），但有些新方法接近零 bubble：
- **Zero Bubble Pipeline**（Qi et al., 2023）通过重排 F/B/W（将 backward 拆分为计算梯度和权重更新两步）几乎消除 bubble
- 代价是更复杂的调度和略多的通信

**Q: 如何选择 microbatch 数量 M？**

A: 经验法则：M ≥ 4P（bubble ≤ 20%）。但 M 越大意味着每个 microbatch 更小，可能导致 GPU 利用率不足。需要在 bubble 和 GPU 利用率之间平衡。

## 延伸阅读

- [GPipe 论文](https://arxiv.org/abs/1811.06965) — Huang et al., 2019
- [PipeDream](https://arxiv.org/abs/1806.03377) — 异步流水线
- [Megatron-LM 3D Parallelism](https://arxiv.org/abs/2104.04473) — Narayanan et al., 2021
- [Zero Bubble Pipeline](https://arxiv.org/abs/2401.10241) — Qi et al., 2024

---

## 术语表

| 术语 | 说明 |
|------|------|
| **流水线并行（Pipeline Parallelism, PP）** | 将模型按层切分到多个设备，数据依次流过各个 Stage |
| **Stage** | 流水线中的一个计算段，通常对应一个设备（或设备组）上的若干连续层 |
| **Microbatch** | 将一个 mini-batch 拆成多个小块（microbatch），让流水线的各 Stage 能重叠工作，减少空闲 |
| **Bubble（空泡）** | 流水线中某些 Stage 空闲等待的时间。Bubble 比例 = (P-1)/(P-1+M)，P=Stage 数，M=Microbatch 数 |
| **GPipe** | Google 提出的流水线并行方案，先做所有 microbatch 的 forward 再做 backward |
| **1F1B** | One Forward One Backward，交替执行 forward 和 backward，减少激活值缓存压力 |
| **Interleaved Pipeline** | 每个 Stage 分配不连续的层（虚拟 Stage），进一步减少 Bubble |
