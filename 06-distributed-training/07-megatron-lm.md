# Megatron-LM

> NVIDIA 的大模型训练框架——3D 并行的标杆实现。

## 核心概念

### Megatron-LM 是什么

Megatron-LM 是 NVIDIA 开发的大规模 Transformer 训练框架，首创并整合了 **3D 并行**（TP + PP + DP）。GPT-3、LLaMA 等大模型的训练都参考了 Megatron 的并行策略。

### 3D 并行架构

```
                        ┌──────────────── Data Parallel ────────────────┐
                        │                                                │
          ┌─────── PP Stage 0 ───────┐    ┌─────── PP Stage 1 ───────┐  
          │  TP Group (NVLink)       │    │  TP Group (NVLink)       │  
          │  ┌─────┐ ┌─────┐        │    │  ┌─────┐ ┌─────┐        │  
  DP      │  │GPU 0│ │GPU 1│ ...    │    │  │GPU 4│ │GPU 5│ ...    │  
  Rank 0  │  └─────┘ └─────┘        │←──→│  └─────┘ └─────┘        │  
          │  Layer 0-15              │    │  Layer 16-31             │  
          └──────────────────────────┘    └──────────────────────────┘  
                                                                        
          ┌─────── PP Stage 0 ───────┐    ┌─────── PP Stage 1 ───────┐  
          │  TP Group (NVLink)       │    │  TP Group (NVLink)       │  
  DP      │  ┌─────┐ ┌─────┐        │    │  ┌─────┐ ┌─────┐        │  
  Rank 1  │  │GPU 8│ │GPU 9│ ...    │←──→│  │GPU12│ │GPU13│ ...    │  
          │  Layer 0-15              │    │  Layer 16-31             │  
          └──────────────────────────┘    └──────────────────────────┘  
```

### 并行度配置原则

```
总 GPU 数 = TP × PP × DP

配置策略:
1. TP 放机内 (NVLink): 通常 TP=8 (DGX H100)
2. PP 跨机 (InfiniBand): PP = 总层数需要的 stage 数
3. DP = 总 GPU 数 / (TP × PP): 剩下的维度给 DP
4. 通常组合 FSDP/ZeRO 做 data parallel 进一步节省显存

示例: 训练 175B GPT-3，使用 512 × H100
  TP = 8 (机内 8 卡 NVLink)
  PP = 8 (每台机器 = 1 stage，共 8 stage)
  DP = 512 / (8 × 8) = 8
```

### 关键技术特性

#### 1. 序列并行 (Sequence Parallelism)

在非 TP 区域（LayerNorm、Dropout）沿序列维度切分：

```
标准 TP → 序列并行:
  AllReduce 拆分为 AllGather + ReduceScatter
  
  LayerNorm(seq/T) → [g: AllGather → TP计算 → f: ReduceScatter] → LayerNorm(seq/T)
  
  激活值显存节省: ~T 倍（因为非 TP 区域也切分了）
  通信量: 不变
```

#### 2. Selective Activation Recomputation

不是重计算所有激活值，而是选择性地只重计算显存占用大但计算便宜的操作：

```
一个 Transformer 层的激活值:

  Attention Score (QK^T):  b × h × s × s × 2    ← 和 s² 成正比，巨大！
  Softmax 输出:             同上
  Dropout Mask:             同上
  
  Linear 层输出:            b × s × h × 2        ← 和 s 成正比
  
策略: 重计算 QK^T / Softmax / Dropout（显存大，但重计算快）
      保留 Linear 层输出（显存占比小，重计算慢）

→ 显存节省约 50-70%，重计算开销 < 5%
```

#### 3. 分布式 Optimizer

Megatron-LM 的分布式优化器将优化器状态切分到 DP 组内（类似 ZeRO Stage 1）：

```
DP=8: 每卡只存 1/8 的优化器状态
更新时: AllGather 获取全部参数 → 本地更新 → ReduceScatter 分发更新

和 ZeRO-1 类似，但 Megatron 的实现和 TP/PP 深度集成
```

## 关键细节

### 训练配置示例

```bash
# Megatron-LM 训练 GPT 模型
python pretrain_gpt.py \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 512 \
    \
    # 3D 并行配置
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 2 \
    # DP 自动推断: total_gpus / (TP × PP)
    \
    # 序列并行
    --sequence-parallel \
    \
    # 混合精度
    --bf16 \
    --use-flash-attn \
    \
    # 激活重计算
    --recompute-granularity selective \
    \
    # 分布式优化器(ZeRO-1)
    --use-distributed-optimizer \
    \
    # 数据路径
    --data-path $DATA_PATH \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model $TOKENIZER_PATH
```

### 性能指标：MFU

**MFU (Model FLOPs Utilization)** 是衡量训练效率的核心指标：

$$\text{MFU} = \frac{\text{实际 FLOPS}}{\text{GPU 峰值 FLOPS}}$$

```
计算实际 FLOPS:
  每步计算量 ≈ 6 × params × tokens_per_step  (近似公式)
  
  实际 FLOPS = 计算量 / 每步时间

例: 7B 模型, batch=512, seq=2048, 每步 8.5s, 8×H100
  tokens/step = 512 × 2048 = 1M
  FLOPs/step = 6 × 7B × 1M = 42 PFLOPs
  实际 FLOPS = 42P / 8.5s = 4.94 PFLOPS
  H100 BF16 峰值 = 1979 TFLOPS × 8 = 15.8 PFLOPS
  MFU = 4.94 / 15.8 = 31.3%
  
优秀的 MFU: >40% (单机可达 50%+, 大集群 30-45%)
```

### 3D 并行度调优

```
目标: 最大化 MFU

TP 的影响:
  + 减少显存（参数/激活值切分）
  - AllReduce 开销（每层 2 次）
  建议: TP ≤ 单机 GPU 数（不跨机）

PP 的影响:
  + 减少显存（层切分）
  + 跨机通信量小（只传激活值）
  - Bubble 开销
  建议: microbatch 数 ≥ 4 × PP

DP 的影响:
  + 线性 scale 吞吐量（理想情况）
  - AllReduce 梯度（但可以和计算 overlap）
  建议: 越大越好，受限于 global batch size

调优流程:
  1. 先确定 TP (通常 = 8 for DGX)
  2. 调 PP 使模型能装进显存
  3. DP 用剩余 GPU
  4. 调 microbatch size 和 gradient_accumulation
  5. Profile，看哪个环节是瓶颈
```

## 常见问题

**Q: Megatron-LM 和 DeepSpeed 能一起用吗？**

A: 可以，这是很常见的组合：Megatron-LM 提供 TP + PP，DeepSpeed 提供 ZeRO（用于 DP 部分）。NVIDIA 和微软联合维护了 [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) 集成版本。

**Q: MFU 为什么很难超过 50%？**

A: 因为理论峰值假设所有时钟周期都在做有效计算，但实际有：
- 通信等待（TP AllReduce、PP bubble、DP 梯度同步）
- 内存访问（GEMM 不是完全 compute-bound）
- 非 GEMM 操作（LayerNorm、Softmax 等 memory-bound 操作）
- Activation Recomputation 的额外计算
- Python/框架开销

**Q: LLaMA / GPT-4 的训练用的是什么并行策略？**

A: 公开信息有限，但推测：
- LLaMA 65B: TP=8, PP=4-8, DP=64+ (2048 A100)
- GPT-4: 可能更大规模，加上 MoE 的 Expert Parallel

## 延伸阅读

- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2020
- [Efficient Large-Scale Language Model Training (3D Parallelism)](https://arxiv.org/abs/2104.04473) — Narayanan et al., 2021
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Korthikanti et al., 2022
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
