# 实战：多 GPU 训练

> 从单卡到多卡再到多机——亲手感受分布式训练的全流程。

## 项目目标

用 Megatron-LM 或 DeepSpeed 在多卡上训练一个小型 GPT 模型，掌握分布式训练配置和 profiling。

## 环境准备

```bash
# 推荐: 2-8 张 GPU (如 A100/H100)
pip install torch torchvision  # PyTorch
pip install deepspeed          # DeepSpeed
# 或克隆 Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
```

## 步骤

### Step 1: 单卡 Baseline

```python
# 用 nanoGPT 或 HuggingFace 训练一个小 GPT (如 125M 参数)
# 记录: 训练速度 (tokens/s)、显存使用、loss 曲线

model = GPT2LMHeadModel(config)  # ~125M params
# 训练 1000 步，记录 baseline 性能
```

### Step 2: DDP 多卡数据并行

```bash
# 用 torchrun 启动 4 卡 DDP
torchrun --nproc_per_node=4 train.py

# 关注:
# - 速度是否接近 4x？差距在哪？
# - nvidia-smi 看 GPU 利用率
# - 用 PyTorch Profiler 看通信占比
```

### Step 3: DeepSpeed ZeRO 优化

```json
// ds_config.json
{
    "train_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "zero_optimization": { "stage": 2 },
    "bf16": { "enabled": true }
}
```

逐步测试 ZeRO Stage 1 → 2 → 3，对比显存和速度变化。

### Step 4: Megatron-LM 3D 并行（需要 8+ GPU）

```bash
# TP=2, PP=2, DP=2 (8 GPU)
python pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --num-layers 12 --hidden-size 768 --num-attention-heads 12 \
    --seq-length 1024 --micro-batch-size 4 --global-batch-size 64 \
    --bf16 --use-flash-attn
```

### Step 5: Profiling & 分析

```bash
# Nsight Systems 采集 trace
nsys profile --trace=cuda,nvtx,nccl torchrun --nproc_per_node=4 train.py

# 分析:
# 1. Forward/Backward 占比
# 2. NCCL AllReduce 耗时
# 3. 数据加载是否有空洞
# 4. 计算 MFU
```

## 期望结果

| 配置 | 相对速度 | 显存/卡 |
|------|---------|--------|
| 单卡 | 1x | 100% |
| DDP 4卡 | ~3.5-3.8x | ~100% |
| ZeRO-2 4卡 | ~3.2-3.6x | ~60% |
| ZeRO-3 4卡 | ~2.8-3.2x | ~40% |

## 参考资料

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy 的简洁 GPT 训练代码
- [DeepSpeed Getting Started](https://www.deepspeed.ai/getting-started/)
- [Megatron-LM Tutorial](https://github.com/NVIDIA/Megatron-LM#training)
