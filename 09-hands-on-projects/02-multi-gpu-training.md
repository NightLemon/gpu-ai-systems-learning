# 实战：多 GPU 训练

> 从单卡到多卡再到多机——亲手感受分布式训练的全流程。

## 项目配置

| 项目 | 值 |
|------|---|
| 模型 | GPT-2 125M（nanoGPT 或 HuggingFace GPT2） |
| 数据集 | OpenWebText (nanoGPT 自带) 或 WikiText-103 |
| 训练步数 | 1000 步 (足够对比性能，不需要训到收敛) |
| 固定变量 | seq_len=1024, dtype=BF16, optimizer=AdamW, lr=6e-4 |
| 硬件 | 4-8× GPU (A100 80GB 推荐，RTX 4090 也可以) |
| 软件 | PyTorch 2.x, DeepSpeed 当前稳定版（本仓库基线见“版本基线”页） |

## Step 1: 单卡 Baseline（~2 小时）

### 操作

```bash
# 方案 A: 使用 nanoGPT (推荐，最简洁)
git clone https://github.com/karpathy/nanoGPT.git && cd nanoGPT
python data/openwebtext/prepare.py
python train.py --batch_size=12 --max_iters=1000 --compile=True

# 方案 B: 使用 HuggingFace
pip install transformers datasets accelerate
# (使用附带的训练脚本，见下方)
```

### 采集指标

```python
# 在训练循环中记录
import time, torch

for step in range(1000):
    start = time.time()
    # ... training step ...
    torch.cuda.synchronize()
    step_time = time.time() - start
    
    tokens_per_step = batch_size * seq_len
    if step % 100 == 0:
        print(f"Step {step}: {step_time:.3f}s, "
              f"{tokens_per_step/step_time:.0f} tokens/s, "
              f"GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
```

### 验收

记录并填入结果表 (Step 5)：tokens/s、显存峰值、step time。

## Step 2: DDP 多卡数据并行（~2 小时）

### 操作

```bash
# 4 卡 DDP
torchrun --nproc_per_node=4 train.py \
    --batch_size=12 --max_iters=1000

# 注意: effective batch size = 12 × 4 = 48
# 为了公平对比，保持 per_gpu_batch_size 不变
```

### 关注点

1. tokens/s 是否接近单卡的 4 倍？差距多少？
2. 用 `nvidia-smi dmon -s u -d 1` 观察 4 张卡的利用率是否均匀
3. 差距的原因：通信开销 + 同步等待

### Profile

```bash
# 用 PyTorch Profiler 看通信占比
# 在训练代码中加入:
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler('./log/ddp'),
) as prof:
    for step in range(5):
        # ... training step ...
        prof.step()
# tensorboard --logdir=./log/ddp
```

## Step 3: DeepSpeed ZeRO 对比（~3 小时）

### 操作

分别用 ZeRO Stage 1/2/3 运行 1000 步，**保持其他变量不变**：

```json
// ds_config_z1.json
{
  "train_micro_batch_size_per_gpu": 12,
  "gradient_accumulation_steps": 1,
  "zero_optimization": {"stage": 1},
  "bf16": {"enabled": true},
  "optimizer": {"type": "AdamW", "params": {"lr": 6e-4}}
}

// ds_config_z2.json — 改 "stage": 2
// ds_config_z3.json — 改 "stage": 3
```

```bash
deepspeed --num_gpus=4 train_ds.py --deepspeed ds_config_z1.json
deepspeed --num_gpus=4 train_ds.py --deepspeed ds_config_z2.json
deepspeed --num_gpus=4 train_ds.py --deepspeed ds_config_z3.json
```

### 关注点

对 ZeRO 1/2/3 分别记录：
- tokens/s（相对 DDP 的变化）
- 显存占用（应该逐级下降）
- 通信时间占比（Stage 3 应该最高）

## Step 4: Profile 深度分析（~2 小时）

```bash
# Nsight Systems 采集完整 trace
nsys profile -o trace_ddp --trace=cuda,nvtx,nccl \
    torchrun --nproc_per_node=4 train.py --max_iters=10

# 打开 trace_ddp.nsys-rep 观察:
# 1. Forward / Backward / AllReduce 的时间占比
# 2. AllReduce 是否和 Backward 重叠（overlap）
# 3. 是否有 GPU 空闲的间隔（数据加载等待）
```

**计算 MFU**：

```python
# GPT-2 125M, 4×A100, BF16
model_params = 125e6
tokens_per_step = batch_size * seq_len * 4  # per_gpu × num_gpus
mfu = 6 * model_params * tokens_per_step / (step_time * 4 * 312e12)
# A100 BF16 peak = 312 TFLOPS
print(f"MFU: {mfu:.1%}")
```

## Step 5: 结果表

复制下表，填入你的实测数据：

```
硬件: ______ × _______ (例: 4 × A100 80GB)
模型: GPT-2 125M, seq_len=1024, per_gpu_batch=12, BF16

配置           | tokens/s | 相对单卡 | 显存峰值/卡 | step time | MFU  | 通信占比
────────────────────────────────────────────────────────────────────────────────
单卡            |          | 1.0x    |             |           |      | 0%
DDP 4卡         |          |    x    |             |           |      |    %
ZeRO-1 4卡      |          |    x    |             |           |      |    %
ZeRO-2 4卡      |          |    x    |             |           |      |    %
ZeRO-3 4卡      |          |    x    |             |           |      |    %
```

### 参考值范围（4× A100 80GB）

| 配置 | 相对单卡加速 | 显存/卡 |
|------|------------|--------|
| DDP | 3.5-3.9x | ~同单卡 |
| ZeRO-1 | 3.4-3.8x | ~85% 单卡 |
| ZeRO-2 | 3.2-3.6x | ~65% 单卡 |
| ZeRO-3 | 2.8-3.3x | ~45% 单卡 |

> 如果你的加速比远低于参考值，常见原因：数据加载瓶颈（加 num_workers）、未开 BF16、batch size 太小。

## 常见问题

**Q: 没有 A100 怎么办？**

A: RTX 4090 / 3090 也可以做单机多卡实验。跨机训练技术上也可以（通过以太网 + NCCL），但消费级 GPU 通常缺少 RDMA 网络，跨机通信会很慢。此外，没有 NVLink 意味着机内多卡 P2P 带宽较低（只有 PCIe），TP 等高通信并行策略收益会明显下降。数值会和 A100 不同，但 DDP vs ZeRO 的相对趋势应该一致。如果显存不够，缩小 batch_size。

**Q: 125M 模型太小，ZeRO 的差异不明显怎么办？**

A: 换 GPT-2 350M 或 GPT-2 774M。模型越大，ZeRO 的显存收益越明显，通信开销差异也越明显。

## 参考资料

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy 的简洁 GPT 训练代码
- [DeepSpeed Getting Started](https://www.deepspeed.ai/getting-started/)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **`torchrun`** | PyTorch 的分布式训练启动器，自动设置 rank、world_size 等环境变量 |
| **nanoGPT** | Andrej Karpathy 的最简洁 GPT 训练代码，适合学习和实验 |
| **tokens/s** | 每秒处理的 token 数，衡量训练吞吐量的核心指标 |
| **step time** | 完成一次前向+反向传播+参数更新的总时间 |
| **`torch.cuda.synchronize()`** | 等待 GPU 上所有工作完成，测量时间前必须调用，否则测不准 |
