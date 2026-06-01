# 实验环境搭建指南

> 不需要 A100 也能学完这套文档的大部分内容。本指南按"你手头有什么"来规划环境和实验路径。

## 核心原则

1. **不要等硬件齐了才开始**。很多关键概念在 CPU 或小显存 GPU 上就能验证
2. **本地做理解，云上做验证**。本地负责写代码、读 profiler 输出、调试逻辑；云上按小时租来跑重实验
3. **环境隔离**。vLLM、flash-attn、deepspeed、triton 的依赖经常互相冲突，不要全塞一个环境

## 按硬件条件选路线

### 路线 A：无独显 / 集显 / 显存 ≤ 4GB

你能做的比你想的多。

**本地可做：**

| 文档章节 | 可做的实验 | 工具 |
|---------|-----------|------|
| Ch01 CPU/Cache/NUMA | `perf stat`、cache miss 观察、NUMA 绑核 | Linux/WSL2 + perf |
| Ch03 CUDA 基础 | 读代码、理解 grid/block/thread 模型 | 纯阅读 + 纸上推演 |
| Ch04 Transformer | 手写 attention、观察 O(n²) 复杂度 | PyTorch CPU |
| Ch05 训练基础 | 训练循环、dataloader 调优、gradient checkpointing | PyTorch CPU |
| Ch05 Profiling | torch.profiler CPU 时间线 | PyTorch + TensorBoard |
| Ch06 分布式概念 | 理解 AllReduce/AllGather 的通信模式 | 纯阅读 + 画图 |

**云上补：**

- 免费：Google Colab (T4 16GB)、Kaggle Notebooks (P100/T4)
- 便宜：AutoDL、Vast.ai、RunPod 按小时租

**推荐本地环境：**
```bash
# WSL2 Ubuntu 22.04 或原生 Linux
python -m venv ~/.venvs/gpu-learning
source ~/.venvs/gpu-learning/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets matplotlib jupyter tensorboard
```

---

### 路线 B：有 GPU，显存 6-12GB（如 RTX 3060/4060/3070）

这档能覆盖文档 60-70% 的可操作内容。

**本地可做：**

| 文档章节 | 可做的实验 | 说明 |
|---------|-----------|------|
| Ch01-02 | 全部 | CPU 架构 + GPU 基本概念 |
| Ch03 CUDA | vector add、小 matmul、shared memory tiling | 需要 CUDA Toolkit |
| Ch04 Transformer | 完整 attention 实现、小 Transformer 训练 | PyTorch GPU |
| Ch05 Mixed Precision | FP32 vs FP16 速度/显存对比 | `torch.amp` |
| Ch05 Gradient Ckpt | 开关对比显存占用 | 小模型即可观察 |
| Ch05 Profiling | Nsight Systems 看 GPU timeline | 需装 Nsight |
| Ch06 DDP 入门 | 单机"伪多卡"DDP（1 GPU 模拟概念） | `torchrun --nproc=1` |
| Ch07 Quantization | bitsandbytes 4bit/8bit 推理 | 小模型如 TinyLlama |
| Ch09 GEMM | 前 3-4 个 kernel 版本（naive→tiled→shared memory） | 适配 sm_86/sm_89 |

**需要云上做：**

- 7B 模型 LoRA/QLoRA（需 16GB+）
- vLLM 完整体验（需 16-24GB）
- FlashAttention 性能对比（需较大 seq_len 才明显）
- 多卡 DDP/FSDP

**推荐本地环境：**
```bash
# WSL2 Ubuntu 22.04
# 确认 Windows 已装 NVIDIA 驱动，WSL 里能看到 GPU
nvidia-smi

# 创建环境
python -m venv ~/.venvs/train
source ~/.venvs/train/bin/activate

# PyTorch GPU 版（选匹配你 CUDA 的版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 训练常用
pip install transformers datasets accelerate peft bitsandbytes
pip install tensorboard wandb

# CUDA 开发（如果要写 kernel）
# 需要安装 CUDA Toolkit，WSL 里用：
# https://developer.nvidia.com/cuda-downloads → Linux → WSL-Ubuntu
```

**查看你的 GPU 计算能力：**
```python
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    cap = torch.cuda.get_device_capability()
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    # GEMM 项目编译时用 -arch=sm_{cap[0]}{cap[1]}
```

---

### 路线 C：有 GPU，显存 16-24GB（如 RTX 4090/3090/4080）

这档能覆盖文档 85%+ 的单卡内容。

**除路线 B 的全部内容外，还能做：**

| 实验 | 说明 |
|------|------|
| 7B 模型 QLoRA 微调 | 4bit 量化 + LoRA，24GB 刚好够 |
| vLLM 推理和 benchmark | 7B 模型 FP16 或 AWQ 4bit |
| FlashAttention 对比 | 长序列下 flash vs naive attention |
| torch.compile 加速 | 观察编译前后的 kernel 融合 |
| Triton 自定义 kernel | 写 fused attention、softmax 等 |
| KV Cache 显存影响 | 不同 seq_len 下的显存变化 |

**仍需云上做：**

- 多卡 DDP/FSDP/ZeRO
- Tensor Parallel / Pipeline Parallel
- 跨机通信和 NCCL 拓扑
- 70B 模型推理

---

## 云平台选择

### 免费

| 平台 | GPU | 显存 | 限制 |
|------|-----|------|------|
| Google Colab 免费版 | T4 | 16GB | 运行时间限制，空闲断开 |
| Kaggle Notebooks | T4/P100 | 16GB | 每周 30h GPU 配额 |

适合：快速验证小实验、跑 notebook 形式的 demo

### 按小时租（推荐用于重实验）

> 云 GPU 价格波动很快，下表只用于判断量级，不用于预算。下单前请按所在区域、GPU 型号、是否抢占式实例、磁盘/公网费用重新核价。

| 平台 | 特点 | 使用建议 |
|------|------|---------|
| [AutoDL](https://www.autodl.com/) | 国内，中文界面，型号选择多 | 适合短时实验，注意镜像、磁盘和排队情况 |
| [Vast.ai](https://vast.ai/) | 全球，价格和机器质量差异大 | 适合低成本试验，务必看主机评分、带宽和退款规则 |
| [RunPod](https://www.runpod.io/) | 上手相对顺滑，支持多种实例 | 适合 notebook、推理服务和中等规模实验 |
| [Lambda Labs](https://lambdalabs.com/) | 机器质量和多卡体验较稳定 | 适合多卡训练、长时间 profiling 和课程项目 |

**建议策略：**

1. 本地写好代码和配置，调通逻辑
2. 云上只做"运行和观察结果"，最大化利用租用时间
3. 结果（log、profiler trace、截图）保存到本地后立即释放

---

## 按文档章节的最小实验清单

每个实验都标注了最低硬件要求。

### Ch01 计算机架构
```
[ ] 用 perf stat 观察 cache miss          → CPU only
[ ] 顺序 vs 随机访问内存的速度差异          → CPU only
[ ] numactl 查看 NUMA 拓扑                 → CPU only (多核心)
```

### Ch03 CUDA 编程
```
[ ] vector_add kernel                      → 任意 NVIDIA GPU
[ ] 对比不同 block_size 的性能              → 任意 NVIDIA GPU
[ ] shared memory tiling matmul             → 任意 NVIDIA GPU
[ ] 用 Nsight Compute 看 occupancy          → 任意 NVIDIA GPU
```

### Ch04 Transformer
```
[ ] 手写 scaled dot-product attention       → CPU 或 GPU
[ ] 观察 seq_len 翻倍时显存和时间的变化      → GPU 4GB+
```

### Ch05 训练基础
```
[ ] FP32 vs BF16/FP16 训练速度对比          → GPU 6GB+
[ ] gradient checkpointing 显存节省量        → GPU 6GB+
[ ] dataloader num_workers/pin_memory 调优   → CPU 或 GPU
[ ] torch.profiler 生成 Chrome trace         → CPU 或 GPU
```

### Ch06 分布式训练
```
[ ] 单机单卡理解 DDP 的 gradient sync 概念   → GPU 6GB+
[ ] 单机多卡 DDP：torchrun --nproc=2+       → 2+ GPU 或 云
[ ] FSDP/ZeRO 对显存的影响                   → 2+ GPU 或 云
```

### Ch07 推理优化
```
[ ] KV Cache：对比有无 cache 的生成速度       → GPU 8GB+
[ ] bitsandbytes 4bit 量化推理               → GPU 8GB+
[ ] vLLM serve 和 benchmark                  → GPU 16GB+
[ ] FlashAttention vs naive 速度对比          → GPU 16GB+
```

### Ch09 GEMM
```
[ ] Kernel v1: naive matmul                  → 任意 NVIDIA GPU
[ ] Kernel v2: global memory coalescing       → 任意 NVIDIA GPU
[ ] Kernel v3: shared memory tiling           → 任意 NVIDIA GPU
[ ] Kernel v4: 对比 cuBLAS                   → 任意 NVIDIA GPU
[ ] 后续 kernel 和 Nsight 分析               → A100+ 更好
```

---

## 环境隔离建议

不要把所有包装在一起。建议至少分 3 个环境：

```
~/.venvs/
├── base-learn/     # PyTorch + transformers + datasets + jupyter
├── cuda-dev/       # CUDA Toolkit + Triton + Nsight
└── infer-serve/    # vLLM（它的依赖比较独立，容易和 flash-attn 版本冲突）
```

或者用 conda/mamba：
```bash
mamba create -n learn python=3.11 pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
mamba create -n cuda-dev python=3.11  # 手动装 triton, flash-attn
mamba create -n serving python=3.11   # 单独装 vllm
```

---

## 快速验证环境可用

装完后跑这个确认基本链路没问题：

```python
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 简单 matmul benchmark
    x = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
    
    # warmup
    for _ in range(3):
        _ = x @ x
    torch.cuda.synchronize()
    
    import time
    start = time.perf_counter()
    for _ in range(10):
        _ = x @ x
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    tflops = 2 * 4096**3 * 10 / elapsed / 1e12
    print(f"FP32 matmul: {tflops:.1f} TFLOPS")
    
    # BF16 (如果支持)
    if torch.cuda.get_device_capability()[0] >= 8:
        x16 = x.to(torch.bfloat16)
        for _ in range(3):
            _ = x16 @ x16
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(10):
            _ = x16 @ x16
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        tflops_bf16 = 2 * 4096**3 * 10 / elapsed / 1e12
        print(f"BF16 matmul: {tflops_bf16:.1f} TFLOPS")
        print(f"BF16/FP32 speedup: {tflops_bf16/tflops:.1f}x")
else:
    print("No GPU — CPU-only experiments available")
    print("See '路线 A' in this guide")
```

---

## 一句话总结

**没有强卡 → 本地学概念写代码，云上按小时验证关键实验。**
**有小卡 → 80% 内容本地可做，只有多卡并行和大模型推理需要云。**
**有 24GB 卡 → 单卡内容几乎全覆盖，只需租云做分布式。**
