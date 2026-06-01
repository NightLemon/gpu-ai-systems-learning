# 工具与开源项目

## 训练框架

| 项目 | 说明 | 链接 |
|------|------|------|
| **PyTorch** | 主流深度学习框架 | [pytorch.org](https://pytorch.org/) |
| **Megatron-LM** | NVIDIA 大模型训练框架 (3D 并行) | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| **DeepSpeed** | 微软深度学习优化库 (ZeRO 系列) | [GitHub](https://github.com/microsoft/DeepSpeed) |
| **FSDP** | PyTorch 原生全分片数据并行 | [PyTorch Docs](https://pytorch.org/docs/stable/fsdp.html) |
| **Composer** | MosaicML 的训练效率库 | [GitHub](https://github.com/mosaicml/composer) |

## 推理框架

| 项目 | 说明 | 链接 |
|------|------|------|
| **vLLM** | 高性能 LLM 推理 (PagedAttention) | [GitHub](https://github.com/vllm-project/vllm) |
| **TensorRT-LLM** | NVIDIA LLM 推理优化 | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **SGLang** | 快速 LLM 推理 (RadixAttention) | [GitHub](https://github.com/sgl-project/sglang) |
| **llama.cpp** | CPU/GPU 推理 (GGUF 量化) | [GitHub](https://github.com/ggerganov/llama.cpp) |
| **TGI** | HuggingFace 推理服务 | [GitHub](https://github.com/huggingface/text-generation-inference) |

## CUDA / GPU 编程

| 项目 | 说明 | 链接 |
|------|------|------|
| **Triton** | Python GPU 编程语言 | [GitHub](https://github.com/triton-lang/triton) |
| **CUTLASS** | NVIDIA CUDA 模板库 (GEMM) | [GitHub](https://github.com/NVIDIA/cutlass) |
| **flash-attention** | FlashAttention 实现 | [GitHub](https://github.com/Dao-AILab/flash-attention) |
| **ThunderKittens** | 简洁的 GPU kernel 开发框架 | [GitHub](https://github.com/HazyResearch/ThunderKittens) |

## 量化工具

| 项目 | 说明 | 链接 |
|------|------|------|
| **GPTQModel** | GPTQ 量化工具，vLLM 当前文档重点覆盖的 GPTQ 路线之一 | [GitHub](https://github.com/ModelCloud/GPTQModel) |
| **AutoAWQ** | AWQ 量化工具；使用前确认当前维护状态和 serving 框架支持 | [GitHub](https://github.com/casper-hansen/AutoAWQ) |
| **bitsandbytes** | INT8/NF4 量化 (QLoRA) | [GitHub](https://github.com/TimDettmers/bitsandbytes) |

## Profiling 工具

| 工具 | 说明 |
|------|------|
| **Nsight Systems** | 系统级 GPU profiling |
| **Nsight Compute** | Kernel 级 GPU profiling |
| **PyTorch Profiler** | PyTorch 原生 profiler |
| **DCGM** | GPU 集群监控 |
| **Weights & Biases** | 训练实验跟踪 |

## 学习用代码库

| 项目 | 说明 | 链接 |
|------|------|------|
| **nanoGPT** | 最简洁的 GPT 训练代码 | [GitHub](https://github.com/karpathy/nanoGPT) |
| **LLM.c** | 纯 C/CUDA 实现 GPT 训练 | [GitHub](https://github.com/karpathy/llm.c) |
| **CUDA-Samples** | NVIDIA 官方 CUDA 示例 | [GitHub](https://github.com/NVIDIA/cuda-samples) |
| **Unsloth** | 高效 LLM 微调 (Triton kernel) | [GitHub](https://github.com/unslothai/unsloth) |
