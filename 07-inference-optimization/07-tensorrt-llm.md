# TensorRT-LLM

> NVIDIA 官方的 LLM 推理优化库——将模型编译为高度优化的执行引擎。

## 核心概念

### TensorRT-LLM 是什么

TensorRT-LLM = TensorRT (图优化/编译器) + LLM 特定优化 (KV-Cache、Attention kernel、量化)

```
工作流:
  HuggingFace Model → TensorRT-LLM Build → TRT Engine → Runtime Inference
  
  Build 阶段 (Offline):
    1. 将 PyTorch 模型权重转换
    2. 图优化: 算子融合、常量折叠
    3. 编译为 TensorRT Engine (GPU 特定优化)
    4. 可选: 量化 (FP8/INT8/INT4)
  
  Runtime (Online):
    1. 加载 Engine
    2. In-flight Batching (Continuous Batching)
    3. PagedKV-Cache
    4. 高度优化的 Attention/GEMM kernel
```

### 关键优化

```
Kernel 融合:
  标准 PyTorch: [Linear] → [Add Bias] → [LayerNorm] → [GeLU]
  TRT-LLM:     [Linear+Bias+LayerNorm+GeLU]  ← 一个融合 kernel
  → 减少 kernel launch 开销和 HBM 读写

FP8 量化 (Hopper 原生):
  TRT-LLM 深度集成 H100 的 FP8 Tensor Core
  → 计算速度翻倍，精度影响极小

GEMM Plugin:
  针对 LLM 常见的 GEMM 形状（长而窄的矩阵）定制优化
  
Custom Attention Kernel:
  MHA / GQA / MQA 专用 kernel
  FlashAttention 集成
  支持 ALiBi、RoPE 等位置编码
```

### 使用示例

```python
# 1. 转换模型 (使用 TRT-LLM 的 Python API)
from tensorrt_llm import Builder
from tensorrt_llm.models import LLaMAForCausalLM

# 从 HuggingFace checkpoint 构建
model = LLaMAForCausalLM.from_hugging_face(
    "meta-llama/Llama-2-7b-hf",
    dtype="float16",
    mapping=tensorrt_llm.Mapping(world_size=2, tp_size=2),
)

builder = Builder()
engine = builder.build_engine(model, BuildConfig(
    max_batch_size=64,
    max_input_len=2048,
    max_seq_len=4096,
    max_beam_width=1,
))
engine.save("llama-7b-trt")

# 2. 推理
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("llama-7b-trt")
outputs = runner.generate(
    input_text=["The capital of France is"],
    max_new_tokens=100,
    temperature=0.7,
)
```

```bash
# 命令行方式
# 转换
trtllm-build --model_dir meta-llama/Llama-2-7b-hf \
    --output_dir ./engine \
    --dtype float16 \
    --tp_size 2 \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16

# FP8 量化 + 编译
trtllm-build --model_dir meta-llama/Llama-2-7b-hf \
    --output_dir ./engine_fp8 \
    --dtype float16 \
    --quantization fp8 \
    --tp_size 2
```

## 关键细节

### TensorRT-LLM vs vLLM 性能对比

```
典型场景 (LLaMA 7B, A100):
  
  指标                 vLLM        TRT-LLM
  ─────────────────────────────────────────
  TTFT (ms)           ~50-80      ~40-60      ← 编译优化
  Decode (tokens/s)   ~40-60      ~50-70      ← kernel 优化
  Max Throughput       ~2000       ~2500-3000  ← 整体优化
  
  差距主要来源:
  - Kernel 融合更激进
  - GEMM 针对特定形状优化
  - FP8 整合更深
  
  但:
  - vLLM 更新快、模型支持广
  - TRT-LLM 需要 build engine（每换一个配置就要重新编译）
```

### Triton Inference Server 集成

TensorRT-LLM 通常搭配 Triton Inference Server 部署：

```
┌──────────────────────────────────────────────┐
│          Triton Inference Server              │
│  ┌─────────┐ ┌──────────────────────┐       │
│  │ HTTP/gRPC│ │  Request Scheduler    │       │
│  │ Frontend │→│  (Dynamic Batching)   │       │
│  └─────────┘ └──────────┬───────────┘       │
│                          │                    │
│  ┌──────────────────────▼───────────────┐   │
│  │      TensorRT-LLM Backend             │   │
│  │   (In-flight Batching, PagedKV, FP8)  │   │
│  └───────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

### 何时选择 TensorRT-LLM

```
✅ 选 TRT-LLM:
  - 生产环境，性能要求极高
  - 使用 NVIDIA GPU (A100/H100)
  - 模型固定，不频繁更换
  - 需要 FP8 量化最优性能
  - 已有 Triton Inference Server 基础设施

✅ 选 vLLM:
  - 快速原型 / 研究
  - 需要频繁切换模型
  - 团队不熟悉 TensorRT
  - 需要最新模型支持（vLLM 跟进更快）
  - 需要灵活的自定义功能
```

## 常见问题

**Q: TensorRT-LLM 的 "build" 过程要多久？**

A: 取决于模型大小和配置。7B 模型通常 5-15 分钟，70B 模型可能 30-60 分钟。每次更改 batch size、sequence length、TP size 等配置都需要重新 build。

**Q: TensorRT-LLM 支持哪些模型？**

A: 支持主流 LLM：LLaMA、GPT、Falcon、Mistral、Mixtral（MoE）、ChatGLM 等。但新模型的支持通常比 vLLM 慢几周到几个月。

**Q: SGLang 和这些推理框架的关系？**

A: SGLang 是另一个高性能推理框架，定位类似 vLLM。它的特色是 RadixAttention（更激进的前缀缓存）和结构化生成优化。性能在某些场景下超过 vLLM 和 TRT-LLM。

## 延伸阅读

- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [SGLang](https://github.com/sgl-project/sglang)
