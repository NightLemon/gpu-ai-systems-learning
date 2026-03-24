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

### Build Engine 的约束与影响

```
Build 阶段的关键参数（一旦确定，运行时不能改）:

  --max_batch_size 64      构建时指定的最大 batch
  --max_input_len 2048     最大输入长度
  --max_seq_len 4096       最大总长度（输入+输出）
  --tp_size 2              Tensor Parallel degree
  --dtype float16          数据类型
  --quantization fp8       量化方式

⚠️ 以下任一改变都需要重新 build:
  - 更换 GPU 型号（A100 → H100）
  - 改变 TP/PP 配置
  - 改变 max_batch_size / max_seq_len
  - 改变量化方式
  - 更新模型权重（如微调后）

对线上发布意味着:
  1. Build 耗时（7B ~10min, 70B ~30-60min）→ 不能热更新模型
  2. 需要为不同配置维护多个 engine 文件
  3. 发布流程: 新模型 → Build Engine → 验证 → 灰度发布
  4. Engine 文件可能数十 GB → 需要存储和分发方案
```

### TRT-LLM 与 Triton 的职责边界

```
┌─────────────────────────────────────────────────────────┐
│          Triton Inference Server                         │
│  负责:                                                   │
│  - HTTP/gRPC API 暴露                                   │
│  - 请求排队和负载均衡                                     │
│  - 多模型管理和版本切换                                   │
│  - 健康检查和指标导出                                     │
│  - 前后处理 pipeline (tokenize/detokenize)               │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │          TensorRT-LLM Backend                      │  │
│  │  负责:                                             │  │
│  │  - 加载和运行 TRT Engine                           │  │
│  │  - In-flight Batching (Continuous Batching)        │  │
│  │  - KV-Cache 管理 (Paged)                          │  │
│  │  - Beam Search / Sampling                          │  │
│  │  - Tensor Parallel 协调                            │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

关键: TRT-LLM 只管推理计算
      Triton 管整个服务的生命周期
      不用 Triton 也能用 TRT-LLM（直接用 Python API），但生产建议搭配
```

### TensorRT-LLM vs vLLM 性能对比

```
典型场景 (LLaMA 7B, A100):
  
  指标                 vLLM        TRT-LLM
  ─────────────────────────────────────────
  TTFT (ms)           ~50-80      ~40-60      ← 编译优化
  Decode (tokens/s)   ~40-60      ~50-70      ← kernel 优化
  Max Throughput       ~2000       ~2500-3000  ← 整体优化
  
  差距主要来源:
  - Kernel 融合更激进（编译时已知 shape → 更优 kernel 选择）
  - GEMM Plugin 针对特定矩阵形状优化
  - FP8 整合更深（Hopper 原生）
```

### 部署选型决策

```
选型维度:

1. 性能要求:
   TRT-LLM 通常比 vLLM 快 15-30%（取决于场景）
   如果 SLA 非常紧（如 TTFT < 50ms），TRT-LLM 更合适
   如果性能差 15% 可接受，vLLM 的运维成本更低

2. 模型更换频率:
   每周或更频繁换模型 → vLLM（pip install 即用）
   模型稳定、长期服务 → TRT-LLM（一次 build 长期运行）

3. 团队能力:
   不熟悉 TensorRT/NVIDIA 生态 → vLLM 上手快
   有 NVIDIA TAM 支持或 TRT 经验 → TRT-LLM 能调到更优

4. 硬件:
   H100 + FP8 → TRT-LLM 的 FP8 优化是独占优势
   A100 / 消费级 GPU → vLLM 和 TRT-LLM 差距缩小

5. 功能需求:
   需要 Speculative Decoding / 自定义采样 → 查各框架的支持程度
   需要 LoRA serving / 多模态 → vLLM 生态更丰富

6. 另一选择: SGLang
   RadixAttention（更激进的前缀缓存）
   某些场景性能超过 vLLM 和 TRT-LLM
   快速迭代中，值得关注
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
