# TensorRT-LLM

> NVIDIA 官方的 LLM 推理优化库——将模型编译为高度优化的执行引擎。示例和参数按概念讲解，实际部署请优先查当前版本文档和 `trtllm-build --help`。

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
  → 吞吐潜力高，但是否可用取决于模型、量化 recipe、校准/scale 管理和任务级精度回归

GEMM Plugin:
  针对 LLM 常见的 GEMM 形状（长而窄的矩阵）定制优化
  
Custom Attention Kernel:
  MHA / GQA / MQA 专用 kernel
  FlashAttention 集成
  支持 ALiBi、RoPE 等位置编码
```

### 使用示例

TensorRT-LLM 的高层 API 和命令行变化比较快。学习时建议把下面代码当作“当前入口形态”的例子，而不是背具体参数名。

```python
# Python 高层 API：适合先验证模型能跑通
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
sampling_params = SamplingParams(max_tokens=100, temperature=0.7)

outputs = llm.generate(
    ["The capital of France is"],
    sampling_params,
)
print(outputs[0].outputs[0].text)
```

```bash
# OpenAI-compatible serving 入口
trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 0.0.0.0 \
    --port 8000

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","prompt":"Hello","max_tokens":64}'
```

```bash
# 低层 engine build 流程仍然存在，但当前版本通常以 checkpoint_dir / build_config 为核心。
# 真正部署前务必以 trtllm-build --help 和对应模型 example 为准。
trtllm-build \
    --checkpoint_dir <converted_checkpoint_dir> \
    --output_dir ./engine \
    --max_batch_size 64 \
    --max_seq_len 4096
```

## 关键细节

### Build Engine 的约束与影响

```
Build 阶段的关键参数（一旦确定，运行时不能改）:

  --max_batch_size 64      构建时指定的最大 batch
  --max_input_len 2048     最大输入长度
  --max_seq_len 4096       最大总长度（输入+输出）
  --tp_size 2              Tensor Parallel degree
  dtype / quantization     数据类型和量化方式（具体参数名随版本变化）

⚠️ 以下任一改变都需要重新 build:
  - 更换 GPU 型号（A100 → H100）
  - 改变 TP/PP 配置
  - 改变 max_batch_size / max_seq_len
  - 改变量化方式
  - 更新模型权重（如微调后）

对线上发布意味着:
  1. Build 耗时可能从数分钟到更久 → 不能像纯解释式服务那样随意热更新模型
  2. 需要为不同配置维护多个 engine 文件
  3. 发布流程: 新模型 → Build Engine → 验证 → 灰度发布
  4. Engine 文件可能数十 GB → 需要存储和分发方案
```

这里真正的工程含义不是“build 很麻烦”，而是 TRT-LLM 更像编译型发布链路：

- 你需要提前确定主要服务形态
- 你需要把 engine 产物纳入版本管理和发布流程
- 你需要接受“换模型快”与“榨性能极致”之间的取舍

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

不要把某篇 benchmark 里的固定数字搬到自己的选型里。TRT-LLM 的优势通常来自编译期 shape 信息、更深的 kernel/engine 优化，以及对 NVIDIA 数据中心 GPU 低精度路径的快速跟进；vLLM/SGLang 的优势通常是模型支持、部署摩擦和迭代速度。

真正决定是否值得切到 TRT-LLM 的，通常是下面三件事：

- 你的线上流量是否稳定到值得为特定 shape 和配置做编译优化
- 你的团队是否能承受更重的 build、验证和灰度流程
- 额外拿到的性能，是否真的能换来可观的 GPU 节省或更紧的 SLA

### 部署选型决策

```
选型维度:

1. 性能要求:
   TRT-LLM 在稳定 shape、稳定模型和 NVIDIA 数据中心 GPU 上经常有优势，但收益幅度必须实测
   如果 SLA 非常紧（如 TTFT < 50ms），TRT-LLM 更合适
   如果性能差 15% 可接受，vLLM 的运维成本更低

2. 模型更换频率:
   每周或更频繁换模型 → vLLM（pip install 即用）
   模型稳定、长期服务 → TRT-LLM（一次 build 长期运行）

3. 团队能力:
   不熟悉 TensorRT/NVIDIA 生态 → vLLM 上手快
   有 NVIDIA TAM 支持或 TRT 经验 → TRT-LLM 能调到更优

4. 硬件:
   H100/Blackwell + FP8/FP4 → TRT-LLM 的 NVIDIA 编译链优势更容易体现
   A100 / 消费级 GPU → vLLM、SGLang 和 TRT-LLM 的差距更依赖具体模型与流量

5. 功能需求:
   需要 Speculative Decoding / 自定义采样 → 查各框架的支持程度
   需要 LoRA serving / 多模态 → vLLM 生态更丰富

6. 另一选择: SGLang
   RadixAttention（更激进的前缀缓存）
   某些场景性能超过 vLLM 和 TRT-LLM
   快速迭代中，值得关注
```

### 一个最小可执行的发布流程

如果你准备用 TRT-LLM 上生产，比较稳妥的流程通常是：

1. 在参考框架上先完成模型正确性验证，例如 vLLM 或 HF baseline。
2. 固定目标 GPU、TP 配置、最大序列长度和量化方式。
3. build engine，并把产物当成版本化发布物。
4. 用真实请求样本做功能回归和性能回归。
5. 灰度上线，重点观察 TTFT、吞吐、显存水位和错误率。
6. 保留上一版 engine，确保快速回滚。

## 常见问题

**Q: TensorRT-LLM 的 "build" 过程要多久？**

A: 取决于模型大小、GPU、量化方式、并行配置和当前 TensorRT-LLM 版本。可以先按“数分钟到数十分钟”做工程预期；每次更改 engine 约束、量化或并行方式，都要重新验证是否需要 rebuild。

**Q: TensorRT-LLM 支持哪些模型？**

A: 支持很多主流 LLM 和多种量化/并行配置，但支持范围随版本变化很快。新模型能否直接 build、是否支持目标量化和并行方式，应查当前 support matrix 和 examples，而不是只看模型家族名字。

**Q: 什么情况下不建议优先上 TRT-LLM？**

A: 如果你还在高频换模型、经常改上下文长度和量化配置、团队对 Triton/TensorRT 生态不熟，或者主要跑的是非 NVIDIA 数据中心 GPU，通常先用 vLLM/SGLang 更现实。TRT-LLM 的强项是把稳定配置压到更高性能，而不是在探索阶段提供最低摩擦。

**Q: SGLang 和这些推理框架的关系？**

A: SGLang 是另一个高性能推理框架，定位类似 vLLM。它的特色是 RadixAttention（更激进的前缀缓存）和结构化生成优化。性能在某些场景下超过 vLLM 和 TRT-LLM。

## 延伸阅读

- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [SGLang](https://github.com/sgl-project/sglang)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **TensorRT** | NVIDIA 的深度学习推理优化器/编译器，将模型编译为 GPU 特定的高效执行引擎 |
| **TensorRT-LLM** | TensorRT + LLM 特定优化（KV-Cache、Attention kernel、量化等）的组合 |
| **Build Engine** | 将模型编译为 TensorRT 引擎的离线步骤，需要指定 max_batch_size、max_seq_len 等 |
| **Kernel Fusion（算子融合）** | 将多个小操作合并为一个 GPU kernel，减少 kernel 启动开销和显存读写 |
| **Triton Inference Server** | NVIDIA 的模型服务器，提供 HTTP/gRPC API、请求调度、多模型管理等 |
| **In-flight Batching** | TensorRT-LLM 对 Continuous Batching 的叫法 |
