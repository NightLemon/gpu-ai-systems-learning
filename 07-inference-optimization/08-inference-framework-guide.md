# 推理框架选型指南

> vLLM、TensorRT-LLM、SGLang、TGI——什么场景选什么？

## 选型决策矩阵

| 场景 | 推荐 | 次选 | 原因 |
|------|------|------|------|
| **快速原型/研究** | vLLM | SGLang | pip install 即用，模型支持最广 |
| **生产高性能 (H100)** | TRT-LLM | SGLang | FP8 优化深，kernel 最快 |
| **生产高性能 (A100)** | vLLM/SGLang | TRT-LLM | A100 上差距缩小，运维更简单 |
| **高频换模型** | vLLM | SGLang | 无需 build engine |
| **前缀缓存重度使用** | SGLang | vLLM | RadixAttention 更激进 |
| **结构化输出 (JSON)** | SGLang | vLLM | 原生 constrained decoding |
| **HuggingFace 生态** | TGI | vLLM | HF 原生，Inference Endpoints 集成 |
| **消费级 GPU / CPU** | llama.cpp | Ollama | GGUF 量化，资源要求低 |
| **嵌入/重排序模型** | vLLM | TGI | 支持 embedding 模型 |
| **多模态 (VLM)** | vLLM | SGLang | LLaVA 等视觉模型支持 |

## 关键对比

| 维度 | vLLM | TRT-LLM | SGLang | TGI |
|------|------|---------|--------|-----|
| **易部署** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **延迟 (TTFT)** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **吞吐 (tokens/s)** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **模型支持广度** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| **模型更新速度** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| **量化 (FP8)** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **前缀缓存** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| **可定制性** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ |

这个表描述的是大致倾向，不是稳定排名。框架迭代很快，同一框架在不同版本、不同模型、不同 GPU 上，结论都可能变化。真正可靠的做法是先缩小候选范围，再用自己的流量形态做 benchmark。

## 性能基准怎么看

```
看推理框架 benchmark 时注意:

1. 区分 Prefill-heavy vs Decode-heavy 场景:
   - 长输入短输出 (如摘要): Prefill 主导 → TRT-LLM 的 kernel 优化优势大
   - 短输入长输出 (如写作): Decode 主导 → 差距缩小（都是 memory-bound）

2. 区分低并发 vs 高并发:
   - 单请求延迟: 各框架差距不大
   - 高并发吞吐: TRT-LLM/SGLang 通常领先

3. 注意硬件:
   - H100 + FP8: TRT-LLM 独占优势
   - A100: 差距缩小（无 FP8 Tensor Core）

4. 不要只看 throughput:
   - 还要看 P99 延迟（长尾请求）
   - 还要看 TTFT（用户感知的响应速度）
```

## 一个更稳妥的选型流程

1. **先明确目标函数**
   是要更低 TTFT、更高吞吐、更低 GPU 成本，还是更快支持新模型？目标不同，结论会完全不同。

2. **再锁定候选框架**
   一般不要四个框架一起比。根据模型类型、硬件代际、功能需求先缩到 2 个。

3. **用自己的请求分布做压测**
   至少覆盖：输入长度分布、输出长度分布、并发区间、流式返回、P95/P99 指标。

4. **最后再考虑运维和发布成本**
   同样快 10%，但 build 流程更重、灰度更难、模型支持更慢，未必值得。

## 常见误区

### 误区 1：只看 tokens/s

服务端的高吞吐不等于用户体验好。很多场景更在意 TTFT、P99 延迟和请求抖动，而不是理论最大 tokens/s。

### 误区 2：拿公开 benchmark 直接做选型

公开 benchmark 往往只覆盖某个模型、某种输入长度和某种 batch 条件。你的真实流量如果以短请求、结构化输出或前缀复用为主，排名可能完全不同。

### 误区 3：把框架选择当成永久决定

推理框架比训练框架更容易更换。很多团队会把“快速支持新模型”和“最高性能生产栈”分开，不需要只押一套框架。

## 常见问题

**Q: 能不能同时用多个框架？**

A: 可以。一种常见做法是用 vLLM 做开发/测试，TRT-LLM 做生产部署。模型验证在 vLLM 上完成后，build TRT engine 上线。

**Q: 什么情况下 vLLM 通常是默认起点？**

A: 当你需要快速支持新模型、团队还在探索产品形态、或者运维和发布流程还没稳定下来时，vLLM 往往是更稳的起点。它不一定永远最快，但通常是综合摩擦最小的选择。

**Q: 什么情况下 TRT-LLM 值得付出额外复杂度？**

A: 当模型和配置相对稳定、硬件以 NVIDIA 数据中心卡为主、性能收益能直接转化为显著成本节省，或者你的 SLA 很紧时，TRT-LLM 的编译式优化更容易回本。

**Q: 这些框架的 API 兼容吗？**

A: vLLM、SGLang 提供较完整的 OpenAI-compatible API（chat/completions、streaming 等），TGI 也有类似接口。TRT-LLM 本身不直接暴露 HTTP API，需要结合 Triton Inference Server 或额外的服务层做 API 适配。切换框架时客户端代码**不一定零改动**——streaming 行为、tool calling 支持、错误码格式、参数支持范围（如 logprobs、suffix）各框架实现程度不同，上线前需要逐项验证。

## 延伸阅读

- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [SGLang](https://github.com/sgl-project/sglang)
- [LLM Inference Benchmark](https://github.com/bentoml/llm-bench)
