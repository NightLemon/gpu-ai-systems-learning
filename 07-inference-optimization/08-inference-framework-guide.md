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

## 常见问题

**Q: 能不能同时用多个框架？**

A: 可以。一种常见做法是用 vLLM 做开发/测试，TRT-LLM 做生产部署。模型验证在 vLLM 上完成后，build TRT engine 上线。

**Q: 这些框架的 API 兼容吗？**

A: vLLM、TGI、SGLang 都提供 OpenAI-compatible API。TRT-LLM 通过 Triton 后端也可以暴露 OpenAI API。所以切换框架时客户端代码通常不需要改。

## 延伸阅读

- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [SGLang](https://github.com/sgl-project/sglang)
- [LLM Inference Benchmark](https://github.com/bentoml/llm-bench)
