# 实战：推理 Serving 部署

> 用 vLLM 部署一个 LLM 推理服务，做一次完整的性能调优。

## 项目目标

部署 LLaMA 2 7B（或同等模型），对比不同配置的性能差异，掌握推理优化调优。

## 环境准备

```bash
pip install vllm
# 模型: 需下载 meta-llama/Llama-2-7b-hf 或使用开放模型
```

## 步骤

### Step 1: 基础部署

```bash
# 启动 vLLM serving
vllm serve meta-llama/Llama-2-7b-hf \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# 测试
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-2-7b-hf", "prompt": "Hello", "max_tokens": 100}'
```

### Step 2: 压测与 Baseline 指标

```python
# 用 benchmark 工具测量
# vLLM 自带 benchmark 脚本
python -m vllm.entrypoints.openai.api_server &

python benchmarks/benchmark_serving.py \
    --model meta-llama/Llama-2-7b-hf \
    --num-prompts 100 \
    --request-rate 10

# 记录: TTFT, TPOT, Throughput (tokens/s)
```

### Step 3: 量化对比

```bash
# FP16 (baseline)
vllm serve meta-llama/Llama-2-7b-hf

# AWQ INT4
vllm serve TheBloke/Llama-2-7b-AWQ --quantization awq

# GPTQ INT4
vllm serve TheBloke/Llama-2-7b-GPTQ --quantization gptq

# 对比: 吞吐量、延迟、显存占用、输出质量
```

### Step 4: 调优参数

```bash
# 调整 block size
vllm serve ... --block-size 32  # vs 默认 16

# 开启 prefix caching
vllm serve ... --enable-prefix-caching

# 调整 max-num-seqs
vllm serve ... --max-num-seqs 512  # 更高并发

# Tensor Parallel (多卡)
vllm serve ... --tensor-parallel-size 2

# Chunked prefill
vllm serve ... --enable-chunked-prefill
```

### Step 5: 输出对比分析表

```
配置                  TTFT   TPOT   Throughput  显存
────────────────────────────────────────────────────────
FP16, 1 GPU           ?ms    ?ms    ? tok/s     ?GB
AWQ INT4, 1 GPU       ?ms    ?ms    ? tok/s     ?GB
FP16, 2 GPU TP        ?ms    ?ms    ? tok/s     ?GB
FP16 + Prefix Cache   ?ms    ?ms    ? tok/s     ?GB
```

## 参考资料

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- [LLM Inference Benchmark](https://github.com/bentoml/llm-bench)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Serving** | 将模型部署为在线服务，接受 HTTP 请求并返回生成结果 |
| **Throughput** | 系统级吞吐量，衡量单位时间内能处理多少 token |
| **TTFT** | Time to First Token，用户发消息到看到第一个字的延迟 |
| **TPOT** | Time per Output Token，生成每个后续 token 的平均时间 |
| **压测（Benchmarking）** | 用工具模拟高并发请求，测量系统在不同负载下的性能表现 |
