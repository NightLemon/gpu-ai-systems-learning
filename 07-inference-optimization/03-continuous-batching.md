# Continuous Batching

> 将 LLM 推理服务的请求调度从“按 batch 调度”升级为“按 iteration 调度”。

## 传统 Batching 的问题

你架设了一个 LLM 推理服务，同时有 3 个用户请求。用户 A 要生成 100 个 token，用户 B 只要 10 个，用户 C 要 50 个。

传统做法（Static Batching）是把它们凑成一个 batch，等**最慢的那个**完成后才处理下一批。B 在第 10 步就完了，但它占着的 GPU 资源要一直空等到 A 生成完 100 个 token——资源浪费严重。

**Continuous Batching** 解决这个问题：每一次 decode 迭代都可以加入新请求或移除已完成的请求。B 完成后立即释放，新请求 D 马上填进来。GPU 始终在做有用工作，吞吐量提升 2-8 倍。

这是现代 LLM 推理引擎（vLLM、TGI、SGLang 等）的标配功能。

## 核心概念

### Static Batching 的问题

```
Static Batching: 等凑齐一个 batch，一起处理，全部完成后再处理下一个

Request A: "Hello" → 生成 100 tokens
Request B: "Hi"    → 生成 10 tokens  
Request C: "Hey"   → 生成 50 tokens

时间线:
  ┌─ Batch ──────────────────────────────────────────────┐
  │ A: [████████████████████████████████████████████████] │ 100 tokens
  │ B: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] │ 10 tokens + 90 idle
  │ C: [████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░] │ 50 tokens + 50 idle
  └──────────────────────────────────────────────────────┘
  
B 在第 10 步就完成了，但必须等 A 生成完 100 tokens 才能释放资源
→ GPU 利用率低, 延迟高
```

### Continuous Batching (Iteration-level Scheduling)

```
核心思想: 每次 decode 迭代都可以加入/移除请求

Iteration 1-10:   [A, B, C] 一起生成
Iteration 11:      B 完成! → 移除 B, 加入新请求 D
Iteration 11-50:  [A, C, D] 一起生成
Iteration 51:      C 完成! → 移除 C, 加入新请求 E
Iteration 51-100: [A, D, E] 一起生成

时间线:
  A: [████████████████████████████████████████████████████]
  B: [████████]
  C: [████████████████████████████]
  D:           [██████████████████████████████████████████]
  E:                              [██████████████████████]

→ GPU 始终在做有用计算
→ 吞吐量提升 2-8x（取决于长度分布）
```

### Prefill-Decode 分离

```
挑战: Prefill (计算密集) 和 Decode (带宽密集) 混在一起效率低

方案 1: Chunked Prefill
  将长 prompt 分成多个 chunk，与 decode 请求交替处理
  → 避免 prefill 霸占 GPU 导致 decode 请求延迟飙升

方案 2: Prefill-Decode Disaggregation
  Prefill 和 Decode 用不同的 GPU 集群
  Prefill GPU: 高算力利用率
  Decode GPU: 大 batch，高带宽利用率
  → Splitwise / DistServe 等论文的思路
```

## 关键细节

### 调度策略

```
FCFS (First Come First Serve):
  按请求到达顺序处理
  简单，但可能让短请求等很久

SJF (Shortest Job First):
  优先处理预计较短的请求
  → 需要预估生成长度

Preemption (vLLM):
  当显存不足时，暂停低优先级请求（swap KV-Cache 到 CPU）
  → 避免 OOM，保证系统稳定
```

### 吞吐量公式

```
Static Batching 吞吐量:
  Throughput = batch_size / max_sequence_length × tokens_per_second

Continuous Batching 吞吐量:
  Throughput ≈ batch_size / avg_sequence_length × tokens_per_second

当 max >> avg 时（常见情况），continuous batching 优势巨大
```

## 常见问题

**Q: Continuous Batching 的实现难点？**

A: 主要挑战是 KV-Cache 管理——不同请求的 KV-Cache 长度不同且动态变化，需要高效的内存分配/回收策略。这正是 vLLM 的 PagedAttention 解决的问题。

**Q: 哪些推理框架支持 Continuous Batching？**

A: vLLM、TensorRT-LLM、TGI (HuggingFace)、Triton Inference Server 都支持。它已经成为生产级推理的标配。

## 延伸阅读

- [Orca: A Distributed Serving System for Transformer-Based LLMs](https://www.usenix.org/conference/osdi22/presentation/yu) — 首次提出 iteration-level scheduling
- [Splitwise: Efficient Generative LLM Inference](https://arxiv.org/abs/2311.18677)
- [DistServe: Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Static Batching** | 传统方式：凑齐一个 batch，等最长的请求完成后才处理下一批。短请求空等 |
| **Continuous Batching** | 每次 decode 迭代都可以加入/移除请求，GPU 始终在做有用工作 |
| **Iteration-level Scheduling** | Continuous Batching 的另一个名称，强调调度粒度是每次 decode 迭代 |
| **Chunked Prefill** | 将长 prompt 分成小块，与 decode 交替执行，避免 prefill 霍占 GPU |
| **TTFT** | Time to First Token，从收到请求到输出第一个 token 的延迟 |
| **TPOT** | Time per Output Token，生成每个 token 的平均时间 |
