# 专家并行（Expert Parallelism）与 MoE

> Mixture of Experts 让模型参数量暴增但计算量可控——专家并行解决其分布式挑战。

## 核心概念

### Mixture of Experts (MoE)

```
标准 FFN:
  所有 token → 同一个 FFN → 输出

MoE FFN:
  所有 token → Router → 选择 Top-K Expert → 加权输出

┌─────────────────────────────────────────┐
│              Router (Gate)               │
│         softmax(x · W_gate)             │
│  → 为每个 token 选择 Top-K 个 expert     │
└────┬────┬────┬────┬────┬────┬────┬──────┘
     │    │    │    │    │    │    │
   ┌─▼─┐┌─▼─┐┌─▼─┐┌─▼─┐┌─▼─┐┌─▼─┐┌─▼─┐
   │E 0││E 1││E 2││E 3││E 4││E 5││E 6│ ... N experts
   └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
     │    │    │    │    │    │    │
     └────┴────┴────┴────┴────┴────┘
              加权求和输出
```

**MoE 的关键优势**：
- 总参数量大（如 8×FFN），但每个 token 只激活 Top-K 个 expert（如 K=2）
- 计算量只增加 ~K/N（而非 N 倍）
- 例：Mixtral 8x7B — 47B 总参数，但每 token 只用 12.9B

### 专家并行的必要性

```
Mixtral 8x7B 的 FFN:
  8 个 expert，每个 expert 是一个完整的 FFN (h=4096, intermediate=14336)
  每个 expert 参数: ~0.2B → 8 个 = 1.6B per layer × 32 layers = ~47B (含 attention)
  
  单卡放不下所有 expert → 需要将 expert 分布到不同 GPU
```

### All-to-All 通信

MoE 的通信模式是 **All-to-All**：每张卡需要把 token 发给持有对应 expert 的卡：

```
4 GPUs, 4 Experts (每卡 1 个):
  Token 分配: T1→E0, T2→E3, T3→E1, T4→E2, T5→E0, T6→E1, ...

  Before All-to-All:
    GPU 0: [T1, T2, T3, T4]  ← 本地 batch 的 token
    GPU 1: [T5, T6, T7, T8]
    ...

  After All-to-All:
    GPU 0 (E0): [T1, T5, ...]     ← 所有被路由到 E0 的 token
    GPU 1 (E1): [T3, T6, ...]     ← 所有被路由到 E1 的 token
    ...

  Expert 计算后，再做一次 All-to-All 把结果发回原来的 GPU

→ 每个 MoE 层需要 2 次 All-to-All
```

## 关键细节

### Router 设计

```python
class TopKRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        logits = self.gate(x)  # (batch, seq_len, num_experts)
        scores = F.softmax(logits, dim=-1)
        
        # 选择 Top-K
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        return top_k_scores, top_k_indices
```

### 负载均衡问题

如果 Router 总是把大量 token 路由给少数几个 expert → 负载不均衡 → 性能下降

解决方案：
1. **辅助损失（Auxiliary Loss）**：鼓励均匀分配
```python
# Load Balancing Loss
def load_balance_loss(router_logits, num_experts):
    # f_i: expert i 被选中的比例
    # P_i: router 分配给 expert i 的概率均值
    routing_probs = F.softmax(router_logits, dim=-1)
    expert_mask = F.one_hot(top_k_indices, num_experts).float()
    
    f = expert_mask.mean(dim=(0, 1))  # 每个 expert 的实际负载
    P = routing_probs.mean(dim=(0, 1))  # 每个 expert 的平均概率
    
    return num_experts * (f * P).sum()  # 均匀时最小
```

2. **Expert Capacity**：限制每个 expert 处理的最大 token 数
```
capacity_factor = 1.25
capacity = (total_tokens / num_experts) × capacity_factor × top_k
超过 capacity 的 token 被丢弃（用 residual 连接兜底）
```

### EP + TP + DP 的组合

```
典型配置 (128 GPUs):
  EP = 8 (每 8 卡共享一组 expert)
  TP = 8 (机内张量并行)  → EP 和 TP 可以共享同一维度
  DP = 16 (跨机数据并行)

实际中 EP 和 TP 常常在同一组 GPU 上:
  机内 8 卡: TP=8 用于非 MoE 层, EP=8 用于 MoE 层
  跨机: DP
```

### DeepSpeed-MoE

DeepSpeed 提供了高效的 MoE 实现：

```python
import deepspeed.moe

# 创建 MoE 层
moe_layer = deepspeed.moe.layer.MoE(
    hidden_size=4096,
    expert=FFN(4096, 14336),
    num_experts=8,
    ep_size=8,  # Expert Parallel degree
    k=2,        # Top-K
    capacity_factor=1.25,
    use_residual=True,
)
```

## 常见问题

**Q: MoE 推理时的挑战是什么？**

A: 推理时 batch size 通常很小（甚至 1），但 token 可能被路由到不同的 expert。这导致：(1) 每个 expert 只处理很少的 token，GEMM 效率低；(2) 需要加载所有 expert 的权重，显存占用大；(3) All-to-All 通信的开销在小 batch 时占比更高。

**Q: MoE 模型比同等计算量的 Dense 模型效果好多少？**

A: 通常在相同计算预算下，MoE 能达到比 Dense 模型更好的效果（Mixtral 8x7B 接近 LLaMA 2 70B）。但 MoE 的推理成本（显存）更高，且训练不稳定性也更大。

**Q: Expert Choice 和 Token Choice 有什么区别？**

A: Token Choice（标准方式）是每个 token 选择 Top-K expert。Expert Choice（Zhou et al., 2022）反过来——每个 expert 选择 Top-K token。Expert Choice 天然解决了负载均衡问题，但可能导致某些 token 不被任何 expert 处理。

## 延伸阅读

- [Switch Transformer](https://arxiv.org/abs/2101.03961) — Fedus et al., 2022
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al., 2024
- [DeepSpeed MoE](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
- [GShard](https://arxiv.org/abs/2006.16668) — Lepikhin et al., 2020
