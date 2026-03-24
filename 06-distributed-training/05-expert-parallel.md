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
# 注意: top_k_indices 来自前面 Router 的输出
def load_balance_loss(router_logits, top_k_indices, num_experts):
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

### 通信量和性能瓶颈分析

```
MoE 层的通信特性 vs Dense 层:

Dense (TP):
  通信: AllReduce (每层 2 次)
  模式: 规则，大小固定 = b × s × h × sizeof(dtype)
  
MoE (EP):
  通信: All-to-All (每层 2 次)
  模式: 不规则！每个 GPU 发送/接收的 token 数取决于路由结果
  大小: 每 GPU 发送 ~(b×s×K/N) × h × sizeof(dtype)
        但实际大小随路由分布波动

All-to-All vs AllReduce:
  AllReduce: 每对 GPU 交换 data_size/N → 带宽效率高 (Ring 算法)
  All-to-All: 每对 GPU 交换不同数量 → 难以用 Ring 优化
  → All-to-All 对网络延迟更敏感，跨机开销更大
```

**MoE 层的三大性能瓶颈**：

```
1. Token Dispatch 不均衡:
   某些 expert 收到大量 token → GEMM 大 → 成为关键路径
   其他 expert 空闲等待
   → 解决: Capacity Factor 截断 + 辅助损失

2. Small GEMM 效率低:
   即使负载均衡，每个 expert 处理的 token 数 = total_tokens / N
   当 N 很大时（如 64 experts），每个 expert 的 GEMM 很小
   → 小矩阵的 Tensor Core 利用率低
   → 解决: Grouped GEMM (将多个 expert 的 GEMM 合并成一个大 GEMM)

3. 跨机 All-to-All 开销:
   机内 NVLink All-to-All 尚可
   跨机 IB All-to-All 延迟高、效率低
   → 解决: EP 尽量放在机内，或用 hierarchical All-to-All
```

### MoE 训练 vs 推理的系统差异

```
训练:
  ✅ batch size 大 → 每个 expert 的 GEMM 规模合理
  ✅ All-to-All 的数据量和计算量可 overlap
  ✅ 主要优势: 相同 FLOPs 下模型能力更强
  ⚠️ 挑战: 训练不稳定（router 崩溃、expert 坍缩到几个）

推理:
  ✗ batch size 小（decode 时每请求 1 token）
  ✗ 所有 expert 权重必须在显存中 → 显存 = N × expert_size
  ✗ 每个 expert 收到 ≤1 个 token → GEMM 退化为 GEMV → 极低效率
  ✗ All-to-All 延迟在 decode 关键路径上
  
→ MoE 训练"省算力"，但推理可能比同等效果的 dense 模型更贵
→ 这就是为什么 MoE 需要特殊的推理优化（expert offloading、expert 稀疏化等）
```

### 近年的重要工程改进

| 技术 | 解决什么问题 | 代表 |
|------|------------|------|
| **Shared Expert** | 保底能力 + 提高路由稳定性 | DeepSeek-MoE |
| **Dropless MoE** | 避免 capacity 溢出丢 token | Megablocks |
| **Expert Placement** | 将热门 expert 复制到多卡减少通信 | DeepSeek-V2 |
| **Grouped GEMM** | 合并多个小 expert GEMM 提高 GPU 利用率 | Megablocks, Triton kernels |
| **Hierarchical All-to-All** | 先机内再跨机，减少跨机通信量 | Tutel |

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

A: 推理时 batch size 通常很小（甚至 1），但 token 可能被路由到不同的 expert。这导致：(1) 每个 expert 只处理极少 token，GEMM 退化为 GEMV；(2) 所有 expert 权重必须在显存中（Mixtral 8x7B 需要 ~94 GB FP16）；(3) All-to-All 延迟难以隐藏。所以 **MoE 模型训练省算力≠推理省成本**。

**Q: MoE 模型比同等计算量的 Dense 模型效果好多少？**

A: 在相同训练 FLOPs 下，MoE 通常优于 Dense（Mixtral 8x7B 接近 LLaMA 2 70B）。但 MoE 的推理成本、显存需求和训练稳定性都不如 Dense。选型要根据实际的训练预算 vs 推理预算比例来决定。

**Q: Expert Choice 和 Token Choice 有什么区别？**

A: Token Choice（标准方式）是每个 token 选择 Top-K expert。Expert Choice（Zhou et al., 2022）反过来——每个 expert 选择 Top-K token。Expert Choice 天然解决负载均衡问题，但可能导致某些 token 不被任何 expert 处理。

## 延伸阅读

- [Switch Transformer](https://arxiv.org/abs/2101.03961) — Fedus et al., 2022
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al., 2024
- [DeepSpeed MoE](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
- [GShard](https://arxiv.org/abs/2006.16668) — Lepikhin et al., 2020

---

## 术语表

| 术语 | 说明 |
|------|------|
| **MoE（Mixture of Experts）** | 混合专家模型。FFN 层被替换为多个“专家”，每个 token 只激活其中 Top-K 个 |
| **Expert** | MoE 中的一个 FFN 子网络。每个 Expert 结构相同但参数不同 |
| **Router / Gate** | 决定每个 token 被发送到哪些 Expert 的软路由网络 |
| **Expert Parallel（EP）** | 将不同 Expert 分布到不同 GPU 上的并行策略 |
| **All-to-All** | 每个 GPU 向每个其他 GPU 发送不同数据的通信操作。MoE 中用于将 token 发送到对应 Expert 所在的 GPU |
| **Capacity Factor** | 每个 Expert 可接受的最大 token 数的缩放系数，超过则丢弃 |
| **Auxiliary Loss（辅助损失）** | 加在训练损失上的额外项，鼓励 Router 将 token 均匀分配给各 Expert |
| **Grouped GEMM** | 将多个小 Expert 的 GEMM 合并为一个大 GEMM，提高 GPU 利用率 |
