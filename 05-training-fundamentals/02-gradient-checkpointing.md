# 梯度检查点（Gradient Checkpointing）

> 用额外 ~30% 的重复计算（时间）换取 ~60-70% 的激活值显存节省（空间）。在显存不足以缓存所有中间层输出时，可以只保存部分层的结果，其余的在反向传播时重新计算。

## 问题场景：显存不够怎么办？

你在训练一个 7B 模型时，发现 OOM（Out of Memory）了。查看显存分布：模型参数 14 GB，梯度 14 GB，优化器状态 84 GB……等等，还有一大块是“激活值”——前向传播时每一层的中间输出，为了反向传播时算梯度而保留。对于大 batch size 和长序列，激活值可能比模型本身还占显存。

Gradient Checkpointing 的思路很直接：**不保存所有层的激活值，只保存部分“检查点”。反向传播时，从最近的检查点开始重新前向计算以恢复需要的激活值**。代价是额外的计算时间（约 30%），但显存可以省 60-70%——在显存紧张时非常值得。

## 核心概念

### 问题

反向传播需要前向传播的中间激活值。标准做法是全部缓存 → 显存随层数线性增长。

### 解决方案

只保留部分层的激活值（checkpoint），其余在反向传播时重算：

```
标准: 保存所有层的激活值
  Forward:  → [Act0] → [Act1] → [Act2] → ... → [ActN] → Loss
  Backward: [ActN] → grad → [ActN-1] → grad → ... → [Act0]
  显存: O(N) 层的激活值

Checkpointing: 只保存每隔 K 层的激活值  
  Forward:  → [Act0✓] → [Act1] → [Act2✓] → ... → Loss
  Backward: 需要 Act1? → 从 Act0✓ 重新计算 Act1 → 计算梯度
  显存: O(N/K) + 重计算开销 ~30%
```

### PyTorch 使用

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Model(nn.Module):
    def forward(self, x):
        for block in self.blocks:
            # 使用 checkpoint: forward 不保存中间激活值，backward 重算
            x = checkpoint(block, x, use_reentrant=False)
        return x
```

### Selective Checkpoint (Megatron-LM)

不是整层重算，而是选择性地只重算"显存大但计算便宜"的操作：
- **重算**：Attention Score ($N^2$ 显存)、Softmax、Dropout
- **保留**：Linear 层输出（显存小但重算贵）

→ 显存节省 50-70%，重计算开销 < 5%

## 延伸阅读

- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) — Chen et al., 2016
- [PyTorch Checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **激活值（Activations）** | 前向传播时每一层的中间输出。反向传播计算梯度时需要用到，因此默认会被缓存在显存中 |
| **Gradient Checkpointing** | 只保存部分层的激活值，其余层在反向传播时从最近的检查点重新计算，用时间换显存 |
| **Selective Checkpoint** | Megatron-LM 的策略：只重计算显存大但计算快的操作（如 Attention Score），保留显存小但重算慢的（如 Linear 输出） |
