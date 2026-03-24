# 投机解码（Speculative Decoding）

> 用小模型"猜"，大模型"验"——在不损失质量的前提下加速推理。

## 核心概念

### 问题

大模型 decode 阶段每步只生成 1 个 token，但读取全量权重，GPU 利用率极低。能否一次验证多个候选 token？

### Draft-Verify 范式

```
Draft Model (小模型，如 7B):
  快速自回归生成 γ 个候选 token: [t1, t2, t3, t4, t5]
  
Target Model (大模型，如 70B):
  一次 forward pass 并行验证所有候选
  → 接受前 k 个正确的 token + 生成 1 个额外 token

示例 (γ=5):
  Draft:  "The" → "capital" → "of" → "France" → "is" → "Rome"
  Target: 验证 → "capital"✓ "of"✓ "France"✓ "is"✓ "Rome"✗ → "Paris"
  
  结果: 一次 target forward 生成了 5 个 token (4 accepted + 1 new)
  加速: 原本需要 5 次 target forward → 现在只需 1 次 draft 序列 + 1 次 target
```

### 数学保证：输出分布完全一致

投机解码使用 **rejection sampling**，保证最终输出的概率分布和单独运行 target model 完全一致：

$$P(\text{accept } t_i) = \min\left(1, \frac{p_{\text{target}}(t_i)}{p_{\text{draft}}(t_i)}\right)$$

- 如果 $p_{\text{target}} \geq p_{\text{draft}}$：一定接受
- 如果 $p_{\text{target}} < p_{\text{draft}}$：以 $p_{\text{target}}/p_{\text{draft}}$ 的概率接受
- 被拒绝时：从修正分布 $\text{norm}(\max(0, p_{\text{target}} - p_{\text{draft}}))$ 采样

**这不是近似——是数学上精确等价的。**

## 关键细节

### 加速效果分析

```
设:
  γ = draft 长度（每轮 draft model 生成的候选 token 数）
  α = 平均单 token acceptance rate (0~1)
  cost_ratio = draft_time / target_time

直观估算:
  每轮期望接受约 γ × α 个 draft token + 1 个 target 新生成的 token
  → 每轮期望产出 ≈ γα + 1 个 token
  → 每轮消耗 = 1 次 target forward + γ 次 draft forward

加速比 ≈ (γα + 1) / (1 + γ × cost_ratio)

当 draft 模型远小于 target 时 cost_ratio ≈ 0:
  加速比 ≈ γα + 1

示例 (cost_ratio ≈ 0.05):
  α=0.8, γ=5:  加速 ≈ (5×0.8+1) / (1+5×0.05) = 5/1.25 = 4.0 → 实际约 2-3x
  α=0.8, γ=10: 加速 ≈ (10×0.8+1)/ (1+10×0.05) = 9/1.5  = 6.0 → 实际约 3-4x
  α=0.5, γ=5:  加速 ≈ (5×0.5+1) / (1+5×0.05) = 3.5/1.25 = 2.8 → 实际约 2x

注: 严格推导需要考虑 rejection sampling 的几何分布特性
    （每个 position 独立以概率 1-α 被拒），具体见原论文。
    上述 γα+1 是常用的直观近似，在 α 较高时偏乐观。

影响 α 的因素:
  - draft/target 模型的相似度（同系列的大小模型 α 高）
  - 任务类型: 代码补全等确定性高的任务 → α 高 → 加速大
  - 任务类型: 创意写作等随机性高的任务 → α 低 → 加速小
  - temperature: 低温 α 高，高温 α 低
```

### 变种

| 方案 | Draft 来源 | 优点 | 缺点 |
|------|-----------|------|------|
| **标准 Speculative** | 独立小模型 | 灵活，draft 质量可控 | 需要额外加载小模型 |
| **Self-Speculative** | 跳层/early exit | 不需要额外模型 | draft 质量受限 |
| **Medusa** | 额外的预测头 | 不改模型结构 | 需要训练预测头 |
| **Lookahead** | N-gram Cache | 零额外显存 | 需要足够长的上下文 |
| **Eagle** | 特征级 draft | 高 acceptance rate | 需要训练 |

### Medusa 方案

```
在模型最后一层添加多个额外的预测头:

Standard: ... → Layer N → LM Head → token_1
Medusa:   ... → Layer N → LM Head → token_1
                        → Medusa Head 1 → token_2 (预测)
                        → Medusa Head 2 → token_3 (预测)
                        → Medusa Head 3 → token_4 (预测)

→ 一次 forward 同时预测多个未来 token
→ 用 tree attention 验证多条候选路径
→ 不需要额外的 draft model
```

## 常见问题

**Q: Speculative Decoding 和 Beam Search 有什么区别？**

A: 完全不同。Beam Search 是一种搜索策略（维护多个候选序列取最高总分）。Speculative Decoding 是一种加速技术（用小模型猜、大模型验），最终结果和不用它时完全一致。

**Q: 什么时候 Speculative Decoding 不值得用？**

A: (1) 大 batch 推理时——GPU 已经很忙了，draft model 增加的开销大于节省；(2) α 很低时——大小模型差距太大，几乎每个 token 都被拒绝；(3) prefill 为主的场景——speculative decoding 只加速 decode。

**Q: vLLM 支持 Speculative Decoding 吗？**

A: 支持。vLLM 提供了多种 speculative decoding 后端（独立 draft model、Medusa、Eagle 等），通过配置启用。

## 延伸阅读

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2023
- [Medusa: Simple LLM Inference Acceleration](https://arxiv.org/abs/2401.10774)
- [Eagle: Speculative Sampling](https://arxiv.org/abs/2401.15077)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Speculative Decoding（投机解码）** | 用小模型快速生成候选 token，大模型一次性并行验证，加速推理 |
| **Draft Model** | 用于生成候选 token 的小模型（快但不够准） |
| **Target Model** | 最终判定的大模型（慢但准） |
| **Rejection Sampling** | 数学上保证输出分布与单独运行 target model 完全一致的采样方法 |
| **γ (gamma)** | 每轮 draft model 生成的候选 token 数 |
| **α (alpha)** | 平均单 token 被接受的概率（acceptance rate） |
| **Medusa** | 一种变体，在模型最后一层加多个预测头，同时预测多个未来 token |
