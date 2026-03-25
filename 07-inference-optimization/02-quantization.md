# 量化（Quantization）

> 用更少的 bit 表示模型权重（和/或激活值），从而减少显存占用和带宽需求。例如，将 FP16（16 bit）权重量化为 INT4（4 bit）后，模型体积变为 1/4，推理速度可以显著提升。

## 量化解决什么问题？

回想 Ch01 内存带宽章节的核心结论：LLM 推理的 decode 阶段是 **memory-bound**——瓶颈在于读权重的带宽，不在算力。

既然瓶颈在带宽，那减少需要读取的数据量就能直接加速。量化就是这个思路：把权重从 FP16（2 bytes/参数）压缩到 INT4（0.5 bytes/参数），模型体积变为 1/4，需要从显存读取的数据也变为 1/4。

当然代价是精度损失——但现代量化方法（GPTQ、AWQ 等）能在 INT4 下把质量损失控制在几乎不可感知的范围内。这是目前生产环境推理部署中最广泛使用的优化手段。

## 核心概念

### 为什么量化？

```
LLM 推理的 Decode 阶段是 Memory-bound:
  瓶颈 = 读取模型权重的带宽

理论带宽收益上限（权重读取量减少）:
  FP16 → INT8: 权重体积减半 → 理论上限 2x
  FP16 → INT4: 权重体积 1/4  → 理论上限 4x

单请求 decode 实测（含反量化开销等）:
  W8A16: ~1.5-1.8x
  W4A16: ~2-3x
  W8A8:  ~1.5-2x (Tensor Core INT8 GEMM 加速)

高并发服务场景:
  加速幅度会降低（大 batch 下逐渐从 memory-bound 转向 compute-bound）
  但量化的显存节省意味着能服务更多并发请求 → 系统级吞吐收益仍然显著
```

### 量化基础

$$x_{\text{quant}} = \text{round}\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}$$
$$x_{\text{dequant}} = (x_{\text{quant}} - \text{zero\_point}) \times \text{scale}$$

```
FP16 范围: ±65504, 精度 ~0.001
INT8 范围: -128 ~ 127, 只有 256 个值
INT4 范围: -8 ~ 7, 只有 16 个值

量化粒度（越细越准，但开销更大）:
  Per-tensor:  整个张量用一个 scale    → 最粗
  Per-channel: 每个输出通道一个 scale   → 常用
  Per-group:   每 G 个元素一个 scale    → 更精细（如 G=128）
  Per-token:   每个 token 一个 scale    → 用于激活值
```

### Weight-Only Quantization (W4A16 / W8A16)

```
只量化权重，激活值保持 FP16:

存储: 权重用 INT4/INT8
计算: 运行时将权重反量化为 FP16 → 做 FP16 GEMM

优点: 
  - 减少显存和带宽（权重读取量减少）
  - Decode 阶段显著加速（memory-bound → 权重更小 → 更快）
  - 精度损失小（权重分布比激活值稳定）

缺点:
  - Prefill 阶段加速有限（compute-bound，带宽不是瓶颈）
  - 反量化有开销
```

### Weight + Activation Quantization (W8A8 / W4A4)

```
权重和激活值都量化:

计算: INT8 × INT8 → INT32 → FP16（Tensor Core 加速）

优点:
  - 计算速度翻倍（INT8 GEMM 比 FP16 快 ~2x）
  - Prefill 和 Decode 都加速

缺点:
  - 激活值的量化更难（动态范围大、有 outliers）
  - 可能需要 calibration 或 QAT
```

## 关键细节

### 主流量化方案对比

| 方案 | 类型 | 精度 | 需要数据？ | 速度 | 代表 |
|------|------|------|-----------|------|------|
| **RTN** | W4A16 | ★★ | 否 | 快 | 最简单的 round-to-nearest |
| **GPTQ** | W4A16 | ★★★ | 校准集 | 较快 | 逐层误差最小化 |
| **AWQ** | W4A16 | ★★★★ | 校准集 | 快 | 保护重要权重不量化 |
| **SmoothQuant** | W8A8 | ★★★★ | 校准集 | 快 | 将激活值的难度转移到权重 |
| **FP8** | W8A8 | ★★★★★ | 否/少量 | 最快 | Hopper 原生支持 |
| **QAT/QLORA** | W4A16 | ★★★★ | 训练 | 慢 | 量化感知训练 |

### GPTQ

**核心思想**：逐层量化，每量化一个权重就调整其他权重来补偿误差。

```
GPTQ 算法（简化）:
  对每一层:
    1. 收集一小批 calibration 数据的激活值
    2. 计算 Hessian (H = X^T X)
    3. 按 Hessian 逆对角线的顺序处理每个权重:
       a. 量化当前权重
       b. 计算量化误差
       c. 将误差分散到尚未量化的权重上（OBC 方法）
    → 整层量化后，总误差最小化

优点: 只需几百条 calibration 样本，几分钟搞定
缺点: 顺序处理，不能完全并行
```

### AWQ (Activation-aware Weight Quantization)

**核心思想**：不是所有权重都同样重要。激活幅度大的通道对应的权重更重要，应该被保护。

```
AWQ 方法:
  1. 用 calibration 数据找出每个通道的激活幅度 s_j = mean(|X_j|)
  2. 按幅度缩放: W_j → W_j × s_j（放大重要权重）, X_j → X_j / s_j
  3. 对缩放后的 W 做标准量化
  
  放大后的重要权重在量化时精度更高
  数学上等价: (X/s) × (W×s) = X × W

优点: 比 GPTQ 更快（不需要逐列优化），效果更好
```

### FP8 量化

```
Hopper (H100) 原生支持 FP8 Tensor Core:
  E4M3: 4-bit 指数 + 3-bit 尾数 → 范围 ±448, 精度适合推理
  E5M2: 5-bit 指数 + 2-bit 尾数 → 范围 ±57344, 精度适合梯度

FP8 vs INT8:
  FP8: 不需要 zero_point，scale 更容易设置
  INT8: 均匀量化，对 outliers 不友好
  
  FP8 通常精度更好，且 Hopper 硬件原生支持
```

### SmoothQuant — 让 W8A8 可行

```
问题: 激活值有大 outliers，直接量化精度差

  激活值 X: [..., 0.1, 0.2, 100.0, 0.3, ...]
                                ↑ outlier
  INT8 量化: scale = 100/127 → 小值全被截到 0 → 精度崩

SmoothQuant:
  将激活值的"尖峰"转移给权重:
  Y = XW = (X/s)(sW)
  s = 按通道的激活值最大值的幂函数
  
  "平滑"后的 X 更容易量化
  虽然 W 的量化难度增加了，但权重是静态的，可以 offline 处理
```

## 代码示例

```python
# 使用 AutoGPTQ 量化
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,              # INT4 量化
    group_size=128,      # Per-group 量化
    desc_act=False,      # 不按激活值排序（更快）
)

# 加载并量化
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config,
)
model.quantize(calibration_data)  # 需要少量校准数据
model.save_quantized("llama-2-7b-gptq-int4")

# 使用 AWQ
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.quantize(
    tokenizer,
    quant_config={"w_bit": 4, "q_group_size": 128, "version": "GEMM"},
)
```

## 常见问题

**Q: INT4 量化对模型质量的影响有多大？**

A: 用 GPTQ/AWQ 这类高质量的 PTQ 方法：
- 7B 模型：PPL 增加 ~0.1-0.3（几乎无损）
- 13B+ 模型：PPL 增加 ~0.05-0.1（影响更小）
- 趋势：模型越大，量化越不敏感

**Q: 量化推理比 FP16 快多少？**

A: 以单请求 decode 为例（memory-bound 场景）：W4A16 ~2-3x，W8A8 ~1.5-2x，FP8 ~1.8-2x。大 batch 时加速幅度下降（逐渐变成 compute-bound）。但即便加速幅度缩小，量化省下的显存可以容纳更多并发请求，系统级吞吐收益往往比单请求加速更重要。

**Q: GGUF 量化是什么？**

A: GGUF 是 llama.cpp 使用的量化格式，支持多种量化级别（Q2_K 到 Q8_0）。它针对 CPU 推理优化，在消费级硬件上很流行。对于 GPU 推理，建议用 GPTQ/AWQ/FP8。

## 延伸阅读

- [GPTQ 论文](https://arxiv.org/abs/2210.17323) — Frantar et al., 2023
- [AWQ 论文](https://arxiv.org/abs/2306.00978) — Lin et al., 2023
- [SmoothQuant 论文](https://arxiv.org/abs/2211.10438) — Xiao et al., 2023
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — Micikevicius et al., 2022

---

## 术语表

| 术语 | 说明 |
|------|------|
| **量化（Quantization）** | 将浮点数用更少的 bit 表示（如 FP16→INT4），牺牲少量精度换取更小的模型体积和更快的速度 |
| **W4A16** | Weight 4-bit, Activation 16-bit，只量化权重，激活值保持 FP16 |
| **W8A8** | Weight 8-bit, Activation 8-bit，权重和激活值都量化，可用 INT8 Tensor Core 加速 |
| **PTQ（Post-Training Quantization）** | 训练后量化，不需要重新训练，只需少量校准数据 |
| **GPTQ** | 一种 PTQ 方法，通过误差补偿逐层量化，效果好但速度较慢 |
| **AWQ** | Activation-aware Weight Quantization，根据激活值幅度保护重要权重，比 GPTQ 更快效果更好 |
| **SmoothQuant** | 将激活值的“尖峰”转移给权重，让 W8A8 量化变得可行 |
| **Scale / Zero Point** | 量化参数。Scale 决定量化区间的缩放比例，Zero Point 决定零点的偏移 |
| **Per-group 量化** | 每 G 个元素共享一个 Scale（如 G=128），比 per-tensor 更精细但开销稍大 |
