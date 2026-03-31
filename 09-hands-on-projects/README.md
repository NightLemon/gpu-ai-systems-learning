# 09 - 实战项目

> 知识要通过动手巩固。每个项目都有明确的输入、输出和验收标准。

⚠️ **没有 A100 / 多卡？** 先看 [00-environment-setup.md](00-environment-setup.md)，里面按硬件条件规划了不同路线和最小实验清单。

| 项目 | 难度 | 预计时间 | 覆盖章节 | 最低硬件要求 |
|------|------|---------|---------|-------------|
| [00-environment-setup.md](00-environment-setup.md) | ⭐ | 1-2 小时 | 全部 | CPU 即可开始 |
| [01-gemm-optimization.md](01-gemm-optimization.md) | ⭐⭐⭐ | 2-4 周 | Ch02, Ch03 | 1× NVIDIA GPU (A100 推荐) |
| [02-multi-gpu-training.md](02-multi-gpu-training.md) | ⭐⭐⭐ | 2-3 周 | Ch05, Ch06 | 4-8× GPU |
| [03-inference-serving.md](03-inference-serving.md) | ⭐⭐ | 1-2 周 | Ch07 | 1-2× GPU (16GB+ 显存) |

## 项目通用要求

每个项目完成后应产出：

1. **性能数据表**：填入你实测的指标（模板在每个项目文档中）
2. **Profile 结果截图或文件**：至少一次 Nsight Systems 或 PyTorch Profiler 的 trace
3. **分析总结**：100-200 字说明你观察到的瓶颈和优化效果

## 如何判断结果是否合理

- 对比文档中给出的参考数值范围
- 你的 GPU 不同，绝对数值会不同，但**相对加速比**应该在合理范围内
- 如果结果离参考值差距过大（>2x），先检查测量方法是否正确
