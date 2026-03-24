# 监控与可观测性

> 训练跑起来后，怎么判断效率高不高？怎么发现问题？

## 核心概念

### 关键指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **MFU** | Model FLOPs Utilization | >40% 良好, >50% 优秀 |
| **GPU Utilization** | SM 活跃时间占比 | >90% |
| **GPU Memory** | 显存使用率 | 80-95%（太低浪费，太高OOM） |
| **Throughput** | tokens/s 或 samples/s | 越高越好 |
| **Loss Curve** | 训练损失下降 | 平滑下降，无突变 |

### DCGM (Data Center GPU Manager)

```bash
# NVIDIA DCGM: GPU 集群级监控
dcgmi discovery -l           # 列出 GPU
dcgmi diag -r 3             # 运行健康检查
dcgmi stats -g 0 --enable   # 启用统计

# 导出到 Prometheus + Grafana
# dcgm-exporter 作为 DaemonSet 运行在每个节点
```

关键 DCGM 指标：
```
DCGM_FI_DEV_GPU_UTIL          GPU 利用率
DCGM_FI_DEV_SM_CLOCK           SM 时钟频率（降频=温度问题）
DCGM_FI_DEV_MEM_COPY_UTIL      显存带宽利用率
DCGM_FI_DEV_GPU_TEMP           温度
DCGM_FI_DEV_POWER_USAGE        功耗
DCGM_FI_DEV_NVLINK_BANDWIDTH   NVLink 带宽
DCGM_FI_DEV_XID_ERRORS         XID 错误（硬件/驱动问题）
```

### MFU 计算

```python
def compute_mfu(model_params, tokens_per_step, step_time, gpu_count, gpu_peak_flops):
    """
    model_params: 模型参数量
    tokens_per_step: 每步处理的 token 数
    step_time: 每步耗时（秒）
    gpu_count: GPU 数量
    gpu_peak_flops: 单卡峰值 FLOPS (如 H100 BF16 = 1979e12)
    """
    flops_per_step = 6 * model_params * tokens_per_step  # 近似: 6P per token
    achieved_flops = flops_per_step / step_time
    peak_flops = gpu_count * gpu_peak_flops
    mfu = achieved_flops / peak_flops
    return mfu

# 例: 7B 模型, 8 × H100, batch=32×2048, 1.5s/step
mfu = compute_mfu(7e9, 32*2048, 1.5, 8, 1979e12)
# → ~0.36 (36%)
```

### 常见故障排查

```
症状                      可能原因                    排查
────────────────────────────────────────────────────────
GPU 利用率低              数据加载瓶颈                Profile DataLoader
GPU 利用率波动            通信等待                    查看 NCCL trace
Loss 突然变 NaN           FP16 溢出 / 学习率过大     检查 loss scale
训练突然变慢              GPU 降频（温度/功耗）       nvidia-smi -q -d TEMPERATURE
某些 GPU 慢               Straggler (故障 GPU)        比较各卡 step time
OOM                       显存不足                    减 batch / 用 checkpoint
XID 错误                  硬件问题                    检查 dmesg / DCGM
```

## 延伸阅读

- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/)
- [dcgm-exporter (Prometheus)](https://github.com/NVIDIA/dcgm-exporter)
- [Weights & Biases](https://wandb.ai/) — 训练实验跟踪
