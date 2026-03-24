# Kubernetes GPU 调度

> 用 K8s 管理 GPU 集群——你的后端经验最能发挥的地方。

## 核心概念

### NVIDIA Device Plugin

```yaml
# Pod 请求 GPU 资源
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: training
    image: nvcr.io/nvidia/pytorch:24.01-py3
    resources:
      limits:
        nvidia.com/gpu: 8  # 请求 8 张 GPU
```

NVIDIA Device Plugin 让 K8s 能发现和分配 GPU 资源。配合 GPU Operator 自动管理驱动和 CUDA 运行时。

### GPU 调度器

| 调度器 | 特点 | 适用场景 |
|--------|------|---------|
| **默认 K8s** | 简单 GPU 分配 | 小规模 |
| **Volcano** | Gang Scheduling（一次分配所有需要的 GPU） | 分布式训练 |
| **Run:ai** | GPU 分时复用、配额管理 | 企业多团队共享 |
| **Kueue** | K8s 原生作业队列 | 批处理训练 |

### Gang Scheduling（为什么重要）

```
分布式训练需要同时启动所有 worker:

❌ 默认调度: Pod A 拿到 4 GPU, Pod B 等待 → 死锁
  Pod A 占着 GPU 等 Pod B，Pod B 拿不到 GPU 等 Pod A

✅ Gang Scheduling (Volcano):
  要么同时分配 8 GPU 给 Job（所有 Pod 一起启动）
  要么整个 Job 排队等待
```

```yaml
# Volcano Job 示例
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: llm-training
spec:
  minAvailable: 4  # 至少 4 个 Pod 同时可用才启动
  schedulerName: volcano
  tasks:
  - replicas: 4
    template:
      spec:
        containers:
        - name: worker
          image: training-image
          resources:
            limits:
              nvidia.com/gpu: 8
```

### MIG (Multi-Instance GPU)

K8s 中使用 MIG 切分单卡为多个实例（推理场景）：

```yaml
# 请求一个 MIG 实例 (如 A100 的 3g.20gb)
resources:
  limits:
    nvidia.com/mig-3g.20gb: 1
```

## 常见问题

**Q: K8s 适合大规模训练吗？**

A: 适合，但有挑战：gang scheduling、网络拓扑感知（NUMA 亲和、NVLink 拓扑）、大规模 checkpoint 存储。很多团队在 K8s 之上用 Volcano + 自定义调度策略。也有团队直接用 Slurm（HPC 传统调度器），不走 K8s。

## 延伸阅读

- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Volcano](https://volcano.sh/)
- [Kueue](https://kueue.sigs.k8s.io/)
