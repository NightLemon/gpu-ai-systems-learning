# Kubernetes GPU 调度

> 用 K8s 管理 GPU 集群——不只是 device plugin，还需要拓扑感知和 gang scheduling。

## 核心概念

### GPU 管理组件栈

```
┌─────────────────────────────────────────────┐
│          Kubernetes API Server               │
├─────────────────────────────────────────────┤
│  Scheduler (+ Volcano/Kueue 扩展)            │  ← 调度决策
├─────────────────────────────────────────────┤
│  GPU Operator (DaemonSet)                    │  ← 自动管理以下组件
│  ├── NVIDIA Driver                           │
│  ├── NVIDIA Device Plugin                    │  ← 让 K8s 发现 GPU
│  ├── NVIDIA Container Toolkit (nvidia-ctk)   │  ← 容器内 GPU 访问
│  ├── DCGM Exporter                           │  ← GPU 监控指标
│  ├── MIG Manager                             │  ← MIG 配置管理
│  └── GPU Feature Discovery (GFD)             │  ← GPU 型号/拓扑标签
└─────────────────────────────────────────────┘

GPU Operator 的价值: 不需要手动在每个节点装驱动和插件
  → 一个 Helm chart 搞定所有 GPU 节点的驱动/runtime/监控
```

### 基本 GPU Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  containers:
  - name: cuda
    image: nvcr.io/nvidia/pytorch:24.01-py3
    resources:
      limits:
        nvidia.com/gpu: 2   # 请求 2 张 GPU
    # 注意: K8s 默认不保证这 2 张卡在同一 NVLink domain
    # 可能分到不同 NUMA 节点的卡 → 性能下降
```

### Gang Scheduling — 分布式训练的必要条件

```
问题: 分布式训练需要同时启动所有 worker

❌ 默认 K8s 调度:
  Job 需要 4 个 Pod × 8 GPU = 32 GPU
  集群只剩 24 GPU 空闲
  → 调度了 3 个 Pod，第 4 个排队
  → 3 个 Pod 占着 24 GPU 空等 → 资源浪费 + 死锁

✅ Gang Scheduling (Volcano/Kueue):
  "要么 4 个 Pod 全部成功调度，要么一个都不调度"
  → 集群 GPU 不够时整个 Job 排队，不会占着资源不释放
```

```yaml
# Volcano Gang Scheduling 示例
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: llm-training
spec:
  minAvailable: 4              # 至少 4 个 Pod 同时调度才启动
  schedulerName: volcano
  plugins:
    svc: ["--publish-not-ready-addresses"]  # headless service for NCCL
  tasks:
  - replicas: 4
    name: worker
    template:
      spec:
        hostNetwork: true      # NCCL/RDMA 通常需要 host networking
        containers:
        - name: trainer
          image: my-training-image:latest
          command: ["torchrun"]
          args:
            - "--nproc_per_node=8"
            - "--nnodes=4"
            - "--rdzv_backend=c10d"
            - "--rdzv_endpoint=$(MASTER_ADDR):29500"
            - "train.py"
          resources:
            limits:
              nvidia.com/gpu: 8
              rdma/rdma_shared_device_a: 1  # RDMA 设备声明
          env:
            - name: NCCL_IB_HCA
              value: "mlx5_0,mlx5_1,mlx5_2,mlx5_3"
            - name: NCCL_DEBUG
              value: "INFO"
        volumes:
        - name: shm
          emptyDir:
            medium: Memory      # /dev/shm for NCCL shared memory
            sizeLimit: "32Gi"
```

## 关键细节

### 拓扑感知调度

默认 K8s 不感知 GPU 之间的互联拓扑，可能出现：

```
❌ 不感知拓扑:
  Pod 请求 4 GPU → 分到 GPU 0,1 (NUMA 0, NVLink) + GPU 4,5 (NUMA 1, NVLink)
  → GPU 0↔GPU 4 只能走 PCIe → P2P 慢 5-10x
  → TP 性能崩

✅ 拓扑感知:
  Pod 请求 4 GPU → 分到 GPU 0,1,2,3 (同一 NVSwitch domain)
  → 所有卡都有 NVLink 全互联 → TP 性能最优

实现方式:
  1. GPU Feature Discovery (GFD) 为节点打标签:
     nvidia.com/gpu.product=A100
     nvidia.com/mig.strategy=single
     
  2. Topology-aware GPU Scheduling:
     - NVIDIA 的 GPU Topology Daemon
     - 或用 nodeAffinity + 手动标签
     - 确保同一 Pod 的 GPU 在同一 NVLink domain

  3. NUMA 亲和性:
     CPU ↔ GPU ↔ 网卡 应在同一 NUMA 节点
     否则 CPU↔GPU PCIe 传输走跨 NUMA QPI → 带宽减半
```

### K8s vs Slurm：训练集群的实际选择

| 方面 | Kubernetes | Slurm |
|------|-----------|-------|
| **设计初衷** | 微服务编排 | HPC 作业调度 |
| **GPU 调度** | 需要额外组件（Volcano等） | 原生支持 GRES (GPU) |
| **拓扑感知** | 需要额外配置 | 原生支持 (`--gres-flags=enforce-binding`) |
| **网络** | 通常 overlay → NCCL 需要 hostNetwork | 裸金属网络，RDMA 开箱即用 |
| **多租户** | 好（namespace 隔离） | 弱（队列 + 账号） |
| **弹性** | 好（Pod 自动调度） | 弱（固定节点分配） |
| **运维** | 复杂（K8s + GPU Operator + CNI + ...） | 相对简单 |
| **生态** | 推理服务、MLOps 全链路 | 纯训练 |

```
实际趋势:
  纯训练集群（>256 GPU，固定团队）: Slurm 仍然主流
    → 更简单，RDMA 开箱即用，拓扑感知原生支持
    
  混合集群（训练 + 推理 + 开发）: Kubernetes
    → 多租户、弹性资源共享、与推理/MLOps 统一平台
    
  大厂做法: 训练用 Slurm，推理用 K8s，或者 K8s 上跑 Slurm
```

### MIG (Multi-Instance GPU)

```
A100/H100 支持将单卡切分为最多 7 个独立 GPU 实例:

H100 MIG 配置示例:
  1 × 7g.80gb  (整卡)
  1 × 4g.40gb + 1 × 3g.40gb
  7 × 1g.10gb (最细粒度)

K8s 中使用:
  nvidia.com/mig-3g.40gb: 1    # 请求一个 3g.40gb 的 MIG 实例

适用场景:
  ✅ 推理服务: 多个小模型共享一张卡
  ✅ 开发环境: 多人共享 GPU 做实验
  ✗ 训练: 通常不用 MIG（需要整卡算力 + NVLink 互联）
```

## 常见问题

**Q: 为什么分布式训练经常需要 `hostNetwork: true`？**

A: NCCL 使用 RDMA (InfiniBand/RoCE) 进行 GPU 间通信。K8s 默认的 overlay 网络（如 Calico/Flannel）不支持 RDMA。使用 hostNetwork 让 Pod 直接使用宿主机网络栈，NCCL 才能找到并使用 RDMA 设备。

**Q: 训练 Pod 需要哪些特殊的 volume/device？**

A: 除了 GPU，通常还需要：(1) 扩大的 `/dev/shm`（NCCL shared memory，默认 64MB 不够）。(2) RDMA 设备（通过 `k8s-rdma-shared-dev-plugin` 暴露）。(3) 共享文件系统（训练数据和 checkpoint）。(4) 有时需要 `IPC_LOCK` capability（pin memory）。

**Q: Kueue 和 Volcano 怎么选？**

A: Kueue 是 K8s SIG 官方项目，更轻量且与 K8s 生态集成更好。Volcano 功能更丰富（支持 MPI Job、TensorFlow Job 等自定义 CRD）。小规模选 Kueue，大规模或需要复杂调度策略选 Volcano。

## 延伸阅读

- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Volcano](https://volcano.sh/) — GPU Gang Scheduling
- [Kueue](https://kueue.sigs.k8s.io/) — K8s 原生作业队列
- [Slurm + GPU](https://slurm.schedmd.com/gres.html) — Slurm GPU 调度

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Kubernetes (K8s)** | 当前最主流的容器编排平台，管理容器化应用的部署、扩缩容和调度 |
| **GPU Operator** | NVIDIA 提供的 K8s Operator，自动管理 GPU 驱动、Device Plugin、监控等组件 |
| **Device Plugin** | K8s 插件，让调度器能发现和分配 GPU 资源（`nvidia.com/gpu`） |
| **Gang Scheduling** | 一组 Pod 要么全部成功调度，要么一个都不调度。分布式训练必须，防止资源死锁 |
| **Topology-aware Scheduling** | 考虑 GPU 的物理互联拓扑（NVLink、NUMA）来分配 GPU，避免分配到不同 NVLink 域的卡 |
| **Volcano** | 专为 HPC/AI 场景设计的 K8s 调度器，支持 Gang Scheduling |
| **Kueue** | K8s SIG 官方的作业队列管理器，更轻量 |
| **Slurm** | 传统 HPC 集群的作业调度器，原生支持 GPU 和拓扑感知 |
| **MIG** | Multi-Instance GPU，将一张 GPU 切分为多个独立实例，适合推理场景多模型共享一张卡 |
| **hostNetwork** | K8s Pod 直接使用宿主机网络（而非虚拟网络），NNCL/RDMA 训练通常需要 |
