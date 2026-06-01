# 实战踩坑案例

> 最后核查：2026-06-01。本页收集 GPU 集群、Kubernetes 和容器镜像在真实训练/推理环境里最常见的版本错配与排障案例。

## 先建立版本账本

很多基础设施问题不是某个组件“坏了”，而是版本组合没有被当成一个整体管理。排障前先记录这一组信息：

| 层级 | 必须记录 | 为什么重要 |
|------|----------|------------|
| 物理机 | GPU 型号、驱动版本、Fabric Manager、网卡固件 | 决定 CUDA runtime、NVLink/NVSwitch、GPUDirect RDMA 能不能工作 |
| 容器镜像 | 镜像 tag、CUDA、PyTorch、NCCL、Transformer Engine、TensorRT-LLM | `nvcr.io/nvidia/pytorch:24.01-py3` 和 `26.04-py3` 不是“新旧补丁”关系，而是整套栈不同 |
| Kubernetes | K8s 版本、containerd/Docker、NVIDIA Container Toolkit、GPU Operator、device plugin | 决定 Pod 里能否看到 `/dev/nvidia*`、MIG/CDI/时分复用怎么暴露 |
| 网络 | IB/RoCE、CNI、RDMA device plugin、`nvidia-peermem`、MTU/PFC/ECN | 决定 NCCL 走 RDMA 还是悄悄退回 socket |
| 应用 | `torchrun`/vLLM/TensorRT-LLM 参数、batch/sequence、并行策略 | 决定是不是把基础设施问题误判成模型或框架问题 |

最小基线脚本：

```bash
nvidia-smi
nvidia-smi topo -m
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
python -m torch.utils.collect_env
```

K8s 节点上再补：

```bash
kubectl get nodes -o wide
kubectl describe node <node> | grep -A5 -E "Capacity|Allocatable|nvidia.com"
kubectl -n gpu-operator get pods -o wide
kubectl -n gpu-operator logs ds/nvidia-device-plugin-daemonset --tail=100
```

## 快速症状表

| 现象 | 优先怀疑 | 第一条验证命令 |
|------|----------|----------------|
| 容器里 `nvidia-smi` 不存在或看不到 GPU | container runtime / CDI / runtimeClass | `kubectl describe pod <pod>` 和 `kubectl describe node <node>` |
| `CUDA driver version is insufficient for CUDA runtime version` | 主机驱动太旧，镜像 CUDA 太新 | `nvidia-smi` + `python -c "import torch; print(torch.version.cuda)"` |
| 旧镜像在 B200/GB200 上性能异常或算子不可用 | 镜像早于 Blackwell 支持窗口 | 查 NGC PyTorch release notes，再换 25.01+ 或 26.x 镜像重测 |
| V100 集群升级到新 NGC 镜像后异常 | 新镜像停止测试/支持 Volta | 回退到仍覆盖 Volta 的镜像或单独维护 V100 软件栈 |
| NCCL 卡住但 GPU 利用率为 0 | rank 没到齐、网络接口选错、RDMA 未暴露、`/dev/shm` 太小 | `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` |
| 多机带宽远低于预期 | NCCL 退回 Socket、PFC/MTU/RDMA 配置问题 | `all_reduce_perf` + `ibv_devinfo` + `NCCL_DEBUG=INFO` |
| Pod 请求 1 个 GPU 却拿到 MIG 资源名不一致 | MIG strategy 和资源名不匹配 | `kubectl describe node | grep nvidia.com/mig` |
| `pip install -U ...` 后原来能跑的容器坏了 | 覆盖了 NGC 镜像内预装依赖或触发 constraints | `pip check` + 对比 `/etc/pip/constraint.txt` |

## 案例 1：`nvcr.io/nvidia/pytorch:24.01-py3` 不是一个安全的长期默认值

### 典型现场

团队从老教程复制了：

```yaml
image: nvcr.io/nvidia/pytorch:24.01-py3
```

在 A100/H100 上还能跑，于是把它当成“标准 PyTorch 镜像”。几个月后换到 B200/GB200、RTX 50 系、CUDA 13 工具链，或者需要新的 Transformer Engine / TensorRT-LLM / Torch-TensorRT，就开始出现：

- `no kernel image is available for execution on the device`
- `CUDA error: operation not supported`
- `undefined symbol` / `GLIBCXX` / `libcuda.so` 相关动态库错误
- FlashAttention、Transformer Engine、TensorRT-LLM 编译或 import 失败
- `pip install -U` 后 PyTorch、Triton、Transformer Engine 版本互相打架

### 根因

NGC PyTorch 镜像是“整套深度学习栈快照”，不是只有 PyTorch。`24.01-py3` 对应的是 2024-01 的栈：Ubuntu 22.04、Python 3.10、CUDA 12.3.2、PyTorch 2.2.0a0、NCCL 2.19.x 和 TensorRT 8.6.x。到了 2026-06-01，官方 PyTorch release notes 最新索引已经到 `26.04-py3`，其中包含 CUDA 13.2.1、PyTorch 2.12.0a0、TensorRT 10.13.x、Torch-TensorRT 2.12.x、DCGM 4.4.x 等明显更新的组件。

更容易被忽略的是支持窗口变化：

- `25.01` 开始针对 NVIDIA Blackwell 架构优化，但官方同时说明 Volta 架构不再支持。
- `24.05` 后镜像不再包含 `torchtext` 和 `torchdata`，旧教程如果默认 import 它们会直接失败。
- `25.03` 起镜像内引入 pip constraints 文件，随手 `pip install -U` 可能被 constraints 限制，也可能破坏 NVIDIA 已验证过的组合。

### 怎么修

1. 不把 `24.01-py3` 当默认模板。示例可以旧，生产镜像必须重新按 GPU 架构、主机驱动、CUDA runtime、框架版本选。
2. 新 Blackwell 节点优先从 25.01+ 或当前 26.x NGC 镜像开始验证；V100/Volta 节点不要盲升到 25.01+。
3. 镜像 tag 固定到明确版本，发布环境再固定 digest。不要用 `latest`。
4. 派生镜像里不要无脑 `pip install -U torch triton flash-attn`。先看 `/etc/pip/constraint.txt`，再用 `pip check` 和 smoke test 验证。
5. 每次换镜像都跑最小 GPU/NCCL/框架 sanity test，而不是直接跑大训练。

```bash
docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.04-py3 \
  python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print((x @ x).float().mean().item())
PY
```

## 案例 2：主机驱动和容器 CUDA runtime 不匹配

### 典型现场

主机 `nvidia-smi` 看起来正常，但容器里 PyTorch 报：

```text
CUDA driver version is insufficient for CUDA runtime version
```

或者 `torch.cuda.is_available()` 是 `False`。

### 根因

`nvidia-smi` 里显示的 `CUDA Version` 是驱动支持的最高 CUDA API 口径，不等于容器里实际安装的 CUDA runtime。容器可以带 CUDA 12.x/13.x runtime，但真正执行 GPU 代码仍依赖主机内核驱动。镜像越新，对主机驱动分支要求越高。

### 怎么修

- 只在主机装/升级 NVIDIA driver，不要在应用容器里装驱动。
- 如果不能升级主机驱动，就选更旧的 NGC 镜像或 CUDA runtime。
- 同时记录 `nvidia-smi`、`torch.version.cuda`、`torch.cuda.get_device_capability()`，不要只看一个数字。
- 多节点训练时要求所有节点驱动分支一致；“有几台能跑”不代表集群能稳定跑。

## 案例 3：GPU Operator 装好了，但 Pod 里还是没有 GPU

### 典型现场

节点上 `nvidia-smi` 正常，`kubectl describe node` 里没有 `nvidia.com/gpu`，或者 Pod 事件里出现 runtime / CDI / device plugin 相关错误。

### 根因

Kubernetes 不是直接调用 `nvidia-smi` 分配 GPU。它需要：

```text
Host driver -> NVIDIA Container Toolkit -> container runtime -> device plugin -> kubelet extended resource
```

任意一层错配，Pod 都可能看不到 GPU。GPU Operator 26.x 这一代还要特别注意 CDI 模式：官方文档说明 25.10 起 CDI 默认启用；如果工作负载依赖老的 `NVIDIA_VISIBLE_DEVICES` 行为，或者没有为需要 GPU 的管理容器配置 `runtimeClassName: nvidia`，迁移时容易出现“host 有 GPU、Pod 无 GPU”。

### 怎么修

```bash
# 1. 节点是否暴露 GPU extended resource
kubectl describe node <node> | grep -A8 -E "Capacity|Allocatable|nvidia.com"

# 2. device plugin 是否健康
kubectl -n gpu-operator get pods -o wide | grep -E "device-plugin|toolkit|driver|validator"
kubectl -n gpu-operator logs ds/nvidia-device-plugin-daemonset --tail=200

# 3. Pod 是否真的请求 GPU
kubectl get pod <pod> -o yaml | grep -A5 "nvidia.com/gpu"
```

K8s GPU 资源属于 extended resource。实践中请只在 `limits` 里写 `nvidia.com/gpu`；如果同时写 `requests`，它必须和 `limits` 相等。不要期待 GPU 像 CPU 一样超卖。

## 案例 4：MIG、time-slicing 和整卡资源名混用

### 典型现场

集群里有的节点暴露：

```text
nvidia.com/gpu: 8
```

有的节点暴露：

```text
nvidia.com/mig-1g.10gb: 56
nvidia.com/mig-3g.40gb: 8
```

作业 YAML 仍然写 `nvidia.com/gpu: 1`，结果调度不到节点，或者调度到了共享 GPU 后出现邻居进程干扰、显存 OOM、延迟尖刺。

### 根因

MIG 是硬隔离切分，资源名会变成 `nvidia.com/mig-*`；time-slicing 是共享同一张 GPU 的时间片，适合开发/轻量推理，但官方文档明确提示它没有显存隔离、没有故障隔离，多个客户端看到的 GPU memory 也不是按份额切开的。

### 怎么修

- 训练任务优先请求整卡，MIG 更适合开发、小模型推理和隔离实验。
- 如果开 MIG，按 profile 写资源名，例如 `nvidia.com/mig-3g.40gb: 1`。
- 如果开 time-slicing，给共享资源改名或打标签，避免用户以为拿到的是独占整卡。
- 关键服务不要把“共享 GPU 数量变多”当成容量提升，必须用真实 QPS/延迟压测验证。

## 案例 5：NCCL 卡住，实际是 `/dev/shm`、memlock 或 NUMA

### 典型现场

单机 8 卡训练偶尔挂住，NCCL 日志停在 init 或某个 collective。Docker/K8s 默认 `/dev/shm` 很小，或者容器缺少 memlock/NUMA 相关能力。

### 根因

NCCL 不只用 GPU 显存。它还会用共享内存、pinned host memory、socket/RDMA buffer。NCCL 文档特别提醒 Docker 默认共享内存和 memlock 限制会导致问题；较新的 NCCL 还可能使用 cuMem host allocations，这依赖 CUDA driver/runtime 和 NUMA 能力。容器环境里 NUMA 被限制时，反而会在 NCCL 初始化阶段暴露问题。

### 怎么修

Docker：

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ...
```

K8s：

```yaml
spec:
  containers:
  - name: trainer
    resources:
      limits:
        nvidia.com/gpu: 8
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: "32Gi"
```

如果确认是 cuMem host allocations 和容器 NUMA 的组合问题，可临时验证：

```bash
export NCCL_CUMEM_HOST_ENABLE=0
```

这应该作为定位手段，不是长期调优口诀。长期方案是修正容器权限、NUMA 能力和 NCCL/CUDA 版本组合。

## 案例 6：你以为走了 RDMA，NCCL 实际退回 Socket

### 典型现场

多机训练可以跑，但带宽只有预期的 10%-30%，step time 抖动大。NCCL 日志里出现 `NET/Socket`，没有走 IB/RDMA。

### 根因

K8s overlay 网络、CNI 策略、RDMA device plugin、`nvidia-peermem`、网卡设备挂载、MTU/PFC/ECN 任一环节有问题，NCCL 都可能退回 socket 或选错网卡。应用层只看到“训练慢”，不一定直接报错。

### 怎么修

```bash
# Pod 内
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET torchrun ...
ibv_devinfo
ls /dev/infiniband

# 节点上
lsmod | grep -E "nvidia_peermem|mlx5"
ibstat
perfquery
```

验证顺序：

1. 单节点 `nvidia-smi topo -m` 和 P2P bandwidth 正常。
2. 跨节点 `ib_write_bw` / `ib_read_bw` 接近链路预期。
3. `nccl-tests all_reduce_perf` 接近同硬件同拓扑的历史基线。
4. 最后才跑完整训练。

## 案例 7：拓扑不感知导致张量并行性能崩

### 典型现场

Pod 请求 4 张 GPU，K8s 给了同一节点上的 4 张卡，但 TP 性能比裸机/Slurm 差很多。

### 根因

默认 Kubernetes 只知道“这台机器还有几张 GPU”，不知道 GPU0-GPU3 是否在同一 NVLink/NVSwitch domain，也不知道 GPU 和 NIC/CPU NUMA 的距离。TP/PP/EP 对 GPU-GPU 和 GPU-NIC 拓扑很敏感，拿到“数量正确但位置很差”的卡会直接掉性能。

### 怎么修

- 用 `nvidia-smi topo -m` 建立节点拓扑档案。
- 用 GPU Feature Discovery / Node Feature Discovery 给节点打 GPU 型号、MIG、拓扑相关标签。
- 关键训练任务通过 nodeAffinity、专用 node pool 或调度器扩展约束拓扑。
- 在 K8s 上开启 CPU Manager / Topology Manager 时，要和 GPU/NIC 的 NUMA 绑定一起验证，不要只看 Pod 成功启动。

## 案例 8：包管理把 NGC 镜像里的验证组合破坏了

### 典型现场

基础镜像能跑，派生镜像里只是多装几个 Python 包，结果 `import transformer_engine`、`import tensorrt_llm` 或 FlashAttention 编译失败。

### 根因

NGC 镜像内的 PyTorch、Triton、CUDA extension、Transformer Engine、TensorRT、NCCL 是一起验证的。`pip install -U` 可能升级其中一个包，留下 ABI 不匹配的另一个包。25.03 起官方还把 constraints 文件放在 `/etc/pip/constraint.txt`，这让依赖解析更可控，但也会让“照网上教程安装最新版”变成不可预测行为。

### 怎么修

```Dockerfile
FROM nvcr.io/nvidia/pytorch:26.04-py3

# 先显式保留或理解 NVIDIA constraints，再安装业务依赖
RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt \
 && python -m pip check
```

派生镜像构建后至少跑：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
for name in ["transformer_engine", "tensorrt", "torch_tensorrt"]:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "ok"))
    except Exception as exc:
        print(name, "FAILED", repr(exc))
PY
pip check
```

## 案例 9：GPU Operator 驱动容器和节点 OS/内核不一致

### 典型现场

GPU Operator 装上后 driver pod 或 validator pod CrashLoop。某些节点成功，某些节点失败。

### 根因

GPU Operator 可以帮你装驱动，但它不是魔法。驱动容器、宿主机内核、内核 headers、Secure Boot、OS 版本都要匹配。官方文档也提示：如果通过 GPU Operator 管理驱动，所有 GPU worker 节点应使用相同 OS 版本。

### 怎么修

- GPU node pool 尽量使用同一 OS image、同一 kernel、同一驱动策略。
- 云上优先用官方 GPU node image 或已验证的 golden image。
- 装 Operator 前先确认 Secure Boot、kernel headers、container runtime 配置。
- 对多 OS/多内核的历史集群，先分 node pool 灰度，不要一次全量升级。

## 实战升级顺序

1. 先在一台非生产节点验证主机驱动 + container runtime + `nvidia-smi`。
2. 再装/升级 GPU Operator、device plugin、DCGM exporter，确认 `nvidia.com/gpu` 出现在 allocatable。
3. 再跑 NGC PyTorch 镜像 sanity test。
4. 再跑 `nccl-tests` 单机和跨机基线。
5. 再跑一个最小 `torchrun` 或 vLLM smoke test。
6. 最后才迁移真实训练/推理 workload。

## 延伸阅读

- [NVIDIA PyTorch Container Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)
- [NVIDIA GPU Operator Release Notes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/release-notes.html)
- [NVIDIA GPU Operator CDI Support](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/cdi.html)
- [NVIDIA GPU Sharing on Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)
- [NVIDIA k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

