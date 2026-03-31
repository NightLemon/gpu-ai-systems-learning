# GPU 集群网络

> 大规模训练中，网络是仅次于 GPU 的第二大瓶颈。理解网络拓扑和故障模式是工程落地的关键。

## 核心概念

### 网络技术栈

```
应用层:    NCCL / MPI
传输层:    RDMA (Remote Direct Memory Access)
链路层:    InfiniBand / RoCE (RDMA over Converged Ethernet)
物理层:    光纤 / 铜缆

RDMA 的核心价值: 绕过 CPU 和 OS 内核，直接在两端内存之间做低开销数据传输。
  → 对训练最重要的是更低的通信开销和更高的带宽利用率
  → 当源或目标内存是 GPU 显存时，通常还需要 GPUDirect RDMA 配合
  → 与 TCP 相比，RDMA 通常能显著降低延迟和 CPU 开销
```

### InfiniBand vs RoCE

| 方面 | InfiniBand | RoCE v2 |
|------|-----------|---------|
| 协议 | 专有（Subnet Manager 管理） | 基于以太网 + UDP |
| 延迟 | ~1 μs | ~2-5 μs |
| 拥塞控制 | 硬件信用机制（Credit-based） | ECN + PFC（需要精心调参） |
| 丢包容忍 | 几乎无丢包（链路级流控） | 依赖 PFC 防止丢包，配置不当会导致 NCCL 卡住 |
| 带宽 | HDR 200Gb, NDR 400Gb, XDR 800Gb | 100/200/400 GbE |
| 成本 | 高（专用交换机 + 光模块） | 中（复用以太网基础设施） |
| 运维 | 需要 IB 专业知识 | 网络团队更熟悉 |
| 适用 | 大规模 AI 训练（>256 GPU） | 中等规模 / 混合集群 |

**选择依据**：大规模训练集群通常优先选 InfiniBand，因为稳定性和运维确定性更高。已有成熟以太网基础设施的团队也可以用 RoCE，但要把 PFC、ECN、QoS 和故障演练一起纳入设计，而不是只看链路带宽。

### GPUDirect 技术栈

```
GPUDirect Storage:
  NVMe SSD → GPU 显存（绕过 CPU 内存）→ 加速 checkpoint 加载

GPUDirect RDMA:
  GPU0 显存 → 网卡 → 网络 → 网卡 → GPU1 显存
  → 数据路径尽量绕过 CPU / 系统内存
  → 是否真正走通取决于驱动、网卡、内核模块和 NCCL 拓扑探测结果

GPUDirect P2P (同一节点):
  GPU0 → NVLink → GPU1（不走 PCIe，延迟最低）
```

## 关键细节

### 网络拓扑与 Oversubscription

```
Fat-tree 拓扑（最常见）:

          ┌──── Core Switch ────┐
          │                      │
     ┌── Spine ──┐         ┌── Spine ──┐
     │           │         │           │
   Leaf        Leaf      Leaf        Leaf
    │           │         │           │
  ┌─┤─┐     ┌─┤─┐     ┌─┤─┐     ┌─┤─┐
  N0 N1 N2  N3 N4 N5  N6 N7 N8  N9 ...

三层带宽递减:
  机内 (NVLink):        900 GB/s (H100 NVSwitch)
  机架内 (Leaf Switch): ~200-400 Gbps × multi-rail
  跨机架 (Spine/Core):  取决于 oversubscription ratio

Oversubscription:
  1:1 = 全线速（理想但昂贵）
  2:1 = 跨机架带宽减半（常见的折中）
  3:1+ = 跨机架带宽严重不足（不适合训练，AllReduce 会卡）
  
⚠️ 对训练集群来说，oversubscription 越低越好，很多生产环境会把关键训练路径控制在 2:1 或更低。
  如果跨机架有 oversubscription → 更适合放 DP 这类更容易 overlap 的通信
  TP/PP 则应尽量避开这类链路，尤其是 TP
```

### 并行策略到网络拓扑的映射

这是系统工程师的核心能力——将逻辑并行维度映射到物理网络层级：

```
原则:
  通信量大 + 延迟敏感 → 放在高带宽低延迟互联上
  通信量小 + 可异步   → 放在低带宽高延迟互联上

典型映射:
  机内 NVLink (900 GB/s, <1μs):     Tensor Parallel（通常优先放这里）
  机架内 IB 全线速 (~200+ Gbps):    Pipeline Parallel (只传激活值)
  跨机架 IB (可能有 oversubscription): Data Parallel (可 overlap)
```

**具体示例：256 卡训练 70B 模型**

```
硬件: 32 台 8×H100 DGX，4 个机架每架 8 台，IB NDR 400Gb，2:1 跨机架 oversubscription

方案 A (推荐):
  TP=8 (机内 NVLink)
  × PP=4 (同机架内 4 台，Leaf 下无 oversubscription)
  × DP=8 (跨机架)
  
  分析:
  - TP 在 NVLink 上: 每层 4 次 AllReduce，带宽和延迟要求最高 ✓
  - PP 在机架内: P2P 传输激活值 ~50-200MB/step，机架内 IB 够用 ✓
  - DP 跨机架: AllReduce 梯度可以和 backward overlap，容忍 oversubscription ✓

方案 B (不推荐):
  TP=16 跨两台机器
  → TP 通信落到跨机网络上，带宽和时延都更差，且通信位于关键路径
  → 结果通常会明显差于机内 TP；是否“崩”取决于网络代际、rail 数量、隐藏维度和 overlap 能力 ✗
```

### 一个实用的放置检查表

把并行策略映射到物理网络时，可以先问四个问题：

1. **哪一类并行最怕延迟？**
  通常是 TP，其通信频繁且常在关键路径上，应优先留给机内 NVLink/NVSwitch。

2. **哪一类并行最吃总带宽？**
  大规模 DP、FSDP、EP 的集合通信或 all-to-all 往往最吃网络总带宽，应尽量放在全双工、低 oversubscription 的路径上。

3. **哪一类并行最容易 overlap？**
  DP 梯度同步通常更容易和 backward overlap，所以更适合放到相对慢一些但可预测的跨机路径上。

4. **哪一类通信最容易被长尾拖慢？**
  EP 的 all-to-all 和跨机 TP 都很怕个别节点或链路异常，放置时要尽量减少跨拥塞域的跳数。

### RoCE 环境的常见故障

```
故障 1: PFC Storm
  现象: NCCL 超时，所有 GPU 卡住
  原因: PFC 配置错误 → 拥塞时交换机端口全部暂停 → 死锁
  排查: ethtool -S <dev> | grep pause
  预防: 正确配置 PFC watchdog，限制 PFC 暂停时间

故障 2: ECN 阈值不当
  现象: 带宽远低于理论值（如 400Gbps 只跑出 100Gbps）
  原因: ECN 阈值太低 → 过早降速；或太高 → 拥塞丢包触发重传
  排查: mlnx_qos 查看 ECN 配置 / perftest 对比理论带宽

故障 3: DSCP/优先级映射不一致
  现象: RDMA 流量走了有损队列 → 随机丢包 → NCCL 间歇性卡顿
  排查: 确认交换机和网卡的 QoS 映射对齐

故障 4: MTU 不一致
  现象: 性能远低于预期
  原因: RDMA 用 Jumbo Frame 9000，路径上某设备 MTU < 9000 → 分片
  排查: ip link show / 交换机端口 MTU 配置

诊断工具:
  ibstat / ibstatus           # IB 链路状态
  show_gids                   # RoCE GID 表
  ib_write_bw / ib_read_bw    # RDMA 带宽测试（perftest 套件）
  nccl-tests (all_reduce_perf)# NCCL 集合通信实测（最终验证）
```

### NCCL 关键环境变量

```bash
# 设备选择
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3  # 使用哪些网卡
export NCCL_SOCKET_IFNAME=eth0                      # 指定 socket 通信使用的网口
export NCCL_NET_GDR_LEVEL=5                         # GPUDirect RDMA 级别

# 算法和协议
export NCCL_ALGO=Ring            # Ring / Tree（NCCL 会自动选，通常不需要手动指定）
export NCCL_PROTO=Simple         # Simple / LL / LL128

# 调试
export NCCL_DEBUG=INFO           # 输出连接信息和算法选择
export NCCL_DEBUG_SUBSYS=ALL     # 输出所有子系统日志

# 排障时常见的相关设置
# 具体 timeout 更常在训练框架或 torch.distributed 层配置，不建议把它当成万能修复手段
```

### 网络排障的先后顺序

不要一开始就改 NCCL 环境变量。更高效的顺序通常是：

1. 先验证链路是否健康：`ibstat`、`show_gids`、交换机端口错误计数。
2. 再验证 RDMA 带宽是否接近预期：`ib_write_bw`、`ib_read_bw`。
3. 然后验证 NCCL 集合通信：`nccl-tests`。
4. 最后才调整 NCCL 算法、协议、rail 绑定和框架参数。

否则很容易把网络故障误判成 NCCL 参数问题。

## 常见问题

**Q: 为什么不直接用 TCP/IP？**

A: TCP/IP 涉及内核协议栈、多次数据拷贝、中断处理，延迟高 (~50μs+)、CPU 占用大。RDMA 是零拷贝、内核旁路的，延迟低 (~1μs)、几乎不消耗 CPU。大规模训练中每步都有数百次通信，累积差距巨大。

**Q: 如何判断训练是不是网络瓶颈？**

A: 三步法：(1) 跑 `nccl-tests` 的 `all_reduce_perf` 单独测通信带宽，对比理论值。(2) 用 Nsight Systems 看时间线中 NCCL 操作占比。(3) 对比 8→64 卡时 MFU 的下降幅度——下降超过 15% 大概率是网络瓶颈。

**Q: Multi-rail（多网卡）是什么？**

A: 单张 IB NDR 网卡 400Gbps (~50 GB/s)，但 8×H100 的跨机通信需求远超此。DGX H100 配 8 张 400Gbps 网卡，每张 GPU 绑定一张。NCCL 自动利用多 rail 实现带宽叠加，支撑 AllReduce 等操作。

## 延伸阅读

- [NVIDIA DGX SuperPOD Architecture](https://docs.nvidia.com/dgx-superpod/)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [nccl-tests](https://github.com/NVIDIA/nccl-tests) — 通信性能测试工具
- [RoCE Configuration Best Practices](https://enterprise-support.nvidia.com/s/article/roce-configuration-for-linux)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **RDMA** | Remote Direct Memory Access，允许网卡绕过 CPU 和 OS 内核，直接读写远端内存，延迟极低 |
| **InfiniBand (IB)** | 专用高速网络协议，原生支持 RDMA，是大规模 AI 训练集群的首选 |
| **RoCE** | RDMA over Converged Ethernet，在标准以太网上实现 RDMA，成本较低但需要精心配置 |
| **PFC** | Priority Flow Control，以太网上的流控机制，防止丢包。配置不当会导致 PFC Storm |
| **ECN** | Explicit Congestion Notification，拥塞通知机制，让发送方感知网络拥塞并主动降速 |
| **GPUDirect RDMA** | GPU 显存直接通过网卡读写远端 GPU 显存，完全绕过 CPU |
| **Oversubscription** | 网络上行/下行带宽不对等的比例。2:1 表示下行带宽是上行的 2 倍，跨机架带宽可能不足 |
| **Fat-tree** | 常见的数据中心网络拓扑，由 Leaf、Spine、Core 三层交换机组成 |
| **Multi-rail** | 一台机器配备多张网卡，每张 GPU 绑定一张网卡，实现带宽叠加 |
