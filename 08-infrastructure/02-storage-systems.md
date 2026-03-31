# 存储系统

> 训练数据和 Checkpoint 是两类完全不同的 IO 负载，需要不同的存储设计。

## 核心概念

### 两类 IO 负载

```
训练数据读取:
  特点: 大量小文件或流式读取，顺序为主，吞吐优先
  要求: 持续的高吞吐 (数 GB/s ~ 数十 GB/s)
  频率: 每个 step 都要读取
  方向: 只读

Checkpoint 保存/恢复:
  特点: 少量超大文件，突发写入
  要求: 写入速度要快（否则 GPU 空等），恢复也要快
  大小: 与参数精度、优化器状态和并行策略有关
        仅权重文件和"可恢复训练状态"的大小往往不是一回事
  频率: 每 1000-5000 步保存一次
  方向: 写为主，恢复时读

→ 两类负载的最优存储方案不同
```

### 分布式文件系统

| 系统 | 特点 | 适合 | 不适合 |
|------|------|------|--------|
| **Lustre** | 高吞吐并行文件系统，HPC 标配 | 训练数据 + checkpoint | 海量小文件 |
| **GPFS (Spectrum Scale)** | IBM 企业级，元数据性能好 | 混合 IO 负载 | 成本敏感场景 |
| **BeeGFS** | 开源，部署简单 | 中小规模集群 | 超大规模 |
| **对象存储 (S3/GCS/MinIO)** | 无限容量，便宜 | 数据集长期存储 | 低延迟随机读 |
| **NVMe over Fabrics** | 网络 SSD，低延迟 | Checkpoint 快速保存 | 大容量存储 |
| **本地 NVMe SSD** | 最低延迟 | 数据缓存层、checkpoint 中转 | 跨节点共享 |

### 训练数据路径设计

```
典型的分层存储架构:

永久存储 (对象存储 / NAS):
  s3://training-data/
  → TB-PB 级，便宜，访问慢

  ↓ 预处理 + 分发

共享文件系统 (Lustre / GPFS):
  /shared/preprocessed/
  → 已 tokenize 的数据，mmap 友好的格式
  → 所有节点都能读取

  ↓ 本地缓存 (可选)

本地 NVMe SSD:
  /local_nvme/cache/
  → 热数据缓存，避免共享文件系统成为瓶颈
  → 特别适合 DataLoader workers 读取
```

**数据格式选择**：
```
不推荐: 原始文本文件 → 每 step 都要 tokenize → CPU 瓶颈
推荐: 预 tokenize 为二进制格式:
  - Memory-mapped 文件 (numpy memmap) → 随机访问快
  - WebDataset (.tar shards) → 流式读，适合对象存储
  - Mosaic StreamingDataset → 自动缓存 + 分布式 shuffle
  - Arrow / Parquet → 列存储，适合多模态
```

## Checkpoint 设计

### Checkpoint 类型对比

```
Full Checkpoint (所有 rank 产出完整 checkpoint):
  ✓ 恢复简单：任意 rank 数都能恢复
  ✗ 保存慢：每个 rank 都保存全量 → IO 放大
  ✗ 大：175B 模型 = ~2 TB × N ranks 的写入

Sharded Checkpoint (每个 rank 只保存自己的 shard):
  ✓ 保存快：N 个 rank 并行写，每个只写 1/N
  ✓ 小：总写入量 = 1 份完整 checkpoint
  ✗ 恢复时必须用相同的并行配置（或做 reshard）

Consolidated Checkpoint (保存后合并为 full):
  ✓ 保存时并行（快）
  ✓ 最终产物可以用任意配置恢复
  ✗ 合并步骤需要时间和临时空间
```

### 怎么选 checkpoint 方案

| 场景 | 更适合的方案 | 原因 |
|------|-------------|------|
| 单机或小规模实验 | Full checkpoint | 实现最简单，恢复方便 |
| 长时间大规模训练 | Sharded checkpoint | 写入压力小，恢复主路径更现实 |
| 训练和发布都需要 | Sharded + 周期性 consolidated | 训练期追求速度，发布期需要通用格式 |
| 云上冷备份 | 本地/共享存储热 checkpoint + 对象存储归档 | 不让对象存储成为恢复主路径 |

### 异步 Checkpoint

```
同步 Checkpoint（默认）:
  Step N: [Train] → [Save Checkpoint] → [Train]
                     GPU 空闲等 IO！
  175B 模型写 2 TB，Lustre ~10 GB/s → 等 200 秒 → GPU 浪费 $$$

异步 Checkpoint:
  Step N:   [Train] → [Copy to CPU] → [Train continues]
                            ↓
            [CPU 后台线程写入存储]
  
  GPU 只需等待 GPU→CPU 拷贝或 staging 完成，然后继续训练
  后台 IO 不阻塞训练

PyTorch 2.0+ (torch.distributed.checkpoint):
  # 基础接口: sharded 保存（同步，但各 rank 只写自己的 shard → 并行 IO）
  save(state_dict, storage_writer=FileSystemWriter(path))
  
  # 异步保存需要额外封装:
  # 1. 先将 state_dict 拷贝到 CPU（快，几秒）
  # 2. 后台线程调用 save() 写入存储（慢，但不阻塞训练）
  # PyTorch 2.4+ 的 AsyncCheckpointer 或自行实现 CPU offload + 后台写入
  # ⚠️ 不是调 save() 就自动异步——默认是同步阻塞的
```

### Checkpoint 一致性问题

```
分布式训练保存 checkpoint 时，所有 rank 必须保存同一 step 的状态:

问题场景:
  Rank 0: 成功保存 step 1000
  Rank 1: 网络抖动，保存失败
  → checkpoint 不完整，恢复时会出错

解决方案:
  1. 两阶段提交:
     a. 所有 rank 写入临时目录
     b. 所有 rank barrier 确认写入成功
     c. 原子 rename: temp/ → checkpoint_1000/
     d. 任何 rank 失败 → 清理临时文件，重试

  2. 版本号 + 验证:
     - 每个 shard 写入时附带 step 号和 md5
     - 恢复前验证所有 shard 的 step 号一致且 md5 正确
     
  3. 保留多份 checkpoint:
     - 保留最近 2-3 个 checkpoint
     - 即使最新的损坏，还能从上一个恢复
```

### Checkpoint 恢复性能

```
恢复瓶颈分析:

175B 模型，64 节点:
  Full Checkpoint: 每节点读 2 TB → 如果都从同一个存储读
    → 存储带宽瓶颈: 64 × 20 GB/s (本地读) > 存储系统总带宽
    
  Sharded Checkpoint: 每节点只读自己的 shard (~30 GB)
    → 并行读，IO 压力均匀分布
    → 但要求恢复时并行配置相同

优化:
  1. Checkpoint 放在高带宽存储（Lustre / 本地 NVMe）
  2. 用 sharded 格式避免 IO 放大
  3. 预取: 训练的同时后台加载 checkpoint 到 CPU 内存
  4. 对象存储只做冷存归档，不做训练恢复的主路径
```

### 训练数据路径和 checkpoint 路径要分开设计

一个常见错误是把训练样本读取和 checkpoint 写入都压到同一条共享存储路径上。这样在 checkpoint 时刻，训练数据读取很容易被写入流量挤压，直接表现为 step time 毛刺。

更稳妥的设计是：

- 训练数据主路径以稳定读吞吐为目标
- checkpoint 主路径以突发写入和恢复速度为目标
- 两者即使共用底层集群，也最好在目录、QoS、缓存层或存储池上隔离

## 常见问题

**Q: 对象存储（S3）适合做训练数据的直接读取吗？**

A: 小规模实验可以（延迟高但能用），大规模训练不推荐。S3 的单连接吞吐 ~100 MB/s，首字节延迟 ~50-200ms。建议将数据预先拉到共享文件系统或本地 NVMe 缓存。StreamingDataset 等库有自动缓存机制可以缓解。

**Q: Checkpoint 多久保存一次？**

A: 取决于训练成本和可接受的损失量。通常每 1000-5000 步一次（~30分钟到数小时）。更频繁 → 恢复后损失的计算量更少，但 IO 开销更大。异步 checkpoint 可以在不影响训练速度的前提下提高保存频率。

**Q: 恢复后需要回溯 DataLoader 到断点位置吗？**

A: 严格来说需要，否则同一 epoch 内会重复或漏掉部分数据。实践中有两种做法：(1) 保存 DataLoader 的 state (offset/seed)。(2) 大规模训练用无限 streaming data，不严格按 epoch，此时只需保存 global step 和 random seed。

## 延伸阅读

- [Lustre Documentation](https://www.lustre.org/)
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
- [Mosaic StreamingDataset](https://github.com/mosaicml/streaming)
- [CheckFreq: Frequent, Fine-Grained DNN Checkpointing](https://www.usenix.org/conference/fast21/presentation/mohan) — 异步 checkpoint 的研究

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Checkpoint** | 训练过程中定期保存的模型状态快照（参数 + 优化器状态 + 训练进度），用于故障恢复 |
| **Sharded Checkpoint** | 每个 rank 只保存自己的那一份 shard，并行写入，速度快但恢复时需要相同的并行配置 |
| **Async Checkpoint** | 异步保存：先将状态拷到 CPU（快），然后后台线程写入存储（慢但不阻塞训练） |
| **Lustre** | 高吞吐并行文件系统，HPC 集群的标配存储 |
| **GPFS (Spectrum Scale)** | IBM 的企业级并行文件系统 |
| **对象存储 (S3/GCS)** | 无限容量的云存储，延迟较高，适合数据集归档但不适合直接训练读取 |
| **NVMe** | Non-Volatile Memory Express，高速 SSD 的接口协议，本地 NVMe 常用作数据缓存层 |
