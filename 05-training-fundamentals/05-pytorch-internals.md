# PyTorch 内部机制

> 理解 PyTorch 的执行模型、编译器栈和内存管理——知道框架"怎么跑的"，才能写出高效的训练代码并有效排障。本章按 PyTorch 2.x 的编译器栈讲解，具体 API 细节请以当前 stable 文档为准。

## 执行模型：Eager vs Graph

PyTorch 有两种执行模式，理解它们的区别是优化训练性能的起点。

### Eager Mode（即时执行）

这是 PyTorch 的默认模式。每一行 Python 代码对应的算子**立即执行**：

```python
# Eager mode: 每行立即执行
x = torch.randn(1024, 1024, device='cuda')
y = x @ x.T          # 立即执行 matmul
z = torch.relu(y)    # 立即执行 relu
```

优点是调试方便（随时 `print`、随时 `pdb`），缺点是每个算子独立 launch 一个 CUDA kernel，没有跨算子优化的机会。

### Graph Mode（图模式）

通过 `torch.compile` 捕获计算图，可以做跨算子融合、内存规划等优化：

```python
# Graph mode: 先捕获计算图，再一次性优化执行
@torch.compile
def fused_forward(x):
    y = x @ x.T
    z = torch.relu(y)
    return z
```

## torch.compile 编译器栈

`torch.compile` 是 PyTorch 2.0 引入的核心特性，它把 Python 代码变成优化过的 GPU kernel。

### 编译流程

```
Python 代码
    │
    ▼  TorchDynamo (Python bytecode 分析)
FX Graph (中间表示)
    │
    ▼  图优化 (算子融合、常量折叠等)
优化后的 FX Graph
    │
    ▼  TorchInductor (代码生成后端)
Triton kernel / C++ kernel
    │
    ▼  执行
GPU 运行
```

- **TorchDynamo**：分析 Python 字节码，捕获计算图。遇到无法跟踪的代码会产生 **graph break**（图断裂），把计算图拆成多段分别编译。
- **TorchInductor**：编译后端，将 FX Graph 翻译成 Triton kernel（GPU）或 C++ 代码（CPU）。Triton kernel 可以自动做算子融合，减少 kernel launch 开销和显存读写。

### 编译模式

```python
# 默认模式: 平衡编译时间和运行性能
model = torch.compile(model)

# reduce-overhead: 减少 kernel launch 开销 (用 CUDA Graphs)
model = torch.compile(model, mode="reduce-overhead")

# max-autotune: 尝试更多 kernel 配置，编译更慢但运行更快
model = torch.compile(model, mode="max-autotune")
```

| 模式 | 编译速度 | 运行加速 | 适用场景 |
|------|---------|---------|---------|
| `default` | 快 | 中等 | 日常开发、快速验证 |
| `reduce-overhead` | 中等 | 较高 | 训练（减少小 kernel 开销） |
| `max-autotune` | 慢 | 最高 | 推理（可以接受长编译时间） |

### 实际使用

```python
import torch

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 编译模型——只需要一行
compiled_model = torch.compile(model)

for batch in dataloader:
    # 第一次迭代会触发编译（较慢），之后走缓存
    loss = compiled_model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 常见限制

- **Graph break**：遇到动态控制流（`if tensor.item() > 0`）、不支持的 Python 特性时会断图，降低优化效果。用 `TORCH_LOGS="graph_breaks"` 查看断图原因。
- **动态 shape**：输入 shape 变化会触发重新编译。使用 `torch.compile(dynamic=True)` 可缓解，但可能损失部分性能。
- **首次编译开销**：首次调用需要编译，可能耗时几十秒到几分钟，之后通过缓存复用。

```bash
# 调试 torch.compile
TORCH_LOGS="graph_breaks" python train.py       # 查看 graph break
TORCH_LOGS="recompiles" python train.py          # 查看重编译
TORCHINDUCTOR_TRACE=1 python train.py            # 查看生成的 kernel 代码
```

## Autograd 机制

Autograd 是 PyTorch 自动求导的核心，理解它对调试梯度问题和写自定义算子非常重要。

### 计算图构建

前向传播时，PyTorch 自动构建计算图：

```python
x = torch.randn(3, requires_grad=True)
y = x * 2          # 记录 MulBackward
z = y.sum()         # 记录 SumBackward
print(z.grad_fn)    # <SumBackward0>
print(z.grad_fn.next_functions)  # 指向 MulBackward0
```

每个操作会创建一个 `grad_fn` 节点，形成一条 **反向链**。

### 反向传播

调用 `z.backward()` 时，PyTorch 按**反向拓扑序**遍历计算图，依次计算每个节点的梯度：

```
前向: x → MulBackward → y → SumBackward → z
反向: z → SumBackward → y → MulBackward → x.grad
```

### 为什么需要 zero_grad()

梯度默认是**累加**的，不会自动清零：

```python
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()         # 梯度累加到 .grad
    optimizer.step()
    optimizer.zero_grad()   # 不清零 → 梯度越来越大 → 训练爆炸
```

梯度累加的设计是为了支持梯度累积（gradient accumulation）：当 batch size 太大无法放入显存时，可以分多个 micro-batch 累加梯度再更新。

### 自定义 Autograd Function

当你需要自定义前向和反向逻辑（比如写一个高效的融合算子）：

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 使用
output = MyReLU.apply(input_tensor)
```

## Dispatcher：算子路由

Dispatcher 是 PyTorch 的核心调度机制，决定每个算子调用应该走哪个实现。

```
torch.add(a, b)
    │
    ▼  Dispatcher
    ├─ Autograd key  → 记录计算图（用于反向传播）
    ├─ CUDA key      → 调用 CUDA kernel
    ├─ CPU key       → 调用 CPU 实现
    ├─ FuncTorch key → 支持 vmap 等函数变换
    └─ ...
```

**为什么需要了解 Dispatcher**：

- 注册自定义算子时，需要为不同的 dispatch key 提供实现
- 理解为什么同一个 `torch.add` 在 CPU 和 CUDA 上的行为不同
- 混合设备训练（CPU offload）时的 tensor 路由

## 内存管理

### CUDA Caching Allocator

PyTorch 使用**缓存分配器**管理 GPU 显存：

```
PyTorch 向 CUDA 申请显存块（一次申请一大块）
    │
    ├─ 内部维护空闲块列表
    ├─ tensor 分配从空闲列表中取
    ├─ tensor 释放归还到空闲列表（不还给 CUDA）
    └─ torch.cuda.empty_cache() → 把空闲列表还给 CUDA
```

这意味着 `nvidia-smi` 显示的显存占用通常远大于实际使用量——因为 PyTorch 拿了不还。

### 为什么 empty_cache() 很少有用

```python
# 常见误解
torch.cuda.empty_cache()  # "释放显存"

# 实际上:
# 1. 只释放缓存中的空闲块，正在使用的 tensor 不受影响
# 2. 释放后 PyTorch 又会重新申请，反而增加 cudaMalloc 开销
# 3. 真正的 OOM 原因通常是峰值显存太高，而不是缓存占用
```

### 显存碎片化

OOM 时即使有空闲显存也可能分配失败——因为空闲显存不连续：

```
显存布局:  [占用][空闲 100MB][占用][空闲 200MB][占用][空闲 150MB]
请求 400MB → 失败！虽然总空闲 450MB，但最大连续块只有 200MB
```

### 诊断工具

```python
# 显存概览
print(torch.cuda.memory_summary())

# 详细统计
stats = torch.cuda.memory_stats()
print(f"当前分配: {stats['allocated_bytes.all.current'] / 1e9:.2f} GB")
print(f"峰值分配: {stats['allocated_bytes.all.peak'] / 1e9:.2f} GB")
print(f"缓存总量: {stats['reserved_bytes.all.current'] / 1e9:.2f} GB")

# 显存快照（定位显存泄漏）
torch.cuda.memory._record_memory_history()
# ... 运行一段训练代码 ...
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")
# 上传到 https://pytorch.org/memory_viz 或使用当前版本推荐的 memory viewer 可视化
```

## 交叉引用

- **CUDA kernel 编写与优化** → 参见 [Ch03 CUDA 编程](../03-cuda-programming/README.md)
- **分布式训练通信** → 参见 [Ch06 分布式训练](../06-distributed-training/README.md)
- **性能分析工具** → 参见 [Ch05 Profiling](04-profiling.md)

## 延伸阅读

- [PyTorch Internals (Edward Z. Yang)](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [torch.compile Documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [TorchDynamo Deep Dive](https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html)
- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [CUDA Caching Allocator](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

---

## 小结

| 概念 | 要点 |
|------|------|
| Eager vs Graph | Eager 逐行执行便于调试，Graph 通过 torch.compile 做跨算子优化 |
| torch.compile | TorchDynamo 捕获计算图 → TorchInductor 生成 Triton kernel，注意 graph break |
| Autograd | 前向时构建计算图（grad_fn 链），反向时按拓扑序求导；梯度默认累加 |
| Dispatcher | 根据 tensor 类型和设备路由到正确的算子实现（CUDA/CPU/Autograd key） |
| 显存管理 | Caching Allocator 一次申请大块显存，内部维护空闲列表；注意碎片化导致的 OOM |

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Eager Mode** | PyTorch 默认的即时执行模式，每个算子立即运行 |
| **Graph Mode** | 通过 torch.compile 捕获计算图后优化执行的模式 |
| **torch.compile** | PyTorch 2.0 的编译器入口，将 Python 代码编译为优化后的 kernel |
| **TorchDynamo** | Python 字节码分析框架，用于捕获计算图 |
| **TorchInductor** | torch.compile 的默认后端，将计算图编译为 Triton 或 C++ kernel |
| **Graph Break** | torch.compile 遇到无法跟踪的代码时，将计算图断开的行为 |
| **Autograd** | PyTorch 的自动微分引擎，在前向传播时构建计算图，反向传播时自动求导 |
| **grad_fn** | tensor 上记录的梯度函数，形成反向传播链 |
| **Dispatcher** | PyTorch 的算子调度器，根据 tensor 类型和设备选择正确的算子实现 |
| **CUDA Caching Allocator** | PyTorch 的 GPU 显存管理器，通过缓存机制减少 cudaMalloc 调用 |
| **显存碎片化** | 空闲显存被已分配的 tensor 分隔成不连续的小块，导致大块分配失败 |
