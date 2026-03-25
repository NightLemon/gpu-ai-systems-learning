# Triton 入门

> Triton 是 OpenAI 开发的 GPU 编程语言和编译器，允许你用 Python 写出接近手写 CUDA 性能的 GPU kernel。它自动处理 Shared Memory 管理、合并访存、Bank Conflict 等底层优化，大幅降低了 GPU 编程的门槛。

## 为什么需要 Triton？CUDA 不够用吗？

前面几节你已经看到，写一个高效的 CUDA kernel 需要关心很多底层细节：Shared Memory 分配、Bank Conflict 避免、合并访存、寄存器压力、Occupancy 调优……这些对于専业 GPU 工程师是日常，但对于想快速实现一个自定义算子的 AI 研究员来说是巨大的门槛。

Triton 的定位是：**用 Python 写 GPU kernel，让编译器而不是程序员来处理那些底层优化**。你只需要描述“这个 Block 要处理哪块数据、做什么计算”，Triton 编译器自动帮你安排 Shared Memory、确保合并访存、避免 Bank Conflict。

这意味着：写 30 行 Triton Python 能达到写 300 行 CUDA C++ 的 70-90% 性能。而且 PyTorch 2.0 的 `torch.compile()` 在幕后就是用 Triton 生成 kernel 的——所以理解 Triton 也能帮你理解 `torch.compile()` 的工作原理。

## 核心概念

### 什么是 Triton？

Triton 是 OpenAI 开发的 GPU 编程语言/编译器：

```
传统 CUDA 开发:
  手动管理: shared memory, 寄存器, coalescing, bank conflict, occupancy...
  → 写 1000 行 CUDA C++ 达到 cuBLAS 70% 性能

Triton:
  自动处理: shared memory tiling, coalescing, bank conflict...
  → 写 30 行 Python 达到 cuBLAS 70-90% 性能
```

**核心设计**：Triton 的编程单元不是 thread 而是 **block**。程序员只需要描述 block 级别的逻辑，Triton 编译器自动将其转换为高效的 GPU 代码（包括 shared memory 管理、向量化等）。

### Triton vs CUDA

| 方面 | CUDA | Triton |
|------|------|--------|
| 语言 | C/C++ | Python |
| 编程单元 | Thread | Block（以 2 的幂次大小的 tensor 操作） |
| 内存管理 | 手动（shared memory、register） | **自动** |
| Coalescing | 手动保证 | **自动** |
| Bank Conflict | 手动避免 | **自动** |
| 灵活性 | 最高 | 中等（不适合所有算子） |
| 学习曲线 | 陡 | 平缓（Python 程序员友好） |
| 典型性能 | 90-100% cuBLAS | 70-90% cuBLAS |

### 第一个 Triton Kernel: Vector Add

```python
import triton
import triton.language as tl
import torch

@triton.jit
def vec_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program（block）处理 BLOCK_SIZE 个元素
    pid = tl.program_id(0)  # 对应 CUDA 的 blockIdx.x
    
    # 计算当前 block 负责的元素范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # 边界检查
    
    # 加载数据（自动 coalesced）
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 计算
    output = x + y
    
    # 写回
    tl.store(output_ptr + offsets, output, mask=mask)

# 调用
def vec_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    vec_add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output

# 使用
x = torch.randn(1000000, device='cuda')
y = torch.randn(1000000, device='cuda')
z = vec_add(x, y)
```

## 关键细节

### Triton 的 Auto-tuning

Triton 内置强大的自动调优框架：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],  # 当这些参数变化时重新调优
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, 
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

### Triton GEMM 实现

```python
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 当前 block 负责 C 的哪个 tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算 offset
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # A 和 B 的起始指针
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Tile 循环
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N))
        
        acc += tl.dot(a, b)  # Triton 自动使用 Tensor Core!
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # 写回 C
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)
```

对比手写 CUDA 版本：**代码量减少了 5-10 倍**，且 Triton 编译器自动处理了 shared memory、bank conflict、向量化等。

### Triton 的核心操作

```python
# 加载/存储
data = tl.load(ptr + offsets, mask=mask, other=0.0)  # 自动 coalesced
tl.store(ptr + offsets, data, mask=mask)

# 数学运算
result = tl.dot(a, b)         # 矩阵乘（自动使用 Tensor Core）
result = tl.exp(x)            # 逐元素
result = tl.log(x)
result = tl.maximum(x, 0)     # ReLU

# 归约
total = tl.sum(x, axis=0)     # 沿轴求和
maximum = tl.max(x, axis=0)   # 沿轴求最大值

# Atomic
tl.atomic_add(ptr + offsets, values)

# Program ID
pid = tl.program_id(axis=0)   # 对应 blockIdx
num_programs = tl.num_programs(axis=0)  # 对应 gridDim
```

### Triton 在实际项目中的应用

| 项目 | 使用 Triton 的部分 |
|------|-------------------|
| **FlashAttention** | Triton 版本的 FlashAttention 实现 |
| **PyTorch 2.0 (torch.compile)** | TorchInductor 后端默认用 Triton 生成 kernel |
| **Unsloth** | 用 Triton 优化 LoRA 微调 |
| **vLLM** | PagedAttention 的 Triton kernel |
| **xFormers** | 部分 attention kernel |

## 常见问题

**Q: 什么场景用 Triton，什么场景必须写 CUDA？**

A: Triton 适合：
- 算子融合（fused kernel）
- Attention 变体
- 常见的逐元素 / reduction 操作
- 需要快速迭代的原型

仍需 CUDA 的场景：
- 极致优化（挤出最后 10-20% 性能）
- 复杂的 warp 级控制（如自定义通信模式）
- 非规则的内存访问模式
- 需要精确控制 register 使用

**Q: `torch.compile` 和 Triton 的关系？**

A: PyTorch 2.0 的 `torch.compile()` 使用 TorchInductor 后端，TorchInductor 会将 PyTorch 计算图自动编译为 Triton kernel。所以即使你不直接写 Triton，`torch.compile()` 也在幕后使用 Triton。理解 Triton 有助于调试 `torch.compile()` 生成的 kernel。

**Q: Triton 的性能通常能达到 cuBLAS 的多少？**

A: 对于 GEMM：70-90%（取决于矩阵大小和形状）。对于融合算子（如 fused softmax、FlashAttention），Triton 版本通常比未融合的 PyTorch 快几倍，因为减少了 HBM 读写。

## 延伸阅读

- [Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — 官方教程（必读）
- [Triton GitHub](https://github.com/triton-lang/triton)
- [PyTorch TorchInductor Deep Dive](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-target-backends/747)
- [Programming GPUs with Triton](https://www.youtube.com/watch?v=DdTsX6DQk24) — Philippe Tillet 的演讲

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Triton** | OpenAI 开发的 GPU 编程语言/编译器。用 Python 写 kernel，编译器自动优化为高效 GPU 代码 |
| **`@triton.jit`** | Triton 的装饰器，将一个 Python 函数编译为 GPU kernel |
| **Program** | Triton 中的执行单元，类似于 CUDA 的 Block。每个 program 处理一个数据块 |
| **`tl.load` / `tl.store`** | Triton 的数据加载/存储原语，自动处理合并访存 |
| **`tl.dot`** | Triton 的矩阵乘法原语，自动使用 Tensor Core |
| **Auto-tuning** | Triton 内置的自动调优框架，在多种配置（Block Size、num_warps 等）中搜索最优参数 |
| **TorchInductor** | PyTorch 2.0 `torch.compile()` 的后端编译器，默认使用 Triton 生成 GPU kernel |
| **算子融合（Kernel Fusion）** | 将多个小操作合并为一个 kernel，减少 kernel 启动开销和中间结果的显存读写 |
