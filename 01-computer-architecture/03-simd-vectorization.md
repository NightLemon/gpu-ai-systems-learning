# SIMD 向量化

> CPU 的并行计算能力——理解 SIMD 有助于对比 GPU 的 SIMT 模型。

## 核心概念

### SIMD vs SIMT

```
SIMD (CPU - AVX-512):
  一条指令处理 512/32 = 16 个 float
  程序员/编译器显式使用向量指令
  向量宽度固定

SIMT (GPU - CUDA):
  32 个线程 (warp) 执行同一指令
  程序员写标量代码，硬件自动并行
  更灵活（支持 divergence，虽然有代价）
```

### CPU 向量化示例

```c
#include <immintrin.h>  // AVX-512

void vec_add_avx512(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i += 16) {  // 每次处理 16 个 float
        __m512 va = _mm512_load_ps(a + i);
        __m512 vb = _mm512_load_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_store_ps(c + i, vc);
    }
}
```

### 自动向量化

```bash
# GCC 自动向量化
gcc -O3 -march=native -ftree-vectorize -fopt-info-vec my_code.c
# -fopt-info-vec 会打印哪些循环被向量化了
```

编译器能自动向量化的条件：无数据依赖、对齐的数据、简单循环。

### 与 GPU 的联系

理解 SIMD 帮助你理解为什么 GPU 更适合大规模并行：
- CPU SIMD 宽度固定（16 float with AVX-512），GPU SIMT 可以调度数千 warp
- CPU 需要手动或编译器向量化，GPU 的 SIMT 模型更自然
- CPU 适合少量数据的低延迟处理，GPU 适合大量数据的高吞吐处理

## 延伸阅读

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
