---
name: triton-gpu-kernel-optimization
description: Teach an AI agent how to write high-performance Triton GPU kernels for DL ops.
---

# Write optimized Triton GPU kernels for deep learning operations

> **Targets:** Triton >= 2.1, SM70+/CDNA2+

Overview
This guide gives actionable patterns and examples for Triton kernels: block tiling, autotune, grouped/L2-aware tile ordering, fused kernels (softmax/attention), seed-based Philox dropout, mixed precision with FP32 accumulation, SRAM residency for intermediates, and benchmarking with triton.testing.Benchmark. Always use tl.constexpr power-of-two BLOCK sizes and mask OOB accesses.

Key principles / step-by-step
- Use @triton.jit and tl.program_id() to assign block work. Compute offsets with tl.arange() and build mask = offs < dim for safe loads/stores.  
- Expose BLOCK sizes as tl.constexpr and use @triton.autotune to sweep BLOCK_SIZE_M/N/K.  
- Keep intermediates in on-chip SRAM; perform reductions inside block (tl.sum, tl.max) before global writes.  
- Use tl.dot() for tensor-core matmul and tl.float32 accumulators when inputs are FP16.  
- For L2 locality, use grouped tile ordering via group_id/num_pid when computing grid.  
- Benchmark with triton.testing.Benchmark; build grid via lambda meta functions.

Practical examples

Tiled GEMM sketch (autotune + grouped ordering):
```python
@triton.autotune(
  configs=[triton.Config({'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':32}, key=['BLOCK_SIZE_M','BLOCK_SIZE_N','BLOCK_SIZE_K'])],
  key=['M','N','K']
)
@triton.jit
def gemm_fp16_fp32(A, B, C, M, N, K,
                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    group = pid % 4
    m_block = (pid // tl.cdiv(N, BLOCK_SIZE_N)) * BLOCK_SIZE_M
    n_block = (pid % tl.cdiv(N, BLOCK_SIZE_N)) * BLOCK_SIZE_N
    offs_m = m_block + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_block + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_off in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A + offs_m[:, None] * K + (k_off + tl.arange(0, BLOCK_SIZE_K))[None, :], mask=(offs_m[:, None] < M) & ((k_off + tl.arange(0, BLOCK_SIZE_K))[None, :] < K), other=0.0).to(tl.float16)
        b = tl.load(B + (k_off + tl.arange(0, BLOCK_SIZE_K))[:, None] * N + offs_n[None, :], mask=((k_off + tl.arange(0, BLOCK_SIZE_K))[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float16)
        acc += tl.dot(a, b).to(tl.float32)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

Fused row-wise softmax (numerically stable, SRAM intermediates):
```python
@triton.jit
def fused_softmax_rowwise(x_ptr, out_ptr, rows, cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    col_idx = col_offs + row_id * cols  # base pointer per row
    mask = col_offs < cols
    x = tl.load(x_ptr + col_idx, mask=mask, other=-1e9)
    m = tl.max(x, axis=0)
    ex = tl.exp(x - m)
    s = tl.sum(ex, axis=0)
    out = ex / s
    tl.store(out_ptr + col_idx, out, mask=mask)
```
This uses tl.arange, tl.exp, tl.sum, mask, @triton.jit and keeps ex and accumulators in SRAM for each block.

Benchmark fused add+ReLU vs PyTorch (grid via lambda):
```python
@triton.jit
def fused_add_relu(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    out = tl.maximum(x + y, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)

def run_benchmark():
    import torch, triton, triton.testing
    x_size = 1 << 20
    bm = triton.testing.Benchmark(
        x_names=['n'],
        x_vals=[x_size],
        line_count=1
    )
    grid = lambda meta: (triton.cdiv(x_size, meta['BLOCK']),)
    # use bm.run(...) to compare fused_add_relu against torch add+relu across BLOCK choices
```

Best practices / common pitfalls
- Always mask: mask = offs < dim. Missing mask corrupts memory.  
- BLOCK sizes should be tl.constexpr and strongly prefer powers of two (required for tl.arange; non-power-of-two works in some cases but may reduce performance).  
- Use tl.dot() + tl.float32 accumulators for FP16 inputs to avoid precision loss.  
- Recompute PRNG masks (Philox) in backward to avoid storing large boolean masks.  
- Profile and autotune per-GPU using triton.testing.Benchmark and @triton.autotune.