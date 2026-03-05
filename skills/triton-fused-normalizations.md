---
name: triton-fused-normalizations
description: Teach an AI agent to implement fused LayerNorm, RMSNorm, and GroupNorm kernels (forward & backward) in Triton.
---

# Fused Normalization Kernels in Triton (LayerNorm, RMSNorm, GroupNorm)

Overview
This guide teaches how to implement fused forward and backward normalization kernels in Triton: LayerNorm, RMSNorm, and GroupNorm. The focus is single-pass mean/variance accumulation, saving mean and rstd (1/std) for backward, two-stage reductions for weight gradients, and safe concurrent accumulation. Use fp32 accumulators even when inputs are fp16, and map one program per normalization "row" (e.g., per token or per group).

Key principles / step-by-step
1. Grid & tiling:
   - Launch one program per row (example: batch*seq*groups). Let BLOCK_F be feature chunk size. If F > BLOCK_F, loop across feature chunks and accumulate partial sums.
   - cdiv(F, BLOCK_F) chunks: compute running sum and sumsq in a single pass per row.
2. Single-pass mean/variance:
   - For each chunk: s += sum(x), ss += sum(x*x). After all chunks: mean = s / F; var = ss / F - mean^2; rstd = 1 / sqrt(var + eps).
   - Save mean and rstd to scratch buffers for backward.
3. Forward formulas:
   - LayerNorm: x_hat = (x - mean) * rstd; y = x_hat * gamma + beta
   - RMSNorm: x_hat = x * rstd_rms where rstd_rms = 1/sqrt(mean(x^2)+eps); y = x_hat * gamma
   - GroupNorm: treat each group as a row; apply LayerNorm per group
4. Backward (VJP) per-row:
   - Given dy and saved x_hat, mean(dy) and mean(dy * x_hat) computed over features:
     dx = (1/std) * (dy - mean(dy) - x_hat * mean(dy * x_hat)) * gamma
     dgamma = sum(dy * x_hat)
     dbeta = sum(dy)
   - Compute per-block partial sums for dgamma/dbeta; write partials in an intermediate buffer.
5. Two-stage reduction:
   - Kernel A: per-row kernels compute forward and backward partials; each block writes partial dgamma/dbeta per (head, channel) to a workspace.
   - Kernel B: small reduction kernel (or warp-level tree) sums partials across blocks and writes final dgamma/dbeta.
   - Alternatively, atomically accumulate into global dgamma/dbeta using tl.atomic_add (watch contention and performance).
6. Synchronization & precision:
   - Use tl.float32 for s, ss, accumulators and intermediate sums.
   - Save mean and rstd for reuse in backward to avoid recomputation.

Practical code examples
LayerNorm forward (simplified):
```python
@triton.jit
def layernorm_fwd(x_ptr, gamma_ptr, beta_ptr, mean_ptr, rstd_ptr, y_ptr,
                  B, F, BLOCK_F: tl.constexpr):
    row = tl.program_id(0)
    offs = row * F + tl.arange(0, BLOCK_F)
    # load chunk (with bounds)
    x = tl.load(x_ptr + offs, mask=offs < row*F + F, other=0.).to(tl.float32)
    s = tl.sum(x, 0)
    ss = tl.sum(x * x, 0)
    # across chunks: use loop and accumulate s, ss
    mean = s_total / F
    rstd = 1.0 / tl.sqrt(ss_total / F - mean*mean + eps)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    x_hat = (x - mean) * rstd
    y = x_hat * tl.load(gamma_ptr + offs).to(tl.float32) + tl.load(beta_ptr + offs).to(tl.float32)
    tl.store(y_ptr + offs, y.to(x.dtype))
```

Backward partials + dx computation (per row):
```python
@triton.jit
def layernorm_bwd_part(x_ptr, dy_ptr, gamma_ptr, mean_ptr, rstd_ptr, dx_ptr,
                       dgamma_p_ptr, dbeta_p_ptr, B, F, BLOCK_F: tl.constexpr):
    row = tl.program_id(0)
    offs = row * F + tl.arange(0, BLOCK_F)
    x = tl.load(x_ptr + offs).to(tl.float32)
    dy = tl.load(dy_ptr + offs).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x_hat = (x - mean) * rstd
    # partial reductions
    s_dy = tl.sum(dy, 0)
    s_dyx = tl.sum(dy * x_hat, 0)
    # write partials to workspace (one slot per block)
    tl.store(dgamma_p_ptr + row*P + block_id, tl.sum(dy * x_hat))
    tl.store(dbeta_p_ptr + row*P + block_id, tl.sum(dy))
    # compute dx using formula with per-row means later reduced across blocks (or compute exact per-row means here)
    dx = (rstd) * (dy - s_dy / F - x_hat * (s_dyx / F)) * tl.load(gamma_ptr + offs).to(tl.float32)
    tl.store(dx_ptr + offs, dx.to(x.dtype))
```

Final reduction kernel:
```python
@triton.jit
def reduce_partials(dgamma_p, dbeta_p, dgamma, dbeta, rows, parts):
    gid = tl.program_id(0)
    # sum over parts per row and write final dgamma/dbeta
```

Best practices & common pitfalls
- Always use fp32 accumulators for sums and reductions (fp16 leads to large errors).
- Ensure BLOCK_F covers many elements for good parallel reduction; loop when F >> BLOCK_F.
- Save mean and rstd per row (or per group) to avoid expensive recomputation in backward.
- For dgamma/dbeta, prefer two-stage reduction to avoid atomic contention; use atomics only if the workspace and contention are small.
- Handle boundary masks when loading feature tail elements.
- Fuse activation (e.g., GELU) in the same kernel to save memory bandwidth and kernel launches.
- Test numerical correctness vs reference (NumPy/PyTorch) with fp16 inputs and fp32 accumulators.

This skill provides the core patterns—adapt BLOCK_F, group partitioning, and reduction strategies to your hardware and model shapes for best performance.