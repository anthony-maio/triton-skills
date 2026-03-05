---
name: triton-fused-epilogue-kernels
description: Teach how to fuse epilogue ops into Triton attention/matmul kernels to avoid HBM round-trips.
---

# Fused Epilogue Kernels in Triton

Overview  
Fusing epilogue work (normalization, bias, activation, dropout, gating, residual) directly into attention or GEMM kernels avoids extra HBM writes/reads and kernel launches. Use tl.constexpr flags to emit specialized variants at compile time and perform all final math in-register immediately before the final tl.store.

Key principles / step-by-step
- Use tl.constexpr bool flags (e.g., APPLY_RMS, APPLY_BIAS, APPLY_LEAKY_RELU, APPLY_DROPOUT, APPLY_GATE) so compilation yields branches-free kernels.
- Load small vectors (bias, norm weights, gate) once before the K-loop or at K-loop start — do not reload per element.
- Keep accumulators (acc, l, m etc.) in FP32 during accumulation; apply epilogue in FP32 then cast: out = out.to(OUT.dtype.element_ty).
- For online-softmax variants, finalize acc/l in-register, compute differential signal, variance, rstd, apply RMS weight, then store.
- For multi-stream (N parallel accumulators) share K/V loads, keep separate m/l/acc states, finalize each stream independently and combine in-register.
- Always add eps when computing rsqrt.

Examples

1) Fused attention + RMSNorm epilogue (sketch)
```python
@triton.jit
def attn_kernel(..., APPLY_RMS: tl.constexpr):
    # ... online-softmax K-loop producing acc_signal, l_signal, acc_noise, l_noise
    if APPLY_RMS:
        diff = acc_signal / l_signal - lam * (acc_noise / l_noise)   # [M,HEAD_DIM]
        var = tl.sum(diff * diff, axis=1) / HEAD_DIM
        rstd = tl.math.rsqrt(var + eps)                             # [M]
        rms_w = tl.load(rms_weight_ptr + head_offset)               # load HEAD_DIM once
        out = diff * rstd[:, None] * rms_w[None, :]
    tl.store(OUT_ptr + offs, out.to(OUT.dtype.element_ty))
```

2) Fused GEMM + bias + leaky-ReLU + dropout
```python
@triton.jit
def gemm_kernel(..., APPLY_BIAS: tl.constexpr, APPLY_LEAKY: tl.constexpr, APPLY_DROPOUT: tl.constexpr):
    # accumulate acc in FP32
    if APPLY_BIAS:
        b = tl.load(bias_ptr + col_offsets)
        acc = acc + b[None, :]
    if APPLY_LEAKY:
        acc = tl.where(acc > 0, acc, acc * 0.01)
    if APPLY_DROPOUT:
        acc = acc * tl.load(dropout_mask_ptr + offs)
    tl.store(C_ptr + offs, acc.to(C.dtype.element_ty))
```

3) Gating + residual fusion
```python
# after computing attn_out
if APPLY_GATE:
    g = tl.load(gate_ptr + row_idx)            # [M]
    res = tl.load(residual_ptr + offs)
    out = g[:, None] * attn_out + res
    tl.store(OUT_ptr + offs, out.to(OUT.dtype.element_ty))
```

Best practices / pitfalls
- Use constexpr flags to avoid runtime branching costs.  
- Keep small weight loads outside heavy K-loops.  
- Maintain FP32 accumulation to reduce precision loss; cast only at tl.store.  
- For RMSNorm: var = tl.sum(x*x, axis=1)/dim; rstd = tl.math.rsqrt(var + eps).  
- Test numerics when combining operations (dropout before/after scaling matters).  
- Beware of register pressure — complex multi-stream fusions can increase register usage and reduce occupancy; balance fusion vs. kernel resource limits.