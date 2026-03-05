---
name: triton-flash-attention-v2
description: Teach an AI agent how to implement FlashAttention v2 kernels in Triton for memory-efficient transformer attention.
---

# FlashAttention v2 kernels in Triton

> **Targets:** Triton >= 2.1, SM70+/CDNA2+

Overview
This guide explains how to implement memory-efficient fused attention (FlashAttention v2) in Triton. The kernel computes O = softmax(QK^T / sqrt(d_k)) V without materializing the N×N attention matrix by iterating over K/V blocks, maintaining running softmax statistics (m, l), and recomputing weights in the backward pass.

Key principles / step-by-step
1. Grid: launch a Triton kernel grid over (batch, heads, query_block_idx). Each kernel handles a BLOCK_M×d_k slice of Q and iterates over K/V blocks of width BLOCK_N.
2. Data types: accept FP16 inputs; use FP32 accumulators for dot products and softmax math for stability.
3. Inner loop (per K/V block):
   - Load Q_block (BLOCK_M×d_k), K_block (BLOCK_N×d_k), V_block (BLOCK_N×d_v).
   - Compute S = tl.dot(Q_block, tl.trans(K_block)) * sm_scale (use fp32 accumulators). Note: use `tl.trans()`, not the deprecated `trans_b` kwarg.
   - If causal and this K_block lies after current Q positions, mask upper triangle in S for partial blocks.
   - Apply per-row max: m_block = tl.max(S, axis=1); compute exp(S - m_block).
   - Update running m_prev, l_prev (where l is sum of exp); when m_new > m_prev, rescale accumulator: acc *= tl.exp(m_prev - m_new).
   - Update acc = acc + exp(S - m_new) @ V_block (use tl.dot for attn@V).
   - Update l_last and store m and logsumexp per query-block for backward: logsumexp = m + log(l).
4. After all blocks, write O_block (cast back to FP16) and store logsumexp per query row.

Backward (recompute-attention):
- Recompute S blocks using saved Q/K blocks and logsumexp to reconstruct normalized weights per-block:
  - attn_block = tl.exp(S - (logsumexp_row))  (use fp32)
  - dV contribution: dO accumulates via tl.dot(attn_block.T, dO_block) etc.
- This avoids storing full attention matrix.

Code examples (pseudocode — adapt offsets/strides for your layout)
```python
@triton.jit
def flash_fwd(Q_ptr, K_ptr, V_ptr, O_ptr, logsumexp_ptr, ...,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # ... load q tile (BLOCK_M, d_k), set up offsets ...
    acc = tl.zeros((BLOCK_M, d_v), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e9, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, N, BLOCK_N):
        k = tl.load(...)   # (BLOCK_N, d_k)
        v = tl.load(...)   # (BLOCK_N, d_v)
        s = tl.dot(q, tl.trans(k)) * sm_scale  # (BLOCK_M, BLOCK_N)
        # apply triangular mask when causal
        m_new = tl.maximum(m, tl.max(s, axis=1))      # unconditional update
        alpha = tl.exp(m - m_new)                      # rescale factor
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    tl.store(O_ptr + ..., (acc / l[:, None]).to(OUT_DTYPE))
    tl.store(logsumexp_ptr + ..., m + tl.log(l))
```

Best practices & pitfalls
- Use FP32 accumulators and compute logsumexp to avoid instability.
- Autotune BLOCK_M/BLOCK_N; choose BLOCK_N that fits L2 for K/V.
- Implement block-level triangular masking for causal attention carefully for partial blocks.
- Rescale accumulators when m changes: acc *= exp(m_prev - m_new).
- Backward must recompute per-block S with identical tiling and masks; ensure determinism.
- Avoid excessive register pressure—favor moderate BLOCK sizes.