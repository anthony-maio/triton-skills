---
name: triton-flash-attention-v2
description: Teach an AI agent how to implement FlashAttention v2 kernels in Triton for memory-efficient transformer attention.
---

# FlashAttention v2 kernels in Triton

Overview
This guide explains how to implement memory-efficient fused attention (FlashAttention v2) in Triton. The kernel computes O = softmax(QK^T / sqrt(d_k)) V without materializing the N×N attention matrix by iterating over K/V blocks, maintaining running softmax statistics (m, l), and recomputing weights in the backward pass.

Key principles / step-by-step
1. Grid: launch a Triton kernel grid over (batch, heads, query_block_idx). Each kernel handles a BLOCK_M×d_k slice of Q and iterates over K/V blocks of width BLOCK_N.
2. Data types: accept FP16 inputs; use FP32 accumulators for dot products and softmax math for stability.
3. Inner loop (per K/V block):
   - Load Q_block (BLOCK_M×d_k), K_block (BLOCK_N×d_k), V_block (BLOCK_N×d_v).
   - Compute S = tl.dot(Q_block, K_block.T) / sqrt(d_k) (use fp32 accumulators).
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

Code examples (conceptual snippets)
```python
@triton.autotune(
  configs=[{"BLOCK_M": 64, "BLOCK_N": 64}, {"BLOCK_M": 128, "BLOCK_N": 64}],
  key=["BLOCK_M","BLOCK_N"]
)
@triton.jit
def flash_fwd(Q_ptr, K_ptr, V_ptr, O_ptr, logsumexp_ptr, ...,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    q = tl.load(Q_ptr + program_id(2)*BLOCK_M*d + tl.arange(0, BLOCK_M)*d)  # simplified
    acc = tl.zeros((BLOCK_M, d_v), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e9, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, N, BLOCK_N):
        k = tl.load(...)
        s = tl.dot(q, k, trans_b=True) * (1.0 / sqrt_dk)
        # apply triangular mask when causal
        m_block = tl.max(s, axis=1)
        exp_s = tl.exp(s - m_block[:, None])
        # rescale when needed
        if m_block > m:
            acc *= tl.exp(m - m_block)
            l = l * tl.exp(m - m_block)
            m = m_block
        acc += tl.dot(exp_s, v_block)
        l += tl.sum(exp_s, axis=1)
    tl.store(O_ptr + ..., acc / l[:, None])
    tl.store(logsumexp_ptr + ..., m + tl.log(l))
```

Best practices & pitfalls
- Use FP32 accumulators and compute logsumexp to avoid instability.
- Autotune BLOCK_M/BLOCK_N; choose BLOCK_N that fits L2 for K/V.
- Implement block-level triangular masking for causal attention carefully for partial blocks.
- Rescale accumulators when m changes: acc *= exp(m_prev - m_new).
- Backward must recompute per-block S with identical tiling and masks; ensure determinism.
- Avoid excessive register pressure—favor moderate BLOCK sizes.