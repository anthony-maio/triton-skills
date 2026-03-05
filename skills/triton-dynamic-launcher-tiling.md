---
name: triton-dynamic-launcher-tiling
description: Build Triton kernel launchers that pick tile sizes, warps, and stages at runtime.
---

# Dynamic Tile & Pipeline Launcher for Triton

> **Targets:** Triton >= 2.1, SM70+/CDNA2+; shared memory heuristics tuned for A100/H100

Overview
For real-time inference you often cannot afford autotune warmup. Write a lightweight launcher that selects BLOCK sizes, num_warps, and num_stages heuristically from input shapes, dtype and device limits. The launcher emits constexpr kernel params so the kernel is optimized without runtime branching.

Key principles / step-by-step
- Base decisions on sequence lengths: choose BLOCK_M (query tile) and BLOCK_N (KV tile) by Lq/Lk ranges (small/medium/large). Cap tiles when HEAD_DIM is large to limit register pressure.
- Be dtype-aware: FP32 consumes twice the bytes of FP16/BF16. Reduce tile sizes when dtype_bytes >= 4 and estimate shared-memory usage.
- Compute tile_bytes = (BLOCK_M + BLOCK_N) * HEAD_DIM * dtype_bytes * 2 to estimate shared-memory footprint. Use hardware SM shared memory budgets to decide num_stages.
- Pipeline stages: fewer stages if tile_bytes large (no buffering), more stages when tiles are tiny for latency hiding. Simple rule: num_stages = 1 if tile_bytes > 64KB else 2 (or 3/4 for <16KB).
- Choose num_warps by tile size: default 4, increase to 8 for very large tiles, decrease to 2 for tiny/decode paths.
- Grid mapping: attention uses 2D grid (cdiv(Q_LEN,BLOCK_M), B*H); matmul can use 1D grid over tiles. For GQA split mapping pass H and H_KV and compute off_h_kv in-kernel.
- Handle optional inputs by passing dummy empty tensors and using tl.constexpr flags to omit loads.

Examples

1) Launcher sketch
```python
def launch_attn_kernel(q, k, v, rms_weight=None):
    B, H, Q_LEN, HEAD_DIM = q.shape
    _, H_KV, K_LEN, _ = k.shape
    dtype_bytes = q.element_size()

    BLOCK_M = 128 if Q_LEN > 64 else (64 if Q_LEN > 16 else 16)
    BLOCK_N = 128 if K_LEN > 256 else (64 if K_LEN > 64 else 32)
    if HEAD_DIM > 128:
        BLOCK_M = min(BLOCK_M, 64); BLOCK_N = min(BLOCK_N, 64)
    if dtype_bytes >= 4:
        BLOCK_M = min(BLOCK_M, 32)

    tile_bytes = (BLOCK_M + BLOCK_N) * HEAD_DIM * dtype_bytes * 2
    num_stages = 1 if tile_bytes > 64*1024 else (3 if tile_bytes < 16*1024 else 2)
    num_warps = 8 if BLOCK_M >=128 and BLOCK_N >=128 else (2 if BLOCK_M<=16 else 4)

    rms_weight = torch.empty(0, device=q.device, dtype=q.dtype) if rms_weight is None else rms_weight
    grid = (triton.cdiv(Q_LEN, BLOCK_M), B * H)
    kernel[grid](q, k, v, rms_weight, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                 num_stages=num_stages, num_warps=num_warps, APPLY_RMS=bool(rms_weight.numel()))
```

2) GQA head mapping note
```python
# in-kernel: off_h_kv = off_h // (H // H_KV)  # compute which kv-head maps to this head
```

Best practices / pitfalls
- Prefer conservative tile sizes to avoid register/shared memory spills which reduce occupancy.  
- Keep dtype_bytes checks early; FP32 needs halved tiles in practice.  
- Use simple heuristics rather than expensive tuning loops for real-time paths.  
- Validate on target hardware (A100/H100 shared memory differs).  
- Use tl.constexpr flags and pass dummy tensors for optional features to eliminate dead code.