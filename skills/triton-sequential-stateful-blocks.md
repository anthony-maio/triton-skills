---
name: triton-sequential-stateful-blocks
description: Teach writing Triton kernels that perform sequential, stateful processing inside one block.
---

# Sequential Stateful Processing in a Single Triton Block

Overview  
Some workloads require one thread block to process a sequence of items with mutable register state (e.g., an LRU cache router). This pattern uses a grid like (B,) — one block per batch element — and updates registers in a sequential loop so each iteration sees the exact mutated state from previous iterations.

Key principles / step-by-step
- Use one block per sequence (grid=(B,)). Keep mutable state in registers / SRAM and never write intermediate state back to HBM until the end.
- Initialize state (used flags, timestamps, cached vectors) into register-backed tl.load values before the candidate loop.
- Iterate sequentially: for t in range(T): compute reductions, make scalar decisions, mutate registers immediately (e.g., update timestamp array in registers).
- Use scalar control flow (if/elif/else) on reduced scalars. Create scalar constants with tl.full([],...). Use tl.zeros([],...)/tl.full([],..., dtype=tl.int1) for booleans.
- For mixed outputs, use separate output pointers/strides for indices (int64), flags (int32/int1), and float scores.
- For multi-head comparisons, perform head-loop reductions inside the candidate loop, aggregate to a scalar decision, mask inactive slots, then argmax/argmin for routing.

Examples

1) LRU routing sketch (core loop)
```python
# per-block registers: used_r (int32[ME]), ts_r (int64[ME]), cache_r (ME, DH)
for t in range(T):
    cand = tl.load(cand_ptr + t * cand_stride)        # (DH,)
    avg_scores = tl.zeros((ME,), dtype=tl.float32)
    for h in range(H_KV):
        head_cache = cache_r[h]                       # (ME, DH)
        scores_h = tl.sum(head_cache * cand, axis=1)
        avg_scores += scores_h
    avg_scores = avg_scores / H_KV
    masked = tl.where((used_r != 0) & active_mask, avg_scores, tl.full((ME,), -1e9))
    best_score = tl.max(masked); best_idx = tl.argmax(masked)
    is_hit = best_score > threshold
    if is_hit:
        ts_r[best_idx] = current_time                   # immediate mutation
    else:
        victim = tl.argmin(ts_r)
        used_r[victim] = tl.full([], 1, dtype=tl.int32)
        ts_r[victim] = current_time
    # write per-candidate outputs to separate buffers
    tl.store(out_idx_ptr + t * s0, tl.cast(best_idx, tl.int64))
    tl.store(out_flag_ptr + t * s1, tl.cast(is_hit, tl.int32))
```

2) Scalar control and constants
```python
flag_false = tl.zeros([], dtype=tl.int1)
one_i32 = tl.full([], 1, dtype=tl.int32)
```

Best practices / pitfalls
- Keep state in registers: spilling to HBM breaks sequential semantics and kills performance.  
- Use int32 for mutable boolean-like registers to avoid type inconsistencies across branches.  
- Avoid large register pressure: big caches or many heads may reduce occupancy; split if necessary.  
- Cast indices to tl.int64 before pointer arithmetic.  
- Ensure dtype consistency across branches to avoid compilation surprises.