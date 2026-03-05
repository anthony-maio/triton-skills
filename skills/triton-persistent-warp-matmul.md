---
name: triton-persistent-warp-matmul
description: Teach an AI agent to implement persistent, warp-specialized matmul kernels in Triton using TMA and producer/consumer warps.
---

# Persistent & Warp-Specialized Matmul Kernels in Triton

Overview
This skill teaches how to implement a persistent GEMM in Triton where fewer thread blocks than output tiles are launched and each block iterates over multiple tiles. It covers tile scheduling (linear tile_id → 2D via divmod), persistent loop strides, TMA/device descriptors, producer/consumer warp roles, and epilogue subtiling for memory efficiency.

Step-by-step / Key principles
1. Partitioning and constants:
   - Define tile sizes BLOCK_M × BLOCK_N and inner block BLOCK_K.
   - num_tiles = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N). Use cdiv(x,y) = (x+y-1)//y.
2. Persistent scheduling:
   - Launch num_blocks < num_tiles. Each block computes:
     for tile_id in range(start_tile + block_id, num_tiles, num_blocks)
   - Convert linear tile_id to 2D: m_block, n_block = divmod(tile_id, num_tiles_n) (or use tile_id // num_tiles_n, tile_id % num_tiles_n).
3. Warp specialization:
   - Split warps into producers (async TMA loads or tl.async_copy into shared memory) and consumers (wait on barrier, compute tl.dot).
   - Producers write tiles to sA/sB, then tl.barrier(); consumers perform tl.dot using shared tiles.
4. TMA / async loads:
   - On SM90+, create device descriptors: desc = tl.make_tensor_descriptor(ptr, shape, strides, block_shape) and use tl.tma_load / tl.tma_store.
5. Epilogue and subtile:
   - Write output in subtile chunks to reduce shared memory and register pressure.
6. Numerical and synchronization:
   - Use fp32 accumulators for mixed precision and careful barrier placement between producer/consumer groups.

Practical examples
- Persistent tile loop + divmod:
```python
def cdiv(a,b): return (a + b - 1) // b

num_tiles_m = cdiv(M, BLOCK_M)
num_tiles_n = cdiv(N, BLOCK_N)
num_tiles = num_tiles_m * num_tiles_n

start = tl.program_id(0)  # persistent start index
num_blocks = tl.num_programs(0)

for tile_id in range(start, num_tiles, num_blocks):
    m_block, n_block = divmod(tile_id, num_tiles_n)
    m0 = m_block * BLOCK_M
    n0 = n_block * BLOCK_N
    # load A_tile, B_tile (producers) -> tl.async_copy or TMA
    # tl.barrier()
    # consumers compute:
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = load_A_tile(m0, k0)
        b = load_B_tile(k0, n0)
        acc += tl.dot(a, b)   # use tl.dot
    # write C_tile subtile-wise
```

Best practices & pitfalls
- Tune BLOCK_M/BLOCK_N to balance shared memory, registers, and TMA granularity.
- Ensure correct alignment and block_shape when creating TMA descriptors.
- Carefully design producer/consumer warp split to avoid idle warps.
- Profile with Triton Proton and compare against cuBLAS; persistent kernels benefit when kernel launch overhead is significant or when overlapping loads and compute increases utilization.