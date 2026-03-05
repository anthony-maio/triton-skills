# triton-skills

Agent skills for writing optimized [Triton](https://triton-lang.org/) GPU kernels. Built for [Claude Code](https://claude.ai/code), generated with [Upskill](https://github.com/huggingface/upskill).

## What's in the box

A progressive-disclosure skill set: the main skill covers core Triton patterns that apply to ~80% of kernel tasks. Specialized sub-skills are loaded on demand when the task involves attention, normalization, quantization, or advanced matmul.

### Core (`skills/SKILL.md`)
Always-loaded fundamentals — `@triton.jit`, block tiling, masking, autotune, grid launching, FP32 accumulation, benchmarking. Includes ready-to-use softmax and dropout examples.

### Specialized sub-skills

| File | Topic | Eval Lift |
|------|-------|-----------|
| `triton-flash-attention-v2.md` | Flash Attention v2 — online softmax, causal masking, recompute-based backward | 20% |
| `triton-fused-normalizations.md` | LayerNorm / RMSNorm / GroupNorm — forward + backward, two-stage reductions | 40% |
| `triton-persistent-warp-matmul.md` | Persistent kernels, warp specialization, TMA descriptors (Hopper+) | 20% |
| `triton-quantized-block-scaled-gemm.md` | FP4/FP8 block-scaled GEMM, OCP microscaling, `tl.dot_scaled` | 20% |
| `triton-memory-efficient-patterns.md` | Philox seed-based dropout, kernel fusion, activation recomputation | 20% |
| `triton-gpu-kernel-optimization.md` | Tiled GEMM, L2 cache-aware tile ordering, autotune sweeps | 20% |

**Eval lift** = improvement in task success rate with the skill vs. without, evaluated on Haiku.

## Installation

Copy the `skills/` directory into your project, or install as a Claude Code skill:

```bash
# From your project root
git clone https://github.com/YOUR_USERNAME/triton-skills.git .claude/skills/triton-skills
```

Then reference the skill in your Claude Code configuration.

## How these were made

Each skill was generated using [Hugging Face Upskill](https://github.com/huggingface/upskill):

1. A detailed Triton task description (derived from the [official tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)) is sent to **Opus** as the teacher model
2. Synthetic test cases are generated for the task
3. **Haiku** evaluates task completion with and without the skill
4. If the skill doesn't improve results, it's refined based on failure descriptions (up to 3 rounds)

All 6 skills passed evaluation on the first or second attempt.

## License

MIT
