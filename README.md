# triton-skills

Agent skills for writing optimized [Triton](https://triton-lang.org/) GPU kernels. Built for [Claude Code](https://claude.ai/code), generated with [Upskill](https://github.com/huggingface/upskill) and refined with verified patterns from production kernels.

## What's in the box

A progressive-disclosure skill set: the main skill covers core Triton patterns that apply to ~80% of kernel tasks. Specialized sub-skills are loaded on demand when the task involves attention, normalization, quantization, or advanced matmul.

### Core (`skills/SKILL.md`)
Always-loaded fundamentals — `@triton.jit`, block tiling, masking, autotune, grid launching, FP32 accumulation, benchmarking. Includes ready-to-use softmax and dropout examples.

### Specialized sub-skills

| File | Topic | Haiku Lift |
|------|-------|------------|
| `triton-flash-attention-v2.md` | FlashAttention v2 — online softmax, causal masking, GQA head routing, multi-stream accumulators | 20% |
| `triton-fused-normalizations.md` | LayerNorm / RMSNorm — standalone & fused epilogue, backward two-stage reductions | 40% |
| `triton-persistent-warp-matmul.md` | Persistent kernels, warp specialization, TMA descriptors (Hopper+) | 20% |
| `triton-quantized-block-scaled-gemm.md` | FP4/FP8 block-scaled GEMM, OCP microscaling, `tl.dot_scaled` | 20% |
| `triton-memory-efficient-patterns.md` | Philox seed-based dropout, kernel fusion, activation recomputation | 20% |
| `triton-gpu-kernel-optimization.md` | Tiled GEMM, stride-based addressing, L2-aware tile ordering, autotune | 20% |
| `triton-fused-epilogue-kernels.md` | Fuse normalization, gating, residual, dropout into attention/matmul epilogues | 40% |
| `triton-sequential-stateful-blocks.md` | Sequential stateful kernels — LRU routing, mutable register state via `tl.where` | 40% |
| `triton-dynamic-launcher-tiling.md` | Launcher tile heuristics, num_stages/num_warps, dtype-aware sizing, fallback patterns | 60% |

**Haiku Lift** = improvement in task success rate with the skill vs. without, evaluated on Haiku (9/9 pass). Sonnet shows lower marginal lift since it already knows many Triton patterns without guidance.

## Installation

Copy the `skills/` directory into your project, or install as a Claude Code skill:

```bash
# From your project root
git clone https://github.com/anthony-maio/triton-skills.git .claude/skills/triton-skills
```

Then reference the skill in your Claude Code configuration.

## How these were made

Each skill was generated and refined through a multi-stage process:

1. **Generation** — A detailed Triton task description (from the [official tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) and real-world patterns) is sent to **Opus** via [Hugging Face Upskill](https://github.com/huggingface/upskill)
2. **Evaluation** — Synthetic test cases verify that **Haiku** performs better with the skill than without (up to 3 refinement rounds)
3. **Multi-model review** — GPT-5.2, Claude Opus 4.6, and Gemini 3.1 Pro independently reviewed all skills for correctness bugs (offset arithmetic, deprecated APIs, unsupported builtins)
4. **Production kernel refinement** — 6 skills were rewritten using verified patterns from production Triton kernels (differential FlashAttention, LRU bank routing) with known tolerance bounds and test suites

## License

MIT
