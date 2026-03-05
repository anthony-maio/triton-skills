# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Claude Code agent skills for writing optimized Triton GPU kernels, generated with Hugging Face Upskill.

## Repository Structure

- `skills/SKILL.md` — Main router skill (core patterns + progressive disclosure table)
- `skills/triton-*.md` — Specialized sub-skills (attention, normalization, quantized matmul, etc.)

## Skill Architecture

Progressive disclosure: `skills/SKILL.md` is always loaded and covers core Triton patterns (~80% of tasks). The routing table at the bottom directs Claude to read specialized files only when the task involves attention, normalization, quantization, persistent kernels, or memory optimization.

## Regenerating Skills

Generation scripts (`generate_triton_skill.py`, `generate_triton_batch.py`) are gitignored but kept locally. They require `ANTHROPIC_API_KEY` in `.env` and use Upskill's FastAgent context:

```python
async with _fast_agent_context() as agent:
    skill = await generate_skill(task, generator=agent.skill_gen, model="opus")
    tests = await generate_tests(task, generator=agent.test_gen)
    results = await evaluate_skill(skill, tests, evaluator=agent.evaluator, model="haiku")
```
