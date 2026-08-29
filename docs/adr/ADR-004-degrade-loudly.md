# ADR-004: Missing kernel dependencies degrade loudly to upstream eager

## Status
Accepted (2026-08-29)

## Context
Fused kernels depend on separately-installed packages (triton-ascend-kernels, fla-npu, ops-nn, ops-transformer). Options: hard-fail, silent fallback, or explicit three-way impl selection.

## Decision
Fallback = do not register the override, so upstream's own torch implementation runs. Emit a WARNING and a provenance entry. Benchmarks must carry the provenance table (P7).

## Alternatives
- Hard fail: kills runnability in partial environments.
- Explicit `impl=ascendc|triton|eager` per kernel (MindSpeed-MM style): more code (three implementations each); the eager path already exists upstream for free.

## Consequences
- Performance numbers are only trusted with provenance; nightly perf baselines are the second line of defence.
