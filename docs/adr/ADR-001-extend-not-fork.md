# ADR-001: Extend torchtitan out-of-tree; never fork

## Status
Accepted (2026-08-29)

## Context
torchtitan moves at ~170 commits/month and provides sanctioned extension points: full-path `--module`, `config.build()` on subclassed `Trainer.Config`, `ModelSpec` callables, and the `@override` mechanism whose README explicitly targets hardware vendors and says vendor kernels should live in an external package.

## Decision
`ascend-torchtitan` is a separate package. It imports torchtitan, never patches its source tree, and layers: L0 governed shims (only for code no extension point reaches), L1 overrides, L2 parallel/graph, L3 recipes, L4 tooling.

## Alternatives
- **Fork**: every upstream commit becomes a merge; rejected.
- **Self-contained distribution pinned to a release**: releases lag main by 6+ months and lack the override mechanism and kimi_k3; rejected.

## Consequences
- We depend on upstream `Config` shapes; `derive()` and the CPU recipe tests are the early-warning system.
- Some computations (kimi_k3 attn_res) are unreachable without an upstream change; those become upstream asks, not parent-block replacements (P6).
