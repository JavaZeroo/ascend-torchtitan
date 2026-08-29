# ADR-005: This repository is Ascend-only; device-agnostic seams go upstream

## Status
Accepted (2026-08-29)

## Context
Other hardware vendors will want the same kind of package. Should this repo host a vendor-agnostic skeleton?

## Decision
No. Code here says `npu`. Anything a second vendor would also need (device capability queries, device-graph abstraction, attention backend registry) belongs in torchtitan core; we propose it there when we hit it. Upstream's per-node override conflict detection already lets multiple vendors' overrides coexist without coordination.

## Consequences
- If a second vendor package ever shares code with us, that is a refactor at that time, with two data points instead of one guess.
