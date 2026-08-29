# ADR-003: Pin torchtitan by commit SHA

## Status
Accepted (2026-08-29)

## Context
See docs/upstream-tracking.md: releases are stale, main tracks torch nightly, torch_npu tracks torch releases.

## Decision
`constraints/npu.txt` holds `torchtitan_sha=`. `scripts/install.sh` checks it out and installs with `--no-deps` plus our dependency list. A bump is a PR carrying a full matrix run. `scripts/probe_compat.sh` reports how far forward we could move.

## Consequences
- `pip install ascend-torchtitan` alone is not enough; the install script is part of the product.
- CI runs two legs: pinned (gate) and main (drift probe, allowed to fail).
