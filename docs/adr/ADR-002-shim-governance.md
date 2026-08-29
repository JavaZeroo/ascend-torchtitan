# ADR-002: Shims are governed debt, not a compatibility layer

## Status
Accepted (2026-08-29)

## Context
The override mechanism only reaches `Configurable.Config` nodes. Some CUDA hard-coding sits in free functions (e.g. the AC D2H policy). A monkeypatch is the only out-of-tree tool, and monkeypatches rot silently.

## Decision
A shim registry (`ascend_titan.compat`) that rejects at import time any shim without `reason` and `upstream` (P4), and any `replace`-type shim without `why_not_wrap` (P3). Shims are applied only by `setup()`. Shim count is reported by `doctor` and expected to trend to zero.

## Alternatives
- Source fingerprints of the patched function: deferred; wrappers inherit upstream changes so fingerprints matter only for replace-type shims, which should be rare.
- A vendor-agnostic shim framework: rejected (single vendor cannot design a good abstraction; upstream is the right home for device-agnostic seams).

## Consequences
- Adding a shim requires filing an upstream issue first. This is intentional friction.
