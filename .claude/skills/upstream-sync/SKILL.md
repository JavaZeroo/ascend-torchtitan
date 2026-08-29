---
name: upstream-sync
description: Bump the pinned torchtitan SHA in constraints/npu.txt safely — probe importability, re-run CPU tests, run the NPU matrix, update shims and docs. Use when CI's main leg shows drift, when a needed upstream feature landed, or on a scheduled sync.
---
# upstream-sync

Bumping is a PR (P5). Never edit `constraints/torchtitan.sha` outside this procedure.

1. `./scripts/probe_compat.sh` → candidate SHA = furthest importable (or the specific commit you need
   if it imports).
2. `git -C ../torchtitan log --oneline <old>..<new> -- torchtitan/config torchtitan/protocols torchtitan/models/common`
   — read every commit touching extension surfaces. For each shim target and each override target,
   confirm it still exists (`apply_all()` will error on a moved target; overrides fail at build).
3. Update `constraints/torchtitan.sha`; `./scripts/install.sh`; `pytest tests/unit -x`.
4. Sync `constraints/titan-deps.txt` with `../torchtitan/.ci/docker/requirements.txt` (minus
   `attn-gym[linear]`).
5. NPU: full matrix run; paste the diff of `docs/capability-matrix.md` into the PR. Any new 🔴 needs
   attribution before merge.
6. Check `docs/upstream-tracking.md`: any shim whose upstream issue is closed ⇒ delete the shim in
   this PR.
7. PR title: `sync: torchtitan <old7> → <new7>`; body = probe report + matrix diff + shims removed.
