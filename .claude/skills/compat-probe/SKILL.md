---
name: compat-probe
description: M0 procedure — establish the torch / torch_npu / CANN / torchtitan version tuple on an NPU box, find the furthest importable torchtitan SHA, answer whether torch autoloads torch_npu, and file torch_npu missing-API issues. Use when setting up a new environment, when `ascend-titan-doctor` shows a mismatch, or before a torchtitan SHA bump.
---
# compat-probe

Goal: fill `constraints/npu.txt` with facts, not hopes. Output is a short report pasted into the PR.

## Steps
1. `ascend-titan-doctor --json > /tmp/doctor.json` on the NPU box. Record torch, torch_npu, CANN,
   `torch_npu_autoload`. If autoload is `false`, `python -m ascend_titan.train` is mandatory (F4)
   — say so in the report.
2. `./scripts/install.sh` (uses the pinned SHA). If `import torchtitan.trainer` fails:
   - read the traceback; classify per CLAUDE.md attribution table;
   - a missing `torch.*` API ⇒ that is the torch-release gap (F1). Note the API and the upstream
     commit that introduced it: `git -C ../torchtitan log -S '<api name>' --oneline | tail -1`.
3. `./scripts/probe_compat.sh` — records the furthest importable SHA and the first breaking commit.
4. Smoke: `COMM_MODE=fake_backend NPU=8 ./scripts/run_train.sh` then
   `NPU=1 ./scripts/run_train.sh --parallelism.data_parallel_shard_degree 1`.
5. Fill the "Run paths (M1)" rows in `docs/capability-matrix.md` and the pins in
   `constraints/npu.txt`. Every missing torch_npu API gets an issue link in
   `docs/upstream-tracking.md` → "Issues we filed". No workarounds (P1).

## Report template
```
tuple: torch=… torch_npu=… CANN=… torchtitan=<sha> (pinned) furthest=<sha>
autoload: yes|no
import torchtitan.trainer: ok | fails at <module>: <api> (introduced by <commit>)
fake_backend smoke: 🟢|🔴 <attribution>
1-NPU eager smoke:  🟢|🔴 <attribution>
issues filed: …
go/no-go: …
```
