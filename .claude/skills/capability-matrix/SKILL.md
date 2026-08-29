---
name: capability-matrix
description: Record the result of an NPU run in docs/capability-matrix.md with a correct attribution (TT / NPU / CANN / DEP). Use after any training run, matrix sweep, or CI nightly, and when triaging a red cell.
---
# capability-matrix

## Flip a cell
1. Identify the row (feature/axis) and the validated tuple (`constraints/npu.txt`).
2. 🟢: write the date and the command. 🔴: attribution is mandatory (table in CLAUDE.md), plus the
   issue link. ⚪ → never go back to ⚪ once measured.
3. Attribution decides the owner:
   - **TT** → upstream issue; consider a wrap-shim (skill `shim-authoring`).
   - **NPU** → torch_npu issue; **no workaround** (P1). Link it in `docs/upstream-tracking.md`.
   - **CANN** → note the error code; no further work.
   - **DEP** → note the package; if it is a kernel, an L1 replacement is an M3+ task.

## Run a sweep
`python -m ascend_titan.tools.matrix --cards 0-7 --jobs 4 --out outputs/matrix/<date>` (see `docs/matrix/README.md`).
It applies `npu_baseline` to each upstream test config and auto-triages reds via the `TRIAGE`
regex table. An `UNKNOWN` code means a new failure signature: read the log, decide the code,
add a regex row to `TRIAGE` and an entry to `docs/issues/<owner>.md`, then re-triage
(`results.json` keeps the log paths).

## Triage a red cell quickly
```
grep -m1 -nE "torch_npu/|torchtitan/|attn_gym|helion|deep_ep|cutlass|EZ[0-9]{4}|EI[0-9]{4}" <log>
```
The first hit usually gives the attribution. Same failure across many cells ⇒ one root cause;
attribute once, reference it from the others.

## Batch, don't stream
When sweeping, run everything first, then attribute together: red cells cluster by root cause.
