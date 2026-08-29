---
description: Rules for the L0 shim layer
globs: ascend_titan/compat/**
---
# Shim rules
- Before writing a shim, grep `../torchtitan/torchtitan/config/configs.py` for a switch (P0).
  If one exists, the fix is a recipe delta, not a shim.
- Attribute first (CLAUDE.md table). Only `TT` failures may become shims. `NPU` never (P1).
- `kind="wrap"` by default. `kind="replace"` needs `why_not_wrap=` and a note in
  `docs/upstream-tracking.md` explaining what upstream change would let us delete it.
- One shim per file in `shims/`; file name = shim function name.
- Every shim gets a CPU unit test that patches a fake target and asserts the wrapper still
  calls the original.
- Update the table in `docs/upstream-tracking.md` in the same PR.
- Deleting a shim is a celebrated PR. Do it as soon as the upstream issue closes.
