---
name: shim-authoring
description: Write, test and register an L0 compat shim in ascend_titan/compat/shims. Use when a torchtitan failure is attributed TT (upstream CUDA hard-coding with no config switch) and an override cannot reach it.
---
# shim-authoring

## Gate (all must be true)
- [ ] Attribution is **TT** (not NPU/CANN/DEP). If NPU: stop, file a torch_npu issue (P1).
- [ ] No config switch exists: grep `../torchtitan/torchtitan/config/configs.py` and the relevant
      `Config` dataclass (P0).
- [ ] The code is not a `Configurable` node (else write an override, P6).
- [ ] An upstream issue is filed; you have the URL (P4).

## Write
1. `ascend_titan/compat/shims/<name>.py`:
   ```python
   from ascend_titan.compat import shim

   @shim(target="torchtitan.x.y:fn", reason="…", upstream="https://github.com/pytorch/torchtitan/issues/N")
   def <name>(original):
       def wrapped(*a, **k):
           …            # add NPU behaviour
           return original(*a, **k)   # always defer to upstream (P3)
       return wrapped
   ```
   Prefer wrapping a *small* helper over a big function. If you must `kind="replace"`, write
   `why_not_wrap=` honestly and add the shim to the replace-type list in `docs/upstream-tracking.md`.
2. Test in `tests/unit/test_shim_<name>.py` with the `clean_registry` fixture and a fake target
   module: assert the original is called and the added behaviour happens.
3. Add a row to `docs/upstream-tracking.md` → "Shims ↔ upstream issues".
4. Flip the matrix cell that motivated it and note the shim name in the attribution column.

## Verify
`pytest tests/unit -x && ascend-titan-doctor` (shim appears under "shims registered"), then the
motivating run on NPU.
