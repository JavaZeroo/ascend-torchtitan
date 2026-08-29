---
description: Rules for L1 override modules
globs: ascend_titan/kernels/**
---
# Override rules
- Module must be import-safe without the kernel dependency installed: probe the dep inside a
  try/except at module level, log a WARNING, and skip `@override` registration on failure (ADR-004).
  Never raise at import time.
- Target only an existing upstream `Configurable.Config` (P6). Cite the upstream file:line of the
  node in the docstring.
- Build the replacement config with `derive(cfg, NewConfig, **deltas)`; never hand-copy fields.
- Expose per-instance selection through `fqns=` on `@override`, not by inspecting fields in the factory.
- Wrap the kernel as `torch.library.custom_op` with `register_fake` and `register_autograd`; ship
  an `opcheck` test and an alignment test against the upstream eager module.
- If the replacement changes parameter layout, bridge checkpoints with
  `register_state_dict_post_hook` / `register_load_state_dict_pre_hook` (see upstream
  `torchtitan/overrides/fused_swiglu.py`).
- Read `../torchtitan/torchtitan/overrides/README.md` before writing the first override.
