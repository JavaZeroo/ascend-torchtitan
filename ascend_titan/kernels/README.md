# kernels (L1) — fused-kernel overrides

Each file wraps one Ascend kernel (AscendC / Triton-Ascend) as a torchtitan
`@override` factory targeting an existing `Configurable.Config` node.

Rules of the road:
- Only target nodes that exist upstream (`docs/design` §"Configurable 节点判据").
  If the computation has no node, do **not** replace the parent block; record it
  in `docs/upstream-tracking.md` as an upstream ask.
- Register the kernel as a `torch.library.custom_op` with `register_fake` and
  `register_autograd`; ship an `opcheck` test.
- Missing dependency ⇒ log a loud warning and do not register (the upstream
  eager path is the fallback). Never silently swap in a slow path without a log
  line and a provenance entry.

Planned for M3+: `situ_glu.py`, `kda.py`, `causal_conv1d.py`, `attention.py`
(`npu_fusion_attention`), `rmsnorm.py`, `rope.py`.
