# Principles

These are cited in code review. Number them when you argue.

| # | Principle | Why |
|---|---|---|
| **P0** | **Configuration before patching.** If torchtitan exposes a switch (`--compile.backend`, `--training.disable_cuda_graphs`, `attn_backend=`), use it. A shim is only for code with no switch. | Every shim is a liability; a config line is not. Static analysis of upstream showed most "CUDA hard-coding" is already behind a switch. |
| **P1** | **Never work around torch_npu.** A failure attributed to torch_npu gets an issue link and a 🔴 cell in the capability matrix. No in-repo bypass. | Bypasses outlive the bug they hide, and hide the demand signal from torch_npu. This is the project's red line. |
| **P2** | **Scope cuts are scheduling, not exclusion.** Multimodal, low precision, more models are on the roadmap. The matrix is three-state: 🟢 / 🔴 (with attribution) / ⚪ not evaluated. | "Not yet tested" and "tested, doesn't work" must never share a cell. |
| **P3** | **Wrap, don't replace.** A shim should call the original and add behaviour around it so upstream changes are inherited. `kind="replace"` requires `why_not_wrap`. | Replacement shims silently drop upstream improvements; wrappers don't. |
| **P4** | **Every shim carries an upstream issue.** Enforced by the registry at import time. A shim is debt with a due date; delete it when upstream lands the fix. | Shim count is a health metric that should trend to zero. |
| **P5** | **Versions are commit SHAs; bumps are PRs.** `constraints/npu.txt` holds the torchtitan SHA. A bump PR attaches a full matrix run. | Upstream releases lag main by 6+ months and lack the features we depend on. |
| **P6** | **Only target upstream `Configurable` nodes.** A fused kernel replaces an existing `Config` node via `@override`. If the computation has no node, ask upstream to extract one; do not replace the parent block. | Replacing a parent block is a fork of that block, and every upstream change to it becomes our merge. |
| **P7** | **Degrade loudly, record provenance.** Missing kernel deps fall back to upstream eager with a WARNING and a provenance entry. Benchmarks without a provenance table are not accepted. | Silent degradation corrupts performance data; loud degradation keeps everything runnable. |
