# constraints

| file | role |
|---|---|
| `torchtitan.sha` | The pinned torchtitan commit. |
| `npu.txt` | **NEXT** track (default): torch 2.13.0 + torch_npu 2.13.0rc1. Pure pip syntax. Together with `torchtitan.sha`: source of truth for "which versions are we on". |
| `npu-stable.txt` | **STABLE** track: torch 2.12.0 + torch_npu 2.12.0 (GA). Same torchtitan SHA. Select with `CONSTRAINTS=constraints/npu-stable.txt ./scripts/install.sh`. |
| `titan-deps.txt` | torchtitan's dependency list **without** the `attn-gym[linear]` extra (which pulls `nvidia-cutlass-dsl[cu13]`). |

Why not `pip install torchtitan`? Its `requirements.txt` hard-pins `attn-gym[linear]==0.0.5`; the
`[linear]` extra drags CUDA-only wheels onto an aarch64/NPU box. `scripts/install.sh` installs
torchtitan with `--no-deps` and then this list.

attn_gym's KDA has a pure-torch `naive` fallback, so kimi_k3 still runs (slowly) without the extra.

Both tracks are validated nightly; a change that is green on one and red on the other is a torch-version issue by definition (see docs/baseline.md).

**Always install with `-c`.** `pip install torchvision` without `-c constraints/<track>.txt` upgrades
`torch` and breaks the torch_npu ABI (`undefined symbol ... deleteNode`; "Failed to load the backend
extension: torch_npu"). `scripts/install.sh` does this right; if you install by hand, pass the
constraints file every time.
