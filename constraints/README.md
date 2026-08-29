# constraints

| file | role |
|---|---|
| `npu.txt` | The pinned torchtitan SHA + pip constraints for the NPU environment. Source of truth for "which versions are we on". |
| `titan-deps.txt` | torchtitan's dependency list **without** the `attn-gym[linear]` extra (which pulls `nvidia-cutlass-dsl[cu13]`). |

Why not `pip install torchtitan`? Its `requirements.txt` hard-pins `attn-gym[linear]==0.0.5`; the
`[linear]` extra drags CUDA-only wheels onto an aarch64/NPU box. `scripts/install.sh` installs
torchtitan with `--no-deps` and then this list.

attn_gym's KDA has a pure-torch `naive` fallback, so kimi_k3 still runs (slowly) without the extra.
