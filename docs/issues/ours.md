# ascend-torchtitan — known gaps (owner: this repo)

| id | gap | plan |
|---|---|---|
| OURS-1 | `AscendFusionAttention` does one D2H sync per step to turn `cu_seq_*` into host ints (`_host_offsets`). | Ask upstream for `include_host_offsets=True` in `get_attention_masks` when the inner attention declares it needs host offsets; or cache per positions-tensor. |
| OURS-2 | `AscendFusionAttention` rejects `out_transform` (LSE epilogue) → no context parallel / attention sinks yet. | M4: return LSE from `softmax_max/softmax_sum`. |
| OURS-3 | Sliding-window (`window_size=(W,0)`) path uses `sparse_mode=4` untested. | Add an NPU test before enabling any model that needs it. |
| OURS-4 | No provenance table yet; degraded overrides are visible only in logs. | M3. |
| OURS-5 | NPU golden frozen (`tests/assets/losses/npu/`, bitwise across NEXT/STABLE) but **not yet compared to a GPU run** of the same recipe; upstream's `qwen3_a10g.txt` is for a different config (MoE param-groups, fsdp2+tp2+cp2+ep8). | Run `qwen3_debugmodel_npu` deltas on a GPU box once and record the loose-tolerance comparison. |
| OURS-6 | Shim `upstream` links are `draft:` pointers until issues are filed. | File TT-1, TT-2, NPU-1, NPU-2, TORCH-1; replace pointers. |
