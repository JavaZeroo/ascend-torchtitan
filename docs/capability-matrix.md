# Capability matrix

Three states (P2): 🟢 works · 🔴 fails (attribution required) · ⚪ not evaluated.
Attribution: **TT** torchtitan (→ `docs/issues/torchtitan.md`) · **NPU** torch_npu (→ `docs/issues/torch_npu.md`, never worked around, P1) · **TORCH** pytorch core (→ `docs/issues/pytorch.md`) · **CANN** unsupported · **DEP** CUDA-only third-party dep.

Base recipe: `qwen3_debugmodel_npu` (see `docs/baseline.md`). Tuples: NEXT = torch 2.13.0 / torch_npu 2.13.0rc1, STABLE = 2.12.0 / 2.12.0; torchtitan `13da2d77c`. Unless a cell says otherwise, results are identical on both tuples. Last full pass: 2026-08-29.

## Run paths (M1)

| path | state | attribution / note |
|---|---|---|
| 1 NPU eager, 10 steps | 🟢 | golden `tests/assets/losses/npu/qwen3_debugmodel_npu__*.txt` |
| FSDP2 ×2, 10 steps | 🟢 | golden `..._fsdp2__*.txt`; HCCL |
| fake_backend (1 NPU, NGPU=8) | 🔴 | NPU-2: fake process group has no `npu` device; reaches the first collective (`set_determinism` broadcast, or all_reduce with `--debug.seed`) |
| FSDP2 ×8 | ⚪ | |

## Attention backend

| backend | state | attribution / note |
|---|---|---|
| ascend_fusion (override on varlen node) | 🟢 | `ascend_titan.kernels.attention`; GQA, per-document causal; op-level max abs err 8e-3 vs fp32 SDPA (bf16) |
| varlen (stock) | 🔴 | NPU-1: no `aten::_flash_attention_forward` NPU kernel |
| flex (stock, upstream default) | 🔴 | TORCH-1: torch rejects npu devices in `flex_attention` (also needs `separate_full_blocks` → torch ≥ 2.13, TT-3) |
| flex_flash | 🔴 | TT-by-design: `has_cuda_capability(9,0)` gate |
| sdpa | 🔴 | TT-7: removed upstream for LMs |

## Loss

| loss | state | attribution / note |
|---|---|---|
| CrossEntropyLoss | 🟢 | M1 default |
| ChunkedLossWrapper (upstream default) | 🔴 | TT-4: backward "data is not allocated yet" (lm_head manual unshard path); with or without AC |

## Parallelism

| axis | state | attribution / note |
|---|---|---|
| FSDP2 (dp_shard 2) | 🟢 | `partial_dtensor` |
| spmd_backend = spmd_types (upstream default) | 🔴 | TT-5: params reach `fully_shard(dp_mesh_dims=)` as plain tensors |
| TP2 / TP2+SP / PP2 / CP2 / EP | ⚪ | M2 |
| async TP | ⚪ | needs symmetric memory; expect 🔴 DEP |
| FSDP symm-mem | ⚪ | upstream gates on cuda capability; expect 🔴 TT |
| CP with ascend_fusion | 🔴 | OURS-2: LSE epilogue not implemented |

## Activation checkpointing

| mode | state | attribution / note |
|---|---|---|
| selective (op) — upstream default | 🟢 | D2H `MUST_SAVE` policy keyed on `"cuda"` string (`activation_checkpoint.py:251`) is inert on NPU: correct, loses one optimisation → wrap-shim candidate (not done) |
| none | 🟢 | |
| full | ⚪ | |

## Compile / graph

| mode | state | attribution / note |
|---|---|---|
| eager | 🟢 | |
| inductor | ⚪ | |
| torchair | ⚪ | M5 |
| CUDA graphs | 🔴 | TT-by-design: upstream falls back to eager on non-CUDA with a warning. Not a bug. |

## Precision

| mode | state | attribution / note |
|---|---|---|
| bf16 | 🟢 | |
| float8 | ⚪ | upstream converter gated by `has_cuda_capability(8,9)`; Ascend FP8 = override on the post-converter tree (M5) |
| mx / nvfp4 | 🔴 | CANN: NVIDIA-only formats. Not planned. |

## Other

| feature | state | attribution / note |
|---|---|---|
| import torchtitan.trainer | 🟢 | needs the `triton` wheel (TT-1) — in `constraints/titan-deps.txt` |
| PG timeout shortening after step 1 | 🟡 | polyfilled; `_set_pg_timeout` ignores HCCL on torch ≤ 2.13 (TORCH-3) |
| DCP checkpoint save/load | ⚪ | |
| profiler | ⚪ | |
| deterministic mode (`--debug.deterministic`) | 🟢 | bitwise-identical curves across NEXT/STABLE; upstream also sets `NCCL_*` env + `torch.backends.cuda.matmul.*` — inert on NPU |
| multimodal (flux) | ⚪ | M5 |
