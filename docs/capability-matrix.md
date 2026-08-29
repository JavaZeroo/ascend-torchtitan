# Capability matrix

Three states (P2): 🟢 works · 🔴 fails (attribution required) · ⚪ not evaluated.
Attribution codes: **TT** upstream torchtitan hard-codes CUDA (→ shim + upstream issue) · **NPU** torch_npu defect/missing API (→ torch_npu issue, no workaround, P1) · **CANN** unsupported (→ record, stop) · **DEP** CUDA-only third-party dep (→ record or Ascend replacement).

Base model: `qwen3_debugmodel_npu`. Validated tuple: see `constraints/npu.txt`. Updated by: CI (nightly) and humans (attribution column).

## Run paths (M1)

| path | state | attribution / note |
|---|---|---|
| 1 NPU eager | ⚪ | |
| fake_backend (1 NPU, NGPU=8) | ⚪ | |
| FSDP2 ×2 | ⚪ | |
| FSDP2 ×8 | ⚪ | |

## Parallelism (M2)

| axis | state | attribution / note |
|---|---|---|
| TP2 | ⚪ | |
| TP2 + SP | ⚪ | |
| PP2 | ⚪ | |
| CP2 | ⚪ | |
| EP (deepseek_v3 debugmodel) | ⚪ | |
| async TP | ⚪ | needs symmetric memory; expect 🔴 DEP |
| FSDP symm-mem | ⚪ | upstream guards `enable_fsdp_symm_mem` with a cuda capability check; expect 🔴 TT |

## Attention backend (M2)

| backend | state | attribution / note |
|---|---|---|
| flex | ⚪ | upstream default; FlexAttention = torch.compile + inductor (`attention.py:222`). **M1 cell.** |
| varlen | ⚪ | `torch.nn.attention.varlen` (nightly API, FA-backed). **M1 cell.** |
| flex_flash | 🔴 | TT-by-design: gated on `has_cuda_capability(9,0)`. Not planned. |
| sdpa | 🔴 | TT-by-design: upstream removed sdpa for LMs (`config_utils.py:97`, needs per-document positions). **Consequence: no plain-eager attention path exists; if flex and varlen are both 🔴, an Ascend inner-attention override is required for M1.** |

## Activation checkpointing (M2)

| mode | state | attribution / note |
|---|---|---|
| none | ⚪ | |
| selective (op) | ⚪ | D2H MUST_SAVE policy keyed on `"cuda"` string (`activation_checkpoint.py:251`): expect 🟢 with lost optimisation → wrap shim candidate |
| full | ⚪ | |

## Compile / graph (M2, M5)

| mode | state | attribution / note |
|---|---|---|
| eager | ⚪ | |
| inductor | ⚪ | |
| torchair | ⚪ | M5 |
| cuda graphs | 🔴 | TT-by-design: upstream `wrap_with_cuda_graph` already falls back to eager on non-CUDA with a warning. Not a bug; recorded so nobody re-investigates. |

## Precision (M5)

| mode | state | attribution / note |
|---|---|---|
| bf16 | ⚪ | |
| float8 | ⚪ | upstream converter gated by `has_cuda_capability(8,9)`; Ascend FP8 would be an override on the post-converter tree |
| mx / nvfp4 | 🔴 | CANN: NVIDIA-only formats. Not planned. |

## Other

| feature | state | attribution / note |
|---|---|---|
| DCP checkpoint save/load | ⚪ | |
| profiler | ⚪ | |
| deterministic mode | ⚪ | upstream sets `NCCL_*` env + `torch.backends.cuda.matmul.*` (`distributed/utils.py:347-379`); HCCL equivalents unknown |
| multimodal (flux) | ⚪ | M5 |
