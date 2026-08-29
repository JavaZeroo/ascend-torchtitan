# Baseline

Established 2026-08-29 (M0 + M1). Hardware: Ascend 910B2 ×8, driver 25.5.1, CANN 9.1.0
(image `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12-devel`).

## Version tuples

| track | torch | torch_npu | torchtitan | file | status |
|---|---|---|---|---|---|
| **NEXT** (default) | 2.13.0 (+cpu wheel) | 2.13.0rc1 | `13da2d77c` (main, 2026-08-29) | `constraints/npu.txt` | 🟢 M1 |
| STABLE | 2.12.0 (+cpu wheel) | 2.12.0 | `13da2d77c` | `constraints/npu-stable.txt` | 🟢 M1 |

Both tracks produce **bitwise-identical** loss/grad_norm curves for the M1 recipes under `--debug.seed 42 --debug.deterministic` (`tests/assets/losses/npu/`). torch is installed from the CPU index; torch_npu provides the device backend and is autoloaded by `import torch` (`torch_npu autoload True` in `ascend-titan-doctor`).

Why not torch nightly: torch_npu tracks torch releases (newest pre-release: 2.13.0rc1); torch nightly is 2.15.0.dev. The gap is bridged by **one polyfill shim** (`torch.distributed.set_timeout`), everything else torchtitan main needs is present in 2.13.0 — see `docs/issues/torchtitan.md`.

## What M1 runs
`qwen3_debugmodel_npu` = upstream `qwen3_debugmodel` + 4 deltas:
1. inner attention `varlen` + override `ascend_titan.kernels.attention.npu_fusion_attention` (stock flex/varlen cannot run on NPU: TORCH-1, NPU-1);
2. `parallelism.spmd_backend = partial_dtensor` (TT-5);
3. checkpointing off (DCP is its own matrix cell);
4. `CrossEntropyLoss` instead of `ChunkedLossWrapper` (TT-4).

| path | cards | result (seed 42, deterministic) |
|---|---|---|
| 1 NPU eager | 1 | step 1 loss 7.6546 → step 10 loss 5.10304 grad_norm 3.3061, ~55k tps, 2.4 GiB |
| FSDP2 ×2 | 2 | step 10 loss 5.07792 grad_norm 3.3201, ~51k tps, 2.2 GiB |
| fake_backend | 1 | 🔴 NPU-2 (fake PG has no npu) |

## Shims in force
| shim | kind | why | goes away when |
|---|---|---|---|
| `dist_set_timeout` | polyfill | torchtitan calls nightly-only `torch.distributed.set_timeout` after step 1 | torch ships it (auto no-op) or torchtitan adds a fallback (TT-2) |

Note: on torch ≤ 2.13 `_set_pg_timeout` only handles nccl/gloo and warns `Set timeout is now only supported for either nccl or gloo`; HCCL group timeouts are therefore **not** shortened after step 1 (behavioural difference vs CUDA, recorded as TORCH-3).

## Reproduce
```bash
WITH_TORCH=1 ./scripts/install.sh                 # NEXT track
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 ./scripts/check_golden.sh qwen3_debugmodel_npu
ASCEND_RT_VISIBLE_DEVICES=0,1 NPU=2 ./scripts/check_golden.sh qwen3_debugmodel_npu_fsdp2
```
