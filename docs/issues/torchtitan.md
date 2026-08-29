# torchtitan — problems / asks (owner: pytorch/torchtitan)

Measured at torchtitan `13da2d77c` (2026-08-29). "nightly-only" = needs a torch feature not in any released torch (≤ 2.13.0).

## TT-1 unconditional `import triton` in core import chain — `draft`
- `torchtitan/models/common/token_dispatcher.py:17 → distributed/minimal_async_ep/api.py:30 → minimal_async_ep/kernels.py:10 import triton`. `import torchtitan.trainer` fails wherever the `triton` wheel is absent, even though MinimalAsyncEP is opt-in.
- Ask: import triton lazily inside the kernels' functions (as `overrides/fused_swiglu.py` does behind a try/except), or guard with `find_spec`. Workaround here: `triton` added to `constraints/titan-deps.txt` (pure-Python wheel; it is *not* Triton-Ascend).

## TT-2 `torch.distributed.set_timeout` used without a feature check — `draft` <a name="set-timeout"></a>
- `torchtitan/distributed/utils.py::set_pg_timeouts` (since #3764, 2026-06-27) calls the nightly-only public API after the first train step → `AttributeError` on torch ≤ 2.13.
- Ask: fall back to `torch.distributed.distributed_c10d._set_pg_timeout` (identical semantics; nightly keeps it as a deprecated alias) or gate with `check_if_feature_in_pytorch`. Polyfilled here (`compat/shims/dist_set_timeout.py`).

## TT-3 `create_block_mask(separate_full_blocks=...)` nightly-only — `info`
- `models/common/attention.py::create_attention_mask` since #4106 (2026-08-11). Present in torch 2.13.0, absent in 2.12.x. Moot on NPU while TORCH-1 stands.

## TT-4 ChunkedLossWrapper backward fails on NPU with "data is not allocated yet" — `investigating`
- `components/loss.py` drives FSDP's lm_head `unshard()` explicitly (since #4143, 2026-08-13). On torch 2.12.0/2.13.0 + torch_npu the backward raises `RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet`, with or without activation checkpointing; plain `CrossEntropyLoss` works. Attribution still open (torch-version FSDP semantics vs NPU allocator/event handling). Matrix cell `loss/chunked` = 🔴.

## TT-5 `spmd_types` backend: params reach `fully_shard(dp_mesh_dims=)` as plain tensors on NPU — `investigating`
- With the default `parallelism.spmd_backend="spmd_types"` (since #4085, 2026-08-18) and dp_shard>1, FSDP raises "all parameters must be DTensors on the full SPMD mesh". `partial_dtensor` works and is used by the recipes. Matrix cell `parallel/spmd_types` = 🔴 pending attribution.

## TT-6 kimi_k3 attention residual is a free function (no `Configurable` node) — `ask, deferred`
- `models/kimi_k3/model.py:135 _apply_attention_residual`; overriding it requires replacing the whole `KimiK3TransformerBlock.Config`. Extracting a `Module` with a `sharding_config` also resolves upstream's own `TODO: Add TP Support`. Wait for kimi_k3 to stabilise (landed 2026-08-24).

## TT-7 `sdpa` inner attention removed for LMs — `info`
- `models/common/config_utils.py:97`. Consequence for non-CUDA backends: no eager attention path exists upstream. Recorded; the override mechanism is the sanctioned answer.
