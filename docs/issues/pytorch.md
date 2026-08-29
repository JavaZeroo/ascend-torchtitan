# pytorch — problems (owner: pytorch/pytorch)

## TORCH-1 FlexAttention hard-codes a device whitelist — `draft`
- `torch/nn/attention/flex_attention.py` (2.12: "only supported on CUDA, CPU or HPU"; 2.13 and nightly 2.15.0.dev20260828: "... HPU, or MPS"): `ValueError` for any privateuse1 device before any backend gets a chance.
- Impact: torchtitan's default LM attention (`attn_backend="flex"`) can never run on NPU regardless of torch_npu support. This also blocks upstream features layered on Flex (document masks via `create_block_mask`, attention sinks).
- Ask: allow privateuse1 devices whose backend registers a flex implementation (or lower to the math path), instead of raising on device type.

## TORCH-2 `fake` process-group backend cannot be extended by out-of-tree backends — `question`
- `Backend.backend_capability["fake"]` is a fixed list. If torch_npu cannot append to it cleanly (see torch_npu NPU-2), a registration API is needed here.

## TORCH-3 `torch.distributed.set_timeout` public API only in nightly — `info`
- Added ~mid-2026 as the public name of `_set_pg_timeout`. Downstream (torchtitan) adopted it immediately; released torch ≤ 2.13 lacks it. Polyfilled in `ascend_titan/compat/shims/dist_set_timeout.py` (auto no-op once torch ships it).
- Caveat: on torch ≤ 2.13 `_set_pg_timeout` handles only nccl/gloo and warns `Set timeout is now only supported for either nccl or gloo` — the HCCL group's timeout is left unchanged. Nightly's `ProcessGroup.set_timeout` forwards to every backend, so on nightly torch + a torch_npu that implements it the behaviour would match CUDA.
