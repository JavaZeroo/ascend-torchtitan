# torch_npu — problems (owner: Ascend/pytorch, gitee.com/ascend/pytorch)

Measured 2026-08-29 on torch 2.12.0 / torch_npu 2.12.0 **and** torch 2.13.0 / torch_npu 2.13.0rc1, CANN 9.1.0, Ascend 910B2. None of these are worked around in this repo (P1).

## NPU-1 `aten::_flash_attention_forward` has no PrivateUse1 kernel  — `draft`
- Symptom: `torch.nn.attention.varlen.varlen_attn` → `NotImplementedError: Could not run 'aten::_flash_attention_forward' ... PrivateUse1` (the op is dispatched through the custom-op `torch_attn::_varlen_attn`, which then calls the aten op).
- Check: `torch._C._dispatch_has_kernel_for_dispatch_key("aten::_flash_attention_forward", "PrivateUse1")` → `False` on 2.12.0 and 2.13.0rc1.
- Impact: torchtitan's `VarlenAttention` (the only non-Flex LM attention upstream) cannot run. ascend-torchtitan ships an override (`ascend_titan.kernels.attention`) on top of `npu_fusion_attention` instead — a kernel replacement, not a workaround of this issue.
- Ask: implement `_flash_attention_forward/_backward` (varlen signature with `cum_seq_q/k`, `max_q/k`, `window_size`) for NPU, e.g. by lowering to `npu_fusion_attention` TND. This would make stock torchtitan varlen work unmodified.

## NPU-2 `npu` is not registered for the `fake` process-group backend — `draft`
- Symptom: `--comm.mode=fake_backend` (torchtitan's single-device dry run) → `RuntimeError: No backend type associated with device type npu` from `torch.distributed.broadcast` / functional `all_reduce`.
- Check: `torch.distributed.Backend.backend_capability["fake"]` → `['cpu', 'cuda', 'hpu', 'xpu']` (same list in torch nightly 2.15.0.dev20260828). HPU added itself; torch_npu should too.
- Impact: every fake-PG based tool (config dry runs, upstream `fake_pg` integration suite, pipeline dry runs) is unusable on NPU.
- Ask: on `import torch_npu`, register `"npu"` in `Backend.backend_capability["fake"]` (and the functional-collective device mapping) so the fake backend accepts npu tensors.

## NPU-3 `aclnnIndex` fails (error 161002) in MoE token dispatch — `draft`
- `RuntimeError: index_high_dims_op_api: .../op_plugin/ops/opapi/IndexKernelNpuOpApi.cpp:40 NPU function error: call aclnnIndex failed, error code is 161002` on `deepseek_v3` debugmodel with EP (upstream `deepseek_v3_fsdp+ep` case, `npu_baseline` applied). Torch 2.12.0 / torch_npu 2.12.0.
- Impact: every MoE model with expert parallel (deepseek_v3, qwen3-MoE, gpt_oss, kimi) is blocked at the same op. Needs a minimal repro of the advanced-indexing pattern used by `token_dispatcher`.

## NPU-4 `ArgSort` int32/int64 falls back to AiCpu (perf warning) — `note`
- `Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu` during `torch.distributed`/mesh setup. Not a correctness issue; recorded so it is not re-investigated.

## NPU-5 `torch_npu==2.13.0rc1` pins `attn-gym` ≥0.0.6 transitively? — `check`
- Installing torch_npu 2.13.0rc1 pulled `attn-gym 0.0.6`, conflicting with torchtitan's `==0.0.5`. Needs confirmation of which package requested it; pinned back to 0.0.5 in `constraints/titan-deps.txt`.
