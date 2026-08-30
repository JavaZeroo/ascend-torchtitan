# ruff: noqa: E402  (probe script: imports interleaved with prints on purpose)
"""Which of the repo's TORCH-*/TT-* gaps still exist on the installed torch? CPU only."""

import inspect
import re

import torch

print("torch", torch.__version__)
import torch.distributed as dist

print("TT-2/TORCH-3 set_timeout present:", hasattr(dist, "set_timeout"))
from torch.distributed.pipelining.schedules import PipelineScheduleMulti, PipelineScheduleSingle

print(
    "TT-8/TORCH-4 step(arg_mbs=) single/multi:",
    "arg_mbs" in inspect.signature(PipelineScheduleSingle.step).parameters,
    "arg_mbs" in inspect.signature(PipelineScheduleMulti.step).parameters,
)
print("TT-9 torch.Tag.inplace:", hasattr(torch.Tag, "inplace"))
import torch.nn.attention.varlen as varlen

src = inspect.getsource(varlen)
m = re.search(r"rng_state\w*\s*=\s*torch\.zeros\([^)]*dtype=torch\.(\w+)", src)
print("TORCH-8 varlen rng_state dtype:", m.group(1) if m else "not found")
from torch.testing._internal.optests import autograd_registration as ar

src = inspect.getsource(ar)
m = re.search(r"NYI devices other than ([^'\"]*)", src)
print("TORCH-7 opcheck autograd devices:", m.group(1) if m else "pattern not found")
print("   privateuse1 mentioned:", "privateuse1" in src.lower())
import torch.nn.attention.flex_attention as fa

src = inspect.getsource(fa)
m = re.search(r"only supported on ([^\"']*)", src)
print("TORCH-1 flex device whitelist:", m.group(0) if m else "pattern not found")
print("TORCH-2/NPU-2 fake backend devices:", dist.Backend.backend_capability.get("fake"))
import torch.distributed.fsdp._fully_shard._fsdp_param as fp

src = inspect.getsource(fp)
print(
    "TT-5/TORCH-6 fsdp_param mentions spmd_types:",
    "spmd_types" in src,
    "| _is_spmd_types_available:",
    hasattr(dist, "_is_spmd_types_available"),
)
print(
    "has_kernel _flash_attention_forward PrivateUse1:",
    torch._C._dispatch_has_kernel_for_dispatch_key("aten::_flash_attention_forward", "PrivateUse1"),
)
# Diagnostic probe: reporting *what is missing* is this file's job, so the
# try/except here is the ADR-007 exemption, not a fallback (P14).
try:
    import torch_npu

    print("torch_npu", torch_npu.__version__)
except Exception as e:
    print("torch_npu import:", type(e).__name__, str(e)[:200])
