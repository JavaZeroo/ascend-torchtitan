import torch
import torch_npu  # noqa
from ascend_titan.kernels.gdn_fla import (
    AscendFusedGatedDeltaKernel, AscendFusedInnerGatedDeltaNet,
    npu_gated_delta_net_fused,
)
from ascend_titan.kernels.gdn import AscendGatedDeltaKernel

# Simulate what apply_overrides does: pass an InnerGatedDeltaNet.Config through the factory
from torchtitan.models.qwen3_5.gdn import InnerGatedDeltaNet
from ascend_titan.kernels import gdn as plain

# build a minimal InnerGatedDeltaNet.Config the way the override target declares it
cfg = AscendFusedInnerGatedDeltaNet.Config(
    kernel=AscendFusedGatedDeltaKernel.Config(chunk_size=64),
    # fill required fields with defaults by constructing via npu factory on a base
)
print("kernel config _owner:", AscendFusedGatedDeltaKernel.Config._owner.__name__)
print("kernel instance:", AscendFusedGatedDeltaKernel.Config(chunk_size=64).build().__class__.__name__)
