import torch
import torch_npu  # noqa
from ascend_titan.kernels.gdn_fla import AscendFusedGatedDeltaKernel, AscendFusedInnerGatedDeltaNet
print("fused kernel Config owner:", AscendFusedGatedDeltaKernel.Config._owner)
print("is fused kernel:", AscendFusedGatedDeltaKernel.Config._owner is AscendFusedGatedDeltaKernel)
print("inner net Config owner:", AscendFusedInnerGatedDeltaNet.Config._owner)
print("inner kernel field type:", AscendFusedInnerGatedDeltaNet.Config.__dataclass_fields__["kernel"].type)
