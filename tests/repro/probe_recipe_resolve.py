import torch
import torch_npu  # noqa
import ascend_titan
r = ascend_titan.setup()
print("device", r.device_type, "fla_npu", r.fla_npu_imported)

from ascend_titan.models.qwen3_5.recipes import qwen35_0_8b_npu_fused
cfg = qwen35_0_8b_npu_fused()
print("override imports:", cfg.override.imports)

from ascend_titan.models.qwen3_5.recipes import qwen35_0_8b_npu
cfg2 = qwen35_0_8b_npu()
print("plain override imports:", cfg2.override.imports)

# resolve the gdn node config classes
from ascend_titan.kernels.gdn import AscendGatedDeltaKernel, AscendInnerGatedDeltaNet
from ascend_titan.kernels.gdn_fla import AscendFusedGatedDeltaKernel, AscendFusedInnerGatedDeltaNet
print("fused kernel cfg", AscendFusedGatedDeltaKernel.Config, "chunk_size default:", AscendFusedGatedDeltaKernel.Config(chunk_size=0).chunk_size)
