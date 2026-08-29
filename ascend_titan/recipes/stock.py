"""Stock upstream configs on NPU: no ascend_titan override at all (matrix rows "stock").

llama3 debugmodel, fully upstream:
Only delta: inner attention flex -> varlen (flex needs inductor/Triton-Ascend). Everything else is
upstream default: ComplexRoPE (complex cache indexing), ChunkedLossWrapper, spmd_types backend,
stock VarlenAttention -> aten::_flash_attention_forward (torch_npu PrivateUse1 kernel).
"""

from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.llama3.config_registry import llama3_debugmodel
from torchtitan.trainer import Trainer


def llama3_debugmodel_stock_npu() -> Trainer.Config:
    config = llama3_debugmodel()
    for _fqn, _cfg, parent, attr in list(config.traverse(FlexAttention.Config)):
        if isinstance(parent, list):
            parent[attr] = VarlenAttention.Config()
        else:
            setattr(parent, attr, VarlenAttention.Config())
    config.override.imports = []
    config.checkpoint.enable = False
    return config


def llama3_debugmodel_stock_npu_fsdp2() -> Trainer.Config:
    config = llama3_debugmodel_stock_npu()
    config.parallelism.data_parallel_shard_degree = 2
    return config
