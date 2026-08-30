"""Llama 3 recipes -- the zero-override reference path.

validated: torchtitan=13da2d77c torch=2.15.0.dev20260812 torch_npu=2.15.0 CANN=9.1.0 date=2026-08-30

This is the strongest statement the project can make: **not a single
ascend_titan override**, and upstream llama3 trains on Ascend. Everything is the
upstream default -- ComplexRoPE (complex cache indexing, needs the NPU-3 fix),
ChunkedLossWrapper (TT-4), spmd_types backend, and stock ``VarlenAttention`` ->
``aten::_flash_attention_forward`` (needs the NPU-1 fix). The only delta is
inner attention flex -> varlen, because model-level flex compiles through
inductor and that needs Triton-Ascend (DEP-INDUCTOR).

If one of these recipes goes 🔴, an Ascend-side regression landed: the fixes in
``patches/`` are exactly what makes the stock path work.

Full guide: ascend_titan/models/llama3/README.md
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
