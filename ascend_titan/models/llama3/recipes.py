"""Llama 3 recipes -- the zero-override reference path.

validated: torchtitan=13da2d77c torch=2.15.0.dev20260812 torch_npu=2.15.0 CANN=9.1.0 date=2026-08-30

This is the strongest statement the project can make: **not a single
ascend_titan override**, and upstream llama3 trains on Ascend. Everything is the
upstream default -- ComplexRoPE (complex cache indexing, needs the NPU-3 fix),
ChunkedLossWrapper (TT-4), spmd_types backend, and stock ``VarlenAttention`` ->
``aten::_flash_attention_forward`` (needs the NPU-1 fix). The only delta is
inner attention flex -> varlen, because model-level flex compiles through
inductor, and 910B2 cannot lower the document mask it builds (Ascend950 only).

If one of these recipes goes 🔴, an Ascend-side regression landed: the fixes in
``patches/`` are exactly what makes the stock path work.

Full guide: ascend_titan/models/llama3/README.md
"""

from torchtitan.models.llama3.config_registry import llama3_debugmodel
from torchtitan.trainer import Trainer

from ascend_titan.recipes.deltas import flex_to_varlen


def npu_deltas(config: Trainer.Config, flavor: str = "") -> None:
    """What Llama 3 needs on Ascend: one node conversion and **zero overrides**.

    Flavor-independent, so ``models/llama3/__init__.py`` can hand it to
    ``_auto.npu_entry_points`` and every upstream flavor gets an entry point.
    Keeping this list at exactly one line is the point of the model (see the
    module docstring): everything else is upstream's own implementation.
    """
    # DELTA 1: flex -> varlen; model-level flex needs to compile a document mask,
    # which 910B2 cannot lower (Ascend950 only). Feature-detected: once that lands
    # here this becomes a no-op and llama3 is byte-for-byte stock upstream.
    flex_to_varlen(config)


def llama3_debugmodel_stock_npu() -> Trainer.Config:
    """Stock upstream llama3 on Ascend, zero overrides. Golden-frozen."""
    config = llama3_debugmodel()
    npu_deltas(config)
    assert config.override.imports == [], "llama3 is the zero-override reference path"
    config.checkpoint.enable = False
    return config
