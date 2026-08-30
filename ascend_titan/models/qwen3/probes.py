"""Measurement-only Qwen3 configs. NOT for training.

Every function here exists to keep one cell of ``docs/capability-matrix.md``
measurable: it isolates a single upstream feature so a red cell can be blamed on
that feature and nothing else. They are deliberately kept out of ``recipes.py``
so that "what we support" and "what we measure" never get mixed up -- a recipe
is something you are meant to run, a probe is an experiment whose expected
result may well be 🔴.

Run one exactly like a recipe::

    MODULE=ascend_titan.models.qwen3.probes CONFIG=qwen3_debugmodel_stock_varlen \\
        NPU=1 ./scripts/run_train.sh
"""

from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.config_registry import qwen3_debugmodel
from torchtitan.trainer import Trainer

from ascend_titan.models.qwen3.recipes import RMSNORM_OVERRIDE, qwen3_debugmodel_npu


def qwen3_debugmodel_stock_flex() -> Trainer.Config:
    """Cell attention/flex: upstream default inner attention, no override.

    Model-level flex compiles through inductor -> needs Triton-Ascend
    (DEP-INDUCTOR). Expected 🔴 until M5.
    """
    config = qwen3_debugmodel_npu()
    config.model_spec = model_registry("debugmodel", attn_backend="flex")
    config.override.imports = []
    return config


def qwen3_debugmodel_stock_varlen() -> Trainer.Config:
    """Cell attention/varlen: upstream varlen kernel, no override.

    Measures ``aten::_flash_attention_forward`` on PrivateUse1: 🔴 on stock
    torch_npu, 🟢 with the NPU-1 fix (patches/torch_npu). This is the cell that
    tells us whether NPU-1 has landed.
    """
    config = qwen3_debugmodel_npu()
    config.override.imports = []
    return config


def qwen3_debugmodel_npu_chunked_loss() -> Trainer.Config:
    """Cell loss/chunked: upstream ``ChunkedLossWrapper`` (the upstream default).

    🔴 on release torch (TT-4), 🟢 on NIGHTLY. When ``recipes.py`` drops DELTA 4
    this probe becomes redundant and should be deleted.
    """
    config = qwen3_debugmodel_npu()
    config.loss = qwen3_debugmodel().loss
    return config


def qwen3_debugmodel_npu_fused_norm() -> Trainer.Config:
    """Cell norm/npu_rms_norm: reference recipe + only the fused RMSNorm override.

    Isolates one kernel at a time; kept out of ``qwen3_debugmodel_npu`` so the
    frozen golden stays bit-exact.
    """
    config = qwen3_debugmodel_npu()
    config.override.imports = [*config.override.imports, RMSNORM_OVERRIDE]
    return config
