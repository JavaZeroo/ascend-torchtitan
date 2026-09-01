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

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import model_registry
from torchtitan.trainer import Trainer

from ascend_titan.models.qwen3.recipes import RMSNORM_OVERRIDE, qwen3_debugmodel_npu


def qwen3_debugmodel_stock_flex() -> Trainer.Config:
    """Cell attention/flex: upstream default inner attention, no override.

    Model-level flex compiles through inductor -> needs Triton-Ascend
    (910B2 cannot lower the document mask's indirect memory). Expected 🔴 on this chip.
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


def qwen3_debugmodel_npu_ce_loss() -> Trainer.Config:
    """Cell loss/chunked: 普通 ``CrossEntropyLoss`` 而不是上游默认的 ``ChunkedLossWrapper``。

    参考 recipe 用上游默认；这条探针让非 chunked 的那条路可测。
    """
    config = qwen3_debugmodel_npu()
    assert config.model_spec is not None
    config.loss = CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(config.model_spec))
    return config


def qwen3_debugmodel_npu_fused_norm() -> Trainer.Config:
    """Cell norm/npu_rms_norm: reference recipe + only the fused RMSNorm override.

    Isolates one kernel at a time; kept out of ``qwen3_debugmodel_npu`` so the
    frozen golden stays bit-exact.
    """
    config = qwen3_debugmodel_npu()
    config.override.imports = [*config.override.imports, RMSNORM_OVERRIDE]
    return config


def qwen3_debugmodel_npu_graph() -> Trainer.Config:
    """Cell graph/torchair: compile the loss with Ascend's GE graph backend.

    Only ``loss`` is compiled: the model contains our varlen attention custom op,
    which torchair cannot lower yet (OURS-13). Numerics move slightly (compiled
    reductions reassociate), so this has no golden of its own.
    """
    from ascend_titan.graph import npu_graph

    return npu_graph(qwen3_debugmodel_npu())
