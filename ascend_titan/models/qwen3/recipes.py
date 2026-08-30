"""Qwen3 recipes -- the Ascend reference path.

validated: torchtitan=13da2d77c torch=2.15.0.dev20260812 torch_npu=2.15.0 CANN=9.1.0 date=2026-08-30
validated: torchtitan=13da2d77c torch=2.13.0 torch_npu=2.13.0rc1 CANN=9.1.0 date=2026-08-29
validated: torchtitan=13da2d77c torch=2.12.0 torch_npu=2.12.0 CANN=9.1.0 date=2026-08-29
(rewritten by CI when a run is green; golden curves in tests/assets/losses/npu/)

Naming: ``qwen3_<flavor>_npu[_<variant>]``. ``<flavor>`` is the upstream config
registry name (``debugmodel``, later ``0_6b`` ...), ``<variant>`` is a parallel
or kernel delta (``fsdp2``, ``fused``). Measurement-only configs -- stock
upstream, single-feature probes -- are NOT recipes and live in ``probes.py``.

The reference path is now two deltas away from stock upstream: the inner-attention
node plus its Ascend kernel, and "no checkpoint I/O in a smoke run". The loss
(``ChunkedLossWrapper``) and the spmd backend (``spmd_types``) are upstream's
defaults, supported, gated and golden-frozen.

Why the attention delta exists: upstream language models offer only ``flex`` and
``varlen`` inner attention (``sdpa`` was removed, ``config_utils.py:97``).
``flex`` needs inductor (Triton-Ascend, DEP-INDUCTOR) at the model level, and
stock ``varlen`` needs ``aten::_flash_attention_forward``, which torch_npu only
grows with the NPU-1 fix. The ``kernels.attention`` override is therefore the
supported path, and ``probes.py`` keeps both stock cells measurable.

Full guide: ascend_titan/models/qwen3/README.md
"""

from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.config_registry import qwen3_0_6b, qwen3_debugmodel
from torchtitan.trainer import Trainer

ATTENTION_OVERRIDE = "ascend_titan.kernels.attention.npu_fusion_attention"
RMSNORM_OVERRIDE = "ascend_titan.kernels.rms_norm.npu_rms_norm"
SWIGLU_OVERRIDE = "ascend_titan.kernels.swiglu.npu_fused_swiglu"
ROPE_COSSIN_OVERRIDE = "ascend_titan.kernels.rope.npu_rotary_cossin"


def qwen3_debugmodel_npu() -> Trainer.Config:
    """Upstream ``qwen3_debugmodel`` + the minimal deltas for an NPU smoke run.
    Each delta is a matrix cell, tracked in docs/capability-matrix.md."""
    config = qwen3_debugmodel()

    # DELTA 1: inner attention = varlen node + Ascend fused-attention override.
    # (matrix: attention/ascend_fusion)
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    config.override.imports = [ATTENTION_OVERRIDE]

    # DELTA 2: no checkpoint I/O in the smoke run (DCP on NPU is its own cell).
    config.checkpoint.enable = False

    # The spmd backend stays upstream's ``spmd_types``. It used to be forced to
    # "partial_dtensor" because FSDP2 only builds DTensor shards from spmd_types
    # annotations on torch >= 2.14.0.dev (TT-5 / TORCH-6) -- again a torch version
    # gap, not an NPU one, so on NIGHTLY the delta only made the reference path
    # differ from upstream (P8/P12). ``probes.qwen3_debugmodel_npu_partial_dtensor``
    # keeps the alternative backend measurable. (matrix: parallel/spmd_types)

    # The loss stays upstream's ChunkedLossWrapper. It used to be swapped for a
    # plain CrossEntropyLoss because the chunked loss (upstream #4143) drives
    # FSDP's lm_head unshard by hand and its backward hit "data is not allocated
    # yet" on release torch + NPU (TT-4) -- a torch version gap that does not
    # exist on NIGHTLY, so carrying the delta only made the reference path differ
    # from upstream for no reason (P8/P12). ``probes.qwen3_debugmodel_npu_ce_loss``
    # keeps the non-chunked path measurable. (matrix: loss/chunked)

    return config


def qwen3_debugmodel_npu_fsdp2() -> Trainer.Config:
    """M1 acceptance path (3): real multi-device FSDP2."""
    config = qwen3_debugmodel_npu()
    # DELTA 3: 2-way FSDP.
    config.parallelism.data_parallel_shard_degree = 2
    return config


def qwen3_debugmodel_npu_fused() -> Trainer.Config:
    """All zero-build torch_npu fused kernels: RMSNorm + fused SwiGLU + rotary kernel.
    The perf recipe; numerics differ from golden at bf16-rounding level."""
    config = qwen3_debugmodel_npu()
    config.override.imports = [
        *config.override.imports,
        RMSNORM_OVERRIDE,
        SWIGLU_OVERRIDE,
        ROPE_COSSIN_OVERRIDE,
    ]
    return config


def qwen3_debugmodel_npu_fused_fsdp2() -> Trainer.Config:
    """Fused perf recipe under FSDP2 x2 (golden-tracked)."""
    config = qwen3_debugmodel_npu_fused()
    config.parallelism.data_parallel_shard_degree = 2
    return config


# --- release 级 recipe（docs/model-release-criteria.md R1）---------------------
# debugmodel 是冒烟件：玩具 tokenizer、几百条样本、2048 上下文。下面的 recipe 用真实
# tokenizer、真实 C4 分片和该尺寸的真实上下文长度，是"这个模型在昇腾上能用"的证据。


def qwen3_0_6b_npu() -> Trainer.Config:
    """Qwen3-0.6B，真实 tokenizer + 真实 C4。单卡或 FSDP2 都跑得动。"""
    from ascend_titan.models.assets import hf_assets_path, local_c4_dataset

    config = qwen3_0_6b()

    # DELTA 1: 与 debugmodel 相同的注意力增量（同样的理由，见文件头）。
    config.model_spec = model_registry("0.6B", attn_backend="varlen")
    config.override.imports = [ATTENTION_OVERRIDE]

    # DELTA 2: 资产落到本地（上游默认路径是 torchtitan 检出里的 ./assets/hf/...，
    # 那是固定上游，我们不往里写东西）。
    config.hf_assets_path = hf_assets_path("Qwen3-0.6B")
    config.dataloader.dataset = ConcatThenSplitPackingConfig(dataset=local_c4_dataset())

    return config


def qwen3_0_6b_npu_fsdp2() -> Trainer.Config:
    """0.6B × FSDP2 8 卡。"""
    config = qwen3_0_6b_npu()
    config.parallelism.data_parallel_shard_degree = 8
    return config


def qwen3_0_6b_npu_tp2() -> Trainer.Config:
    """0.6B × (FSDP2 4 × TP 2)，8 卡。"""
    config = qwen3_0_6b_npu()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    return config


def qwen3_0_6b_npu_pp2() -> Trainer.Config:
    """0.6B × (PP 2 × FSDP2 4)，8 卡。"""
    config = qwen3_0_6b_npu()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 4
    return config
