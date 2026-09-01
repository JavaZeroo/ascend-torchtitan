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
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.config_registry import (
    qwen3_0_6b,
    qwen3_14b,
    qwen3_debugmodel,
)
from torchtitan.trainer import Trainer

from ascend_titan.kernels import (
    ATTENTION_OVERRIDE,
    RMSNORM_OVERRIDE,
    ROPE_COSSIN_OVERRIDE,
    SWIGLU_OVERRIDE,
)


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


def qwen3_8b_npu_pp2() -> Trainer.Config:
    """Qwen3-8B × (PP 2 × FSDP2 4)，8 卡 —— 流水并行的证据。

    为什么不是 0.6B：0.6B / 1.7B / 4B（连 debugmodel 也是）都 tie 了 embedding 与
    lm_head，而上游明确 ``Weight tying is not supported with Pipeline Parallel``。
    8B 是第一个不 tie 的尺寸。这是上游的限制，与昇腾无关。

    为什么不是 14B：14B 在 8×910B2 上装不下——`FullAC` + 1×4096 微批仍然 OOM
    （54.81 GiB 已分配时再要 1.60 GiB 失败）。参数、梯度与 AdamW 状态本身就占掉
    绝大部分，不是激活能省出来的。
    """
    from ascend_titan.models.assets import hf_assets_path, local_c4_dataset

    # 上游没有裸的 8B Trainer 配置（只有 nvfp4 变体），所以从 14B 的配置起手换模型。
    config = qwen3_14b()
    config.model_spec = model_registry("8B", attn_backend="varlen")
    config.override.imports = [ATTENTION_OVERRIDE]
    config.hf_assets_path = hf_assets_path("Qwen3-8B")
    config.dataloader.dataset = ConcatThenSplitPackingConfig(dataset=local_c4_dataset())
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 4
    # 微批数必须 >= 流水级数，否则流水线排不出来（上游默认 1）。
    config.parallelism.num_pp_microbatches = 2
    # 上游的配置校验禁止 CUDA graphs 与 PP 并存（"CUDA graphs do not support pipeline
    # parallelism yet"）。这个开关在昇腾上本来就不起作用——它门控的是 torch.cuda.CUDAGraph
    # ——但校验是配置级的，与设备无关，所以 PP recipe 必须显式关掉。
    config.training.disable_cuda_graphs = True
    config.activation_checkpoint = FullAC.Config()
    config.training.num_tokens_per_microbatch_per_dp_rank = 2 * 4096
    return config
