"""L1: fused-kernel overrides (torchtitan ``@override`` factories).

Modules here are imported by torchtitan's override mechanism. The only import-time
side effect allowed is an optional-addon probe that logs a loud WARNING when the
accelerator package is absent (ADR-004); no module may load an NPU op or reach
fla_npu.ops.ascendc at import. See .claude/skills/override-authoring.

The ``*_OVERRIDE`` constants below are the ``override.imports`` target paths for
every factory this package ships -- the single place that spells them out (P11).
Recipes and transforms import them from here; ``tests/unit/test_recipes.py``
checks each one resolves to a real ``def`` in this package.
"""

ATTENTION_OVERRIDE = "ascend_titan.kernels.attention.npu_fusion_attention"
ROPE_OVERRIDE = "ascend_titan.kernels.rope.real_cache_rope"
ROPE_COSSIN_OVERRIDE = "ascend_titan.kernels.rope.npu_rotary_cossin"
RMSNORM_OVERRIDE = "ascend_titan.kernels.rms_norm.npu_rms_norm"
SWIGLU_OVERRIDE = "ascend_titan.kernels.swiglu.npu_fused_swiglu"
SITU_GLU_OVERRIDE = "ascend_titan.kernels.situ_glu.ops_nn_situ_glu"
GDN_OVERRIDE = "ascend_titan.kernels.gdn.npu_gated_delta_net"
GDN_FUSED_OVERRIDE = "ascend_titan.kernels.gdn_fla.npu_gated_delta_net_fused"
KDA_OVERRIDE = "ascend_titan.kernels.kda.npu_kda"
