import pytest

from ascend_titan.tools.matrix import parse_cards, triage


def test_triage_priority_and_unknown():
    assert (
        triage("blah\nRuntimeError: No backend type associated with device type npu")[0] == "NPU-2"
    )
    assert (
        triage("NotImplementedError: Could not run 'aten::_flash_attention_forward'")[0] == "NPU-1"
    )
    assert (
        triage("ValueError: FlexAttention is only supported on CUDA, CPU or HPU devices.")[0]
        == "TORCH-1"
    )
    assert triage('  File "/x/torch_npu/foo.py", line 1\nRuntimeError: boom')[0] == "NPU"
    assert triage("something\n__TIMEOUT__\n")[0] == "HANG"
    assert triage("ERR99999 UNKNOWN applicaiton exception\nRuntimeError: x")[0] != "CANN"
    assert triage("[ERROR] EZ9999 op failed")[0] == "CANN"
    code, note = triage("NPU function error: call aclnnIndex failed, error code is 161002")
    assert code == "NPU-OP" and "aclnnIndex" in note and "161002" in note
    code, note = triage(
        "[rank0]:[rank0]: ZeroDivisionError: division by zero\nChildFailedError: \n"
    )
    assert code == "UNKNOWN" and note.startswith("ZeroDivisionError")


def test_parse_cards():
    assert parse_cards("0-3,6") == [0, 1, 2, 3, 6]


def _expected_spmd_backend() -> str:
    """npu_baseline only forces partial_dtensor on a torch whose FSDP2 cannot read
    spmd_types annotations (torch <= 2.13); nightly keeps the upstream default."""
    from ascend_titan.recipes.transforms import _torch_fsdp_reads_spmd_types

    return "spmd_types" if _torch_fsdp_reads_spmd_types() else "partial_dtensor"


@pytest.mark.titan
def test_npu_baseline_transform_on_upstream_recipes():
    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.models.common.attention import FlexAttention, VarlenAttention
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    from ascend_titan.recipes.transforms import ATTENTION_OVERRIDE, npu_baseline

    cfg = llama3_debugmodel()
    assert any(True for _ in cfg.traverse(FlexAttention.Config))
    a = npu_baseline(cfg)
    assert a.flex_to_varlen > 0
    assert not any(True for _ in cfg.traverse(FlexAttention.Config))
    assert any(True for _ in cfg.traverse(VarlenAttention.Config))
    assert ATTENTION_OVERRIDE in cfg.override.imports
    assert "ascend_titan.kernels.rope.real_cache_rope" in cfg.override.imports
    assert "ascend_titan.kernels.rms_norm.npu_rms_norm" in cfg.override.imports
    assert cfg.parallelism.spmd_backend == _expected_spmd_backend()
    # TT-4 is gone on the NIGHTLY track; the upstream default loss wrapper stays in place.
    assert isinstance(cfg.loss, ChunkedLossWrapper.Config)
    # idempotent
    b = npu_baseline(cfg)
    assert b.flex_to_varlen == 0 and cfg.override.imports.count(ATTENTION_OVERRIDE) == 1


@pytest.mark.titan
def test_matrix_module_resolves_upstream_recipe():
    import ascend_titan.recipes.matrix as m

    fn = getattr(m, "torchtitan.models.llama3.config_registry__llama3_debugmodel")
    cfg = fn()
    assert cfg.parallelism.spmd_backend == _expected_spmd_backend()
    stock = getattr(m, "torchtitan.models.llama3.config_registry__llama3_debugmodel__stock")()
    assert stock.override.imports == []
    with pytest.raises(AttributeError):
        _ = m.nonsense


@pytest.mark.titan
def test_npu_baseline_skips_rope_when_upstream_block_override_present():
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    from ascend_titan.recipes.transforms import ROPE_OVERRIDE, npu_baseline

    cfg = llama3_debugmodel()
    cfg.override.imports = ["torchtitan.overrides.fused_mla.fused_mla"]
    a = npu_baseline(cfg)
    assert ROPE_OVERRIDE not in cfg.override.imports and not a.rope_override
