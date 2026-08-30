"""torch_npu is a base dependency (P14), not an optional accelerator.

Every L1 module must fail *at import* when torch_npu -- or an op it needs -- is
missing. The old behaviour (import quietly, log a WARNING, skip the override)
turned a broken environment into an unnoticed eager run; that is exactly the
class of bug this test now forbids. ADR-004's "degrade loudly" survives only for
genuinely optional add-ons (ops-nn), covered at the bottom.
"""

import importlib

import pytest

# module -> the torch_npu op whose absence must break the import
MODULES = {
    "ascend_titan.kernels.attention": "npu_fusion_attention",
    "ascend_titan.kernels.rms_norm": "npu_rms_norm",
    "ascend_titan.kernels.swiglu": "npu_swiglu",
    "ascend_titan.kernels.rope": "npu_rotary_mul",
    "ascend_titan.kernels.situ_glu": None,  # needs torch_npu, but no op of its own
}


@pytest.mark.titan
@pytest.mark.parametrize("name", list(MODULES))
def test_import_without_torch_npu_raises(no_torch_npu, name):
    with pytest.raises(ImportError):
        importlib.import_module(name)


@pytest.mark.titan
@pytest.mark.parametrize("name,op", [(k, v) for k, v in MODULES.items() if v])
def test_import_with_incomplete_torch_npu_raises(npu_stub_missing_op, name, op):
    """A torch_npu without the op is an Ascend-side gap (P9), never a fallback."""
    npu_stub_missing_op(op)
    from ascend_titan.kernels._probe import MissingNpuOpError

    with pytest.raises(MissingNpuOpError, match=op):
        importlib.import_module(name)


@pytest.mark.titan
def test_optional_addon_still_degrades_loudly(npu_stub, no_ops_nn):
    """ops-nn needs its own run package + JIT build: optional, so ADR-004 applies."""
    mod = importlib.import_module("ascend_titan.kernels.situ_glu")
    assert mod._AVAILABLE is False
    assert not hasattr(mod, "ops_nn_situ_glu")
