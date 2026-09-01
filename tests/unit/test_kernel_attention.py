"""CPU tests for the Ascend attention override: everything except the kernel
call itself (which lives in tests/npu). Import-time dependency behaviour is
covered by test_kernel_import_safety.py (P14)."""

import importlib
import sys

import pytest


def _reload(monkeypatch):
    """Import ascend_titan.kernels.attention fresh, on top of the npu_stub fixture."""
    for name in ("ascend_titan.kernels.attention", "ascend_titan.kernels.rms_norm"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    mod = importlib.import_module("ascend_titan.kernels.attention")
    importlib.import_module("ascend_titan.kernels.rms_norm")
    return mod


@pytest.mark.titan
def test_override_registers_and_derives(monkeypatch, npu_stub):
    from torchtitan.config.override import clear_overrides
    from torchtitan.models.common.attention import VarlenAttention

    clear_overrides()
    mod = _reload(monkeypatch)
    cfg = VarlenAttention.Config(window_size=(-1, 0))
    new = mod.npu_fusion_attention(cfg)
    assert isinstance(new, mod.AscendFusionAttention.Config)
    assert isinstance(new, VarlenAttention.Config)  # decoder's isinstance dispatch still works
    assert new.window_size == (-1, 0)
    module = new.build()
    assert type(module).__name__ == "AscendFusionAttention"
    clear_overrides()


@pytest.mark.titan
def test_unsupported_window_rejected_at_build(monkeypatch, npu_stub):
    from torchtitan.config.override import clear_overrides

    clear_overrides()
    mod = _reload(monkeypatch)
    with pytest.raises(NotImplementedError, match="window_size"):
        mod.AscendFusionAttention.Config(window_size=(-1, -1)).build()
    clear_overrides()


@pytest.mark.titan
def test_recipe_activates_override_on_config_tree(monkeypatch, npu_stub):
    """apply_overrides must find our module and swap every VarlenAttention node."""
    from torchtitan.config.override import apply_overrides, clear_overrides

    from ascend_titan.models.qwen3 import qwen3_debugmodel_npu

    clear_overrides()
    _reload(monkeypatch)
    cfg = qwen3_debugmodel_npu()
    assert "ascend_titan.kernels.attention.npu_fusion_attention" in cfg.override.imports
    apply_overrides(cfg.override, cfg)
    inner = cfg.model_spec.model.layers[0].attention.inner_attention
    assert type(inner).__qualname__ == "AscendFusionAttention.Config"
    clear_overrides()


@pytest.mark.titan
def test_flex_entry_point_claims_decoder_layers_but_not_a_vision_tower(monkeypatch, npu_stub):
    """The CLI-addressable entry: a stock upstream config -> the Ascend kernel.

    The ``fqns`` glob is the whole safety story here. A vision tower's attention is
    fed a BlockMask this kernel cannot consume, and its nodes live outside
    ``layers.*.attention``, so the include-glob must leave them flex.
    """
    from torchtitan.config.override import apply_overrides, clear_overrides
    from torchtitan.models.common.attention import FlexAttention
    from torchtitan.models.qwen3.config_registry import qwen3_debugmodel

    from ascend_titan.kernels import ATTENTION_FROM_FLEX_OVERRIDE

    clear_overrides()
    _reload(monkeypatch)

    cfg = qwen3_debugmodel()  # stock upstream: flex everywhere, no overrides
    flex_before = [fqn for fqn, *_ in cfg.traverse(FlexAttention.Config)]
    assert flex_before, "upstream debugmodel should default to flex attention"

    cfg.override.imports = [ATTENTION_FROM_FLEX_OVERRIDE]
    applied = apply_overrides(cfg.override, cfg)
    assert len(applied) == len(flex_before)
    inner = cfg.model_spec.model.layers[0].attention.inner_attention
    assert type(inner).__qualname__ == "AscendFusionAttention.Config"

    # Every claimed node sits under a decoder layer; nothing else was touched.
    assert all(".attention.inner_attention" in line for line in applied)
    clear_overrides()
