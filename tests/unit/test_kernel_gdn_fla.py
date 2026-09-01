import importlib
import sys
import types

import pytest

pytest.importorskip("torch")
pytest.importorskip("attn_gym")
# ascend_titan.kernels.gdn_fla imports upstream qwen3_5, whose module-level import fla is CUDA-only.
pytest.importorskip("fla", reason="pip install fla-core (qwen3_5 extra)")


def _block_fla_npu(monkeypatch):
    monkeypatch.setitem(sys.modules, "fla_npu", None)


@pytest.mark.titan
def test_import_safe_without_fla_npu(npu_stub, monkeypatch):
    """fla-npu (AscendC wheel) is an optional add-on (ADR-004): no hard fail."""
    _block_fla_npu(monkeypatch)
    mod = importlib.import_module("ascend_titan.kernels.gdn_fla")
    assert mod._AVAILABLE is False
    assert not hasattr(mod, "npu_gated_delta_net_fused")
    assert not hasattr(mod, "fused_chunk_gdn")


@pytest.mark.titan
def test_fused_shape_gate_bounds(npu_stub, monkeypatch):
    """The gate is the single authority for which shapes may enter the fused op."""
    _block_fla_npu(monkeypatch)
    import torch

    from ascend_titan.kernels.gdn_fla import _fused_shape_gate

    def tens(B, H, T, K, V, dtype=torch.bfloat16):
        q = torch.empty(B, H, T, K, dtype=dtype)
        k = torch.empty(B, H, T, K, dtype=dtype)
        v = torch.empty(B, H, T, V, dtype=dtype)
        return q, k, v

    q, k, v = tens(1, 16, 128, 128, 128)
    assert _fused_shape_gate(q, k, v, 64) is True
    assert _fused_shape_gate(q, k, v, 128) is True
    assert _fused_shape_gate(*tens(1, 16, 128, 128, 64), 64) is False
    assert _fused_shape_gate(*tens(1, 8, 128, 256, 128), 64) is False
    assert _fused_shape_gate(q, k, v, 256) is False
    assert _fused_shape_gate(*tens(1, 16, 128, 128, 128, dtype=torch.float32), 64) is False
    q2 = torch.empty(1, 8, 128, 128, dtype=torch.bfloat16)
    assert _fused_shape_gate(q2, k, v, 64) is False


@pytest.mark.titan
def test_gate_rejects_missing_required_ops(npu_stub, monkeypatch):
    """Calling the op without all 11 ascendc entries raises cleanly."""
    _block_fla_npu(monkeypatch)
    import ascend_titan.kernels.gdn_fla as m

    fake = types.ModuleType("fla_npu")
    ops = types.ModuleType("fla_npu.ops")
    ascendc = types.ModuleType("fla_npu.ops.ascendc")
    ops.ascendc = ascendc
    fake.ops = ops
    for name in m._REQUIRED_ASCENDC_OPS:
        setattr(ascendc, name, lambda *a, **k: None)
    del ascendc.solve_tri
    monkeypatch.setitem(sys.modules, "fla_npu", fake)
    m._fla_npu = fake
    with pytest.raises(RuntimeError, match="missing entries"):
        m._ascendc()
