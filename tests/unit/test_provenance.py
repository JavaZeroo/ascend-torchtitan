import pytest

pytestmark = pytest.mark.titan


def test_provenance_marks_ascend_nodes(monkeypatch):
    import sys
    import types

    from torchtitan.config.override import clear_overrides

    from ascend_titan.tools import provenance

    fake = types.ModuleType("torch_npu")
    fake.npu_fusion_attention = fake.npu_rms_norm = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    for m in ("ascend_titan.kernels.attention", "ascend_titan.kernels.rms_norm"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    clear_overrides()
    cfg = provenance.build_config("ascend_titan.models.qwen3", "qwen3_debugmodel_npu_fused_norm")
    rows = provenance.collect(cfg)
    summary = provenance.summarize(rows)
    ascend = [c for c, v in summary.items() if v["origin"] == "ascend"]
    assert any("AscendFusionAttention" in c for c in ascend)
    assert any("AscendRMSNorm" in c for c in ascend)
    assert "ascend-backed nodes" in provenance.render(summary)
    clear_overrides()
