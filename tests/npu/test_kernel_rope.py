import pytest
import torch

pytestmark = [pytest.mark.npu, pytest.mark.titan]


def test_rope_matches_cpu_reference_on_npu():
    from torchtitan.models.common.rope import ComplexRoPE

    from ascend_titan.kernels.rope import real_cache_rope

    cfg = ComplexRoPE.Config(dim=64, max_context_length=128, scaling="llama")
    ref = ComplexRoPE.Config(dim=64, max_context_length=128, scaling="llama").build()  # CPU
    dev = torch.device("npu:0")
    with torch.device(dev):
        new = real_cache_rope(cfg).build()
    torch.manual_seed(0)
    q = torch.randn(100, 8, 64, dtype=torch.bfloat16)
    k = torch.randn(100, 2, 64, dtype=torch.bfloat16)
    pos = torch.cat([torch.arange(60), torch.arange(40)])
    q1, k1 = ref(q, k, pos)
    q2, k2 = new(q.to(dev), k.to(dev), pos.to(dev))
    torch.testing.assert_close(q1, q2.cpu(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k1, k2.cpu(), atol=2e-2, rtol=2e-2)
