"""Numerics of AscendFusionAttention vs an fp32 per-document SDPA reference."""

import pytest
import torch

pytestmark = [pytest.mark.npu, pytest.mark.titan]


def _reference(q, k, v, cu, hq, hkv):
    import torch.nn.functional as F

    outs, s = [], 0
    for e in cu[1:]:
        qq = q[s:e].float().transpose(0, 1)
        kk = k[s:e].float().transpose(0, 1).repeat_interleave(hq // hkv, 0)
        vv = v[s:e].float().transpose(0, 1).repeat_interleave(hq // hkv, 0)
        outs.append(F.scaled_dot_product_attention(qq, kk, vv, is_causal=True).transpose(0, 1))
        s = e
    return torch.cat(outs)


@pytest.mark.parametrize("hq,hkv", [(8, 8), (8, 2)])
def test_fwd_bwd_matches_reference(hq, hkv):
    from torchtitan.models.common.attention import VarlenMetadata

    from ascend_titan.kernels.attention import AscendFusionAttention

    torch.manual_seed(0)
    T, D = 160, 64
    dev = torch.device("npu:0")
    q = torch.randn(T, hq, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(T, hkv, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(T, hkv, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, 48, 100, 160], device=dev, dtype=torch.int32)
    meta = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=60, max_k=60)

    attn = AscendFusionAttention.Config(window_size=(-1, 0)).build()
    out = attn(q, k, v, attention_masks=meta, enable_gqa=hq != hkv)
    ref = _reference(q.detach(), k.detach(), v.detach(), cu.tolist(), hq, hkv)
    assert out.shape == (T, hq, D)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    out.float().sum().backward()
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    _reference(q2, k2, v2, cu.tolist(), hq, hkv).sum().backward()
    for a, b in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
        torch.testing.assert_close(a.float(), b.float(), atol=5e-2, rtol=5e-2)
