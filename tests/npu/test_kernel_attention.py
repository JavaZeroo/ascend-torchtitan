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


def test_custom_op_opcheck_and_compile():
    """opcheck validates fake/autograd registration; compile must not graph-break."""
    from ascend_titan.kernels import attention as A

    dev = torch.device("npu:0")
    torch.manual_seed(0)
    T, hq, hkv, D = 96, 4, 2, 32
    q = torch.randn(T, hq, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(T, hkv, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(T, hkv, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, 40, 96], device=dev, dtype=torch.int32)
    # torch.testing.opcheck's autograd-registration check is CPU/CUDA/XPU-only (TORCH-7);
    # the autograd path is covered by the gradient comparison above.
    torch.library.opcheck(
        A._fa_fwd,
        (q, k, v, cu, cu, D**-0.5, 3, A._INT_MAX),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )

    # aot_eager: proves dynamo + AOTAutograd trace the op without graph breaks; the
    # inductor backend needs Triton-Ascend / torchair on NPU (matrix: compile/inductor).
    fn = torch.compile(
        lambda a, b, c: A.fusion_attention_varlen(a, b, c, cu, cu, scale=D**-0.5),
        fullgraph=True,
        backend="aot_eager",
    )
    out = fn(q, k, v)
    ref = A.fusion_attention_varlen(q, k, v, cu, cu, scale=D**-0.5)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_lse_matches_reference():
    """LSE reconstructed from softmax_max/softmax_sum equals logsumexp of scaled scores."""
    from ascend_titan.kernels import attention as A

    dev = torch.device("npu:0")
    torch.manual_seed(0)
    T, H, D = 64, 2, 32
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0, 24, 64], device=dev, dtype=torch.int32)
    _, lse = A.fusion_attention_varlen(q, k, v, cu, cu, scale=D**-0.5, return_lse=True)
    ref = []
    for s, e in ((0, 24), (24, 64)):
        sc = torch.einsum("thd,shd->hts", q[s:e].float(), k[s:e].float()) * D**-0.5
        mask = torch.triu(torch.ones(e - s, e - s, device=dev, dtype=torch.bool), 1)
        sc = sc.masked_fill(mask, float("-inf"))
        ref.append(torch.logsumexp(sc, dim=-1).transpose(0, 1))  # (t, h)
    ref = torch.cat(ref)
    torch.testing.assert_close(lse.float(), ref, atol=3e-2, rtol=3e-2)
