"""The fused GDN recurrence on the 910B2, against the plain-torch oracle.

``tests/unit/test_kernel_gdn_fla.py`` covers the optional-addon degrade path on CPU.
Here fla_npu is present and the AscendC pipeline runs end-to-end: the custom op
must agree with ``ascend_titan.kernels.gdn.ascend_chunk_gdn`` (the same math in
plain torch) in forward and all five gradients, within bf16/fp16 tolerance, and
pass ``torch.library.opcheck``'s non-autograd checks.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla", reason="pip install fla-core (qwen3_5 extra)")
fla_npu = pytest.importorskip("fla_npu", reason="fla-npu AscendC wheel not installed")

pytestmark = pytest.mark.npu


def _inputs(batch, heads, sequence, key_dim, value_dim, dtype, device, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device)

    def unit(*shape):
        return torch.nn.functional.normalize(randn(*shape), dim=-1).to(dtype)

    return (
        unit(batch, heads, sequence, key_dim),
        unit(batch, heads, sequence, key_dim),
        randn(batch, heads, sequence, value_dim).to(dtype),
        -torch.rand(batch, heads, sequence, generator=gen).to(device),
        torch.rand(batch, heads, sequence, generator=gen).to(device),
    )


def test_opcheck_fused_chunk_gated_delta_rule():
    """Schema/faketensor/aot-dispatch checks. test_autograd_registration is
    skipped: torch.library.opcheck's autograd path rejects privateuse1 (TORCH-7),
    so gradient correctness is asserted numerically below instead.
    """
    import torch_npu  # noqa: F401  (registers the npu device)
    from torch.library import opcheck

    from ascend_titan.kernels.gdn_fla import _chunk_gdn_fwd

    q, k, v, g, beta = _inputs(1, 4, 128, 128, 128, torch.bfloat16, "npu:0")
    # Pass the CustomOpDef directly (as attention's opcheck does): torch 2.15
    # nightly's CustomOpDef has no .op attribute.
    opcheck(
        _chunk_gdn_fwd,
        (q, k, v, g.float(), beta.float(), 64),
        test_utils=["test_schema", "test_faketensor", "test_aot_dispatch_dynamic"],
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_matches_plain_torch_forward(dtype):
    import torch_npu  # noqa: F401

    from ascend_titan.kernels.gdn import ascend_chunk_gdn
    from ascend_titan.kernels.gdn_fla import fused_chunk_gdn

    q, k, v, g, beta = _inputs(1, 4, 512, 128, 128, dtype, "npu:0", seed=3)
    got = fused_chunk_gdn(q, k, v, g, beta, chunk_size=64)
    want = ascend_chunk_gdn(q, k, v, g, beta, chunk_size=64)
    torch.testing.assert_close(got, want, rtol=3e-2, atol=3e-2)


def test_fused_gradients_match_plain_torch():
    import torch_npu  # noqa: F401

    from ascend_titan.kernels.gdn import ascend_chunk_gdn
    from ascend_titan.kernels.gdn_fla import fused_chunk_gdn

    base = _inputs(1, 4, 256, 128, 128, torch.bfloat16, "npu:0", seed=7)

    def grads(fn):
        q, k, v, g, beta = [t.clone().requires_grad_() for t in base]
        out = fn(q, k, v, g, beta, chunk_size=64)
        out.square().sum().backward()
        return [q.grad, k.grad, v.grad, g.grad, beta.grad]

    for got, want in zip(grads(fused_chunk_gdn), grads(ascend_chunk_gdn), strict=True):
        torch.testing.assert_close(got, want, rtol=3e-2, atol=3e-2)
