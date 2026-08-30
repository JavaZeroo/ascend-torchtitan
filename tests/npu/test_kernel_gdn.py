"""``ascend_chunk_gdn`` on the device it was written for.

``tests/unit/test_kernel_gdn.py`` checks the math on CPU. This one checks that
the same comparison holds on 910B2, where the matmuls run in a different order
and, for bf16, a different accumulation. Both compare against attn_gym's
reference recurrence -- the oracle, not the runtime path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("attn_gym")
pytest.importorskip("fla", reason="pip install fla-core (qwen3_5 extra)")

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


@pytest.mark.parametrize(("dtype", "tol"), [(torch.float32, 1e-3), (torch.bfloat16, 3e-2)])
def test_matches_attn_gym_reference_on_npu(dtype, tol):
    import torch_npu  # noqa: F401  (registers the npu device)
    from attn_gym.linear.gdn import chunk_gdn

    from ascend_titan.kernels.gdn import ascend_chunk_gdn

    args = _inputs(1, 4, 512, 64, 64, dtype, "npu:0")
    want = chunk_gdn(*args, impl="reference")
    want = want[0] if isinstance(want, tuple) else want
    got = ascend_chunk_gdn(*args)
    torch.testing.assert_close(got, want, rtol=tol, atol=tol)


def test_gradients_match_on_npu():
    import torch_npu  # noqa: F401
    from attn_gym.linear.gdn import chunk_gdn

    from ascend_titan.kernels.gdn import ascend_chunk_gdn

    base = _inputs(1, 4, 512, 64, 64, torch.float32, "npu:0", seed=5)

    def grads(fn):
        args = [t.clone().requires_grad_() for t in base]
        out = fn(*args)
        out = out[0] if isinstance(out, tuple) else out
        out.square().sum().backward()
        return [t.grad for t in args]

    for got, want in zip(
        grads(ascend_chunk_gdn),
        grads(lambda *a: chunk_gdn(*a, impl="reference")),
        strict=True,
    ):
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)
