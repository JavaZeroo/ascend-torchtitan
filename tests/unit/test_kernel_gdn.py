"""``ascend_chunk_gdn`` must agree with attn_gym's reference recurrence.

attn_gym's ``chunk_gdn(impl="reference")`` is the oracle: our version is the same
chunk decomposition with the intra-chunk forward-substitution loop replaced by a
closed-form inverse (see ``ascend_titan/kernels/gdn.py``). If these two ever
disagree, ours is wrong.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("attn_gym")
# ``ascend_titan.kernels.gdn`` imports upstream's qwen3_5 package, whose module-level
# ``import fla`` is what this override exists to route around.
pytest.importorskip("fla", reason="pip install fla-core (qwen3_5 extra)")


def _inputs(batch, heads, sequence, key_dim, value_dim, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype)

    def unit(*shape):
        return torch.nn.functional.normalize(randn(*shape).float(), dim=-1).to(dtype)

    return (
        # Q/K arrive L2-normalised: ``use_qk_l2norm_in_kernel=True`` upstream, and
        # ``_chunk_gdn_btnk`` does it explicitly. Without that the transition matrix
        # is not contractive and the recurrence is ill-conditioned for *any*
        # implementation, so an un-normalised comparison measures nothing.
        unit(batch, heads, sequence, key_dim),
        unit(batch, heads, sequence, key_dim),
        randn(batch, heads, sequence, value_dim),
        # log-decay is a negative log gate; beta is a (0, 1) write gate.
        -torch.rand(batch, heads, sequence, generator=gen).float(),
        torch.rand(batch, heads, sequence, generator=gen).float(),
    )


def test_unit_lower_inverse_matches_forward_substitution():
    from ascend_titan.kernels.gdn import _unit_lower_inverse

    size = 64
    gen = torch.Generator().manual_seed(1)
    # Scaled to the regime the kernel actually produces: the transition matrix is
    # built from L2-normalised keys times a decay < 1, so its entries are small.
    # With N(0, 1) entries the Neumann series itself is fine but fp32 cancellation
    # in *any* method swamps the comparison.
    lower = (torch.randn(3, size, size, generator=gen) * 0.1).tril(-1)
    got = _unit_lower_inverse(lower)
    want = torch.linalg.inv(torch.eye(size) - lower)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("sequence", [64, 128, 200])
def test_matches_attn_gym_reference(sequence):
    from attn_gym.linear.gdn import chunk_gdn

    from ascend_titan.kernels.gdn import ascend_chunk_gdn

    args = _inputs(2, 3, sequence, 16, 16, torch.float32)
    want = chunk_gdn(*args, impl="reference")
    want = want[0] if isinstance(want, tuple) else want
    got = ascend_chunk_gdn(*args)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_matches_attn_gym_reference_bf16():
    from attn_gym.linear.gdn import chunk_gdn

    from ascend_titan.kernels.gdn import ascend_chunk_gdn

    args = _inputs(1, 2, 128, 32, 32, torch.bfloat16)
    want = chunk_gdn(*args, impl="reference")
    want = want[0] if isinstance(want, tuple) else want
    got = ascend_chunk_gdn(*args)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


def test_gradients_match_attn_gym_reference():
    from attn_gym.linear.gdn import chunk_gdn

    from ascend_titan.kernels.gdn import ascend_chunk_gdn

    base = _inputs(1, 2, 128, 16, 16, torch.float32, seed=7)

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
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunk_size", [32, 64])
def test_chunk_size_does_not_change_the_answer(chunk_size):
    """The chunk decomposition is exact; only the floating-point order changes."""
    from attn_gym.linear.gdn import chunk_gdn

    from ascend_titan.kernels.gdn import ascend_chunk_gdn

    args = _inputs(1, 2, 512, 32, 32, torch.float32, seed=3)
    want = chunk_gdn(*args, impl="reference", chunk_size=64)
    want = want[0] if isinstance(want, tuple) else want
    got = ascend_chunk_gdn(*args, chunk_size=chunk_size)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_why_chunk_size_stays_64():
    """``CHUNK_SIZE`` is 64 for conditioning, not for taste -- this is the reason.

    The intra-chunk transition matrix is ``(I - A)^-1`` for a strictly
    lower-triangular ``A`` whose entries are ``beta * (k_i . k_j) * decay``, i.e.
    bounded by one. Its magnitude grows fast with the chunk size, and everything
    downstream multiplies by it. This is a property of the decomposition, not of
    ``_unit_lower_inverse``: forward substitution, which attn_gym uses, produces
    the same huge numbers, which is why fla and attn_gym also stop at 64.

    Not to be confused with qwen3.5-0.8B's step-4 non-finite loss: that happens
    at C=64 as well, so it is the training configuration, not this constant.
    """
    from ascend_titan.kernels.gdn import CHUNK_SIZE, _unit_lower_inverse

    assert CHUNK_SIZE == 64

    peaks = {}
    for size in (64, 128, 256):
        gen = torch.Generator().manual_seed(7)
        lower = (torch.rand(size, size, generator=gen) * 2 - 1).tril(-1)
        exact = torch.linalg.inv(torch.eye(size) - lower)
        peaks[size] = exact.abs().max().item()
        # Ours tracks the exact inverse relatively well at every size; the
        # problem is the size of the answer, not the method.
        got = _unit_lower_inverse(lower)
        assert (got - exact).abs().max() <= 1e-5 * peaks[size]

    assert peaks[64] < 1e5 < peaks[128] < peaks[256]


@pytest.mark.parametrize(
    ("label", "beta_scale", "decay_scale"),
    [
        ("nominal", 1.0, 1.0),
        # beta -> 1 with almost no decay is the worst case for the delta rule:
        # every token writes at full strength and nothing forgets, so the
        # transition matrix is as far from the identity as the gates allow.
        ("beta->1, decay->0", 1.0, 1e-4),
        ("beta small", 0.1, 1.0),
    ],
)
def test_stays_finite_and_matches_reference_under_extreme_gates(label, beta_scale, decay_scale):
    """Rules out the kernel as the source of a non-finite loss.

    qwen3.5-0.8B goes non-finite early in training; this pins that it is not
    because ``ascend_chunk_gdn`` blows up where attn_gym's reference would not.
    Measured at seq 16384 too (same result, 1.6e-7 apart); 512 keeps the test
    fast enough for the CPU suite.
    """
    from attn_gym.linear.gdn import chunk_gdn

    from ascend_titan.kernels.gdn import ascend_chunk_gdn

    gen = torch.Generator().manual_seed(2)

    def unit(*shape):
        return torch.nn.functional.normalize(torch.randn(*shape, generator=gen), dim=-1)

    args = (
        unit(1, 4, 512, 64),
        unit(1, 4, 512, 64),
        torch.randn(1, 4, 512, 64, generator=gen),
        -torch.rand(1, 4, 512, generator=gen) * decay_scale,
        torch.rand(1, 4, 512, generator=gen) * beta_scale,
    )
    want = chunk_gdn(*args, impl="reference")
    want = want[0] if isinstance(want, tuple) else want
    got = ascend_chunk_gdn(*args)
    assert torch.isfinite(got).all(), label
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_inverse_backward_matches_float64_ground_truth():
    """``_UnitLowerInverse`` differentiates as ``X^T g X^T`` -- checked in float64.

    The ground truth is autograd through ``torch.linalg.inv``, which implements
    the same identity. The comparison is deliberately *not* against autograd
    walking our own doubling series: that chain keeps every ``A**k`` alive in the
    backward graph, and those powers are far larger than the sum they telescope
    into, so it is the less accurate of the two (measured: they disagree by ~4e7
    on gradients of order 1e7 at C=64).
    """
    from ascend_titan.kernels.gdn import _unit_lower_inverse

    gen = torch.Generator().manual_seed(4)
    lower = (torch.rand(2, 64, 64, generator=gen, dtype=torch.float64) * 2 - 1).tril(-1)
    upstream_grad = torch.randn(2, 64, 64, generator=gen, dtype=torch.float64)

    ours = lower.clone().requires_grad_()
    _unit_lower_inverse(ours).backward(upstream_grad)

    truth = lower.clone().requires_grad_()
    eye = torch.eye(64, dtype=torch.float64)
    torch.linalg.inv(eye - truth).backward(upstream_grad)

    assert ours.grad is not None and truth.grad is not None
    # A is strictly lower triangular, so only that triangle is a real derivative;
    # ours masks the rest to zero instead of reporting the chain rule's leftovers.
    torch.testing.assert_close(ours.grad, truth.grad.tril(-1), rtol=1e-9, atol=1e-9)
    assert (ours.grad.triu() == 0).all()
