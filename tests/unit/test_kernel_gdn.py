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


def test_inverse_is_accurate_in_the_regime_the_kernel_runs_in():
    """``|A| <= 1`` is the real regime: ``beta * (k_i . k_j) * decay`` with unit k."""
    from ascend_titan.kernels.gdn import _unit_lower_inverse

    gen = torch.Generator().manual_seed(12)
    lower = (torch.rand(4, 64, 64, generator=gen) * 2 - 1).tril(-1)
    exact = torch.linalg.inv(torch.eye(64) - lower)
    ours = _unit_lower_inverse(lower)
    torch.testing.assert_close(ours, exact, rtol=1e-4, atol=1e-4 * exact.abs().max())


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


def _reference_per_segment(q, k, v, g, beta, bounds):
    """attn_gym's reference run independently per packed segment, then concatenated.

    That *is* the definition of packing for a recurrence: the state must restart
    at every document boundary.
    """
    from attn_gym.linear.gdn import chunk_gdn

    from ascend_titan.kernels.gdn import l2norm

    out = []
    for begin, end in zip(bounds[:-1], bounds[1:], strict=True):
        if end <= begin:
            continue
        piece = chunk_gdn(
            l2norm(q[begin:end].unsqueeze(0)).transpose(1, 2),
            l2norm(k[begin:end].unsqueeze(0)).transpose(1, 2),
            v[begin:end].unsqueeze(0).transpose(1, 2),
            g[begin:end].unsqueeze(0).transpose(1, 2),
            beta[begin:end].unsqueeze(0).transpose(1, 2),
            impl="reference",
        )
        piece = piece[0] if isinstance(piece, tuple) else piece
        out.append(piece.transpose(1, 2).squeeze(0))
    return torch.cat(out, dim=0)


def test_packed_path_matches_per_segment_reference():
    """The packed branch had no test at all -- and it is the branch that runs.

    Real C4 packed to 4096 tokens carries dozens of document boundaries, so
    ``AscendGatedDeltaKernel.forward`` takes the ``cu_seqlens`` loop, not the
    dense call that every other test in this file exercises. Segment lengths
    below and above ``CHUNK_SIZE`` are both included: a short segment is padded
    to a full chunk, which is exactly where an off-by-one would hide.
    """
    from ascend_titan.kernels.gdn import AscendGatedDeltaKernel

    kernel = AscendGatedDeltaKernel(AscendGatedDeltaKernel.Config())

    gen = torch.Generator().manual_seed(9)
    heads, key_dim, value_dim = 4, 32, 32
    lengths = [7, 64, 65, 130]  # < chunk, == chunk, > chunk, multi-chunk
    tokens = sum(lengths)
    bounds = [0]
    for n in lengths:
        bounds.append(bounds[-1] + n)

    q = torch.randn(tokens, heads, key_dim, generator=gen)
    k = torch.randn(tokens, heads, key_dim, generator=gen)
    v = torch.randn(tokens, heads, value_dim, generator=gen)
    g = -torch.rand(tokens, heads, generator=gen)
    beta = torch.rand(tokens, heads, generator=gen)

    cu = torch.tensor(bounds, dtype=torch.int32)
    got = kernel(q, k, v, g, beta, cu_seqlens=cu, cu_seqlens_cpu=cu)
    want = _reference_per_segment(q, k, v, g, beta, bounds)

    assert got.shape == want.shape, (got.shape, want.shape)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_packed_path_is_not_the_dense_path():
    """A packed run must differ from ignoring the boundaries -- else the loop is dead.

    Guards against a regression where ``cu_seqlens`` stops reaching the kernel
    and the recurrence silently runs across document boundaries.
    """
    from ascend_titan.kernels.gdn import AscendGatedDeltaKernel

    kernel = AscendGatedDeltaKernel(AscendGatedDeltaKernel.Config())
    gen = torch.Generator().manual_seed(10)
    tokens, heads, dim = 128, 4, 32
    args = (
        torch.randn(tokens, heads, dim, generator=gen),
        torch.randn(tokens, heads, dim, generator=gen),
        torch.randn(tokens, heads, dim, generator=gen),
        -torch.rand(tokens, heads, generator=gen),
        torch.rand(tokens, heads, generator=gen),
    )
    cu = torch.tensor([0, 40, 128], dtype=torch.int32)
    packed = kernel(*args, cu_seqlens=cu, cu_seqlens_cpu=cu)
    dense = kernel(*args)
    assert not torch.allclose(packed, dense, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("size", [8, 16, 24, 32, 64])
def test_inverse_handles_sizes_that_do_not_divide_the_base_block(size):
    """``BASE_BLOCK`` does not divide every chunk size; that path must still be exact.

    Sizes below the base block, equal to it, and indivisible by it (24) all take
    a different branch in ``_UnitLowerInverse.forward``.
    """
    from ascend_titan.kernels.gdn import _unit_lower_inverse

    gen = torch.Generator().manual_seed(size)
    lower = (torch.rand(3, size, size, generator=gen) * 2 - 1).tril(-1)
    exact = torch.linalg.inv(torch.eye(size) - lower)
    got = _unit_lower_inverse(lower)
    torch.testing.assert_close(got, exact, rtol=1e-4, atol=1e-4 * exact.abs().max())
