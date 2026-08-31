"""CPU tests for the Ascend KDA override.

The kernel path itself (attn_gym's reference impl) is upstream's own code, so
what needs testing here is the piece we wrote: the depthwise causal convolution,
including the packed-sequence boundary rule that the CuTeDSL kernel implements
with cu_seqlens.
"""

import pytest
import torch

pytestmark = pytest.mark.titan

# kimi_k3 imports attn_gym's cute backend at module level; `nvidia-cutlass-dsl`
# provides it and has aarch64 wheels, but it is an extra, not a base dependency.
pytest.importorskip("cutlass", reason="pip install nvidia-cutlass-dsl (kimi_k3 extra)")


@pytest.fixture(autouse=True)
def _clear_overrides():
    """npu_stub re-imports the kernel modules, which re-runs the @override
    decorators; torchtitan rejects a duplicate registration."""
    from torchtitan.config.override import clear_overrides

    clear_overrides()
    yield
    clear_overrides()


def _naive_causal_conv1d(x_BTC, weight_CW, starts):
    """Straightforward reference: one output element at a time."""
    B, T, C = x_BTC.shape
    W = weight_CW.shape[1]
    out = torch.zeros(B, T, C, dtype=torch.float32)
    for b in range(B):
        for t in range(T):
            for c in range(C):
                acc = 0.0
                for i in range(W):
                    src = t - (W - 1 - i)
                    if src >= starts[t]:
                        acc += float(x_BTC[b, src, c]) * float(weight_CW[c, i])
                out[b, t, c] = acc
    return out


def test_causal_conv1d_matches_naive_dense(npu_stub):
    from ascend_titan.kernels.kda import ascend_causal_conv1d

    torch.manual_seed(0)
    x = torch.randn(2, 9, 4)
    w = torch.randn(4, 3)
    got = ascend_causal_conv1d(x, w)
    want = _naive_causal_conv1d(x, w, [0] * 9)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_causal_conv1d_respects_sequence_boundaries(npu_stub):
    """A tap must never read across a packed sequence start (cu_seqlens contract)."""
    from ascend_titan.kernels.kda import ascend_causal_conv1d

    torch.manual_seed(0)
    x = torch.randn(1, 8, 3)
    w = torch.randn(3, 4)
    cu = torch.tensor([0, 5, 8], dtype=torch.int32)
    starts = [0] * 5 + [5] * 3
    got = ascend_causal_conv1d(x, w, cu_seqlens=cu)
    want = _naive_causal_conv1d(x, w, starts)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)

    # and the packed result must equal convolving each sequence on its own
    first = ascend_causal_conv1d(x[:, :5], w)
    second = ascend_causal_conv1d(x[:, 5:], w)
    torch.testing.assert_close(got, torch.cat((first, second), dim=1), atol=1e-5, rtol=1e-5)


def test_causal_conv1d_silu_and_dtype(npu_stub):
    from ascend_titan.kernels.kda import ascend_causal_conv1d

    x = torch.randn(1, 6, 2, dtype=torch.bfloat16)
    w = torch.randn(2, 2, dtype=torch.bfloat16)
    plain = ascend_causal_conv1d(x, w)
    activated = ascend_causal_conv1d(x, w, activation="silu")
    assert activated.dtype == torch.bfloat16
    torch.testing.assert_close(
        activated.float(), torch.nn.functional.silu(plain.float()), atol=2e-2, rtol=2e-2
    )
    with pytest.raises(NotImplementedError, match="gelu"):
        ascend_causal_conv1d(x, w, activation="gelu")


def test_override_replaces_the_whole_kda_subtree(npu_stub):
    """One override claims InnerKDA and derives its nested kernel config itself:
    torchtitan rejects an override whose ancestor another override claims."""
    from torchtitan.models.kimi_k3.kda import InnerKDA, KDAKernel

    from ascend_titan.kernels.kda import AscendInnerKDA, AscendKDAKernel, npu_kda

    cfg = InnerKDA.Config(head_dim=128, kernel=KDAKernel.Config(lower_bound=-4.0))
    new = npu_kda(cfg)
    assert isinstance(new, AscendInnerKDA.Config)
    assert isinstance(new.kernel, AscendKDAKernel.Config)
    assert new.kernel.lower_bound == -4.0  # deltas preserved


def test_causal_conv1d_matches_upstreams_own_dense_expression(npu_stub):
    """Dense branch must equal upstream's ``F.pad`` + ``F.conv1d`` + ``silu``.

    The other tests here compare against ``_naive_causal_conv1d``, which is our
    own reading of what the kernel should do -- same author, so the same
    misreading would pass both. This one compares against the expression
    torchtitan actually runs on the dense path
    (``models/qwen3_5/gdn.py::InnerGatedDeltaNet.forward``), which is the thing
    our override replaces.
    """
    import torch
    import torch.nn.functional as F

    from ascend_titan.kernels.kda import ascend_causal_conv1d

    gen = torch.Generator().manual_seed(3)
    tokens, channels, width = 37, 8, 4
    x_TC = torch.randn(tokens, channels, generator=gen)
    weight_C1W = torch.randn(channels, 1, width, generator=gen)

    x_1CT = F.pad(x_TC.transpose(0, 1).unsqueeze(0), [width - 1, 0])
    want = F.silu(F.conv1d(x_1CT, weight_C1W, None, groups=channels)).squeeze(0).transpose(0, 1)
    got = ascend_causal_conv1d(x_TC.unsqueeze(0), weight_C1W[:, 0], activation="silu").squeeze(0)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
