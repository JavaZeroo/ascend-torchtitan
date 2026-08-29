"""AscendComplexRoPE must be numerically identical to upstream ComplexRoPE (CPU)."""

import pytest
import torch

pytestmark = pytest.mark.titan


@pytest.mark.parametrize("scaling", ["none", "llama", "yarn"])
@pytest.mark.parametrize("with_positions", [False, True])
def test_matches_upstream_complex_rope(scaling, with_positions):
    from torchtitan.config.override import clear_overrides
    from torchtitan.models.common.rope import ComplexRoPE

    from ascend_titan.kernels.rope import AscendComplexRoPE, real_cache_rope

    kw = dict(dim=32, max_context_length=64, scaling=scaling)
    if scaling == "yarn":
        kw["rope_factor"] = 2.0
    ref = ComplexRoPE.Config(**kw).build()
    new = real_cache_rope(ComplexRoPE.Config(**kw)).build()
    assert isinstance(new, AscendComplexRoPE)
    assert new.cache.dtype == torch.float32 and new.cache.shape == (64, 16, 2)

    torch.manual_seed(0)
    q = torch.randn(40, 4, 32, dtype=torch.bfloat16)
    k = torch.randn(40, 2, 32, dtype=torch.bfloat16)
    pos = torch.cat([torch.arange(24), torch.arange(16)]) if with_positions else None
    q1, k1 = ref(q, k, pos)
    q2, k2 = new(q, k, pos)
    torch.testing.assert_close(q1, q2, atol=0, rtol=0)
    torch.testing.assert_close(k1, k2, atol=0, rtol=0)
    clear_overrides()


def test_override_is_exact_and_registered():
    from torchtitan.config.override import clear_overrides
    from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE

    from ascend_titan.kernels.rope import AscendComplexRoPE, real_cache_rope

    out = real_cache_rope(ComplexRoPE.Config(dim=8, max_context_length=8))
    assert isinstance(out, AscendComplexRoPE.Config) and isinstance(out, ComplexRoPE.Config)
    assert not isinstance(out, CosSinRoPE.Config)
    clear_overrides()
