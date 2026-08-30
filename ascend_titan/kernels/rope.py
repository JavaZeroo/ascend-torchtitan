"""Complex-convention RoPE with a real-valued cache, for Ascend.

Targets ``torchtitan/models/common/rope.py::ComplexRoPE.Config`` (llama3,
deepseek_v3, gpt_oss, kimi, muse_glimmer, ...). Upstream keeps the RoPE cache as
a complex64 tensor and gathers it with ``cache[positions]``; torch_npu's
``aclnnIndex`` rejects advanced indexing on complex tensors (error 161002,
docs/issues/torch_npu.md NPU-3), so every model using ``ComplexRoPE`` dies in
the first forward.

This module keeps upstream's *math* (adjacent-pair rotation == complex
multiply, so checkpoints and numerics are unchanged) but stores the cache as
``view_as_real(freqs_cis)`` -> ``(L, dim/2, 2)`` and applies the rotation with
real ops. It is not a workaround inside torch_npu: it is an alternative upstream
node implementation selected by the override mechanism, exactly like the
attention override.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchtitan.config import derive, override
from torchtitan.models.common.rope import (
    ComplexRoPE,
    CosSinRoPE,
    _maybe_check_max_pos,
    _maybe_wrap_positions,
)

from ascend_titan.kernels._probe import require_op

# torch_npu is a base dependency (P14): a missing module or op raises at import.
# The rotation itself still runs in plain torch for non-NPU tensors (device
# dispatch, not a fallback for a missing dependency).
_npu_rotary_mul = require_op("npu_rotary_mul")


def _rotary_kernel(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: str
) -> torch.Tensor:
    """``torch_npu.npu_rotary_mul`` on ``(T, N, H)`` with ``(T, 1, H)`` cos/sin.

    The kernel wants 4-D inputs; a leading batch of 1 is added and removed. It
    computes in the input dtype (bf16 in, bf16 out) -- within bf16 rounding of
    the upstream fp32 math (see tests/npu/test_kernel_rope.py).
    """
    return _npu_rotary_mul(
        x.unsqueeze(0),
        cos.to(x.dtype).unsqueeze(0),
        sin.to(x.dtype).unsqueeze(0),
        rotary_mode=mode,
    ).squeeze(0)


def _use_kernel(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


class AscendComplexRoPE(ComplexRoPE):
    """``ComplexRoPE`` semantics with a real ``(L, dim/2, 2)`` [cos, sin] cache."""

    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    def _precompute_cache(self) -> torch.Tensor:
        # Same frequencies (incl. llama / yarn scaling) as upstream, then real view.
        return torch.view_as_real(super()._precompute_cache()).contiguous()

    def _reshape_cache(
        self, query: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        positions = _maybe_wrap_positions(positions, query)
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        num_tokens = query.shape[0]
        cache = self.cache[:num_tokens] if positions is None else self.cache[positions]
        return cache.view(num_tokens, 1, cache.shape[-2], 2)  # (T, 1, dim/2, 2)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor, key: torch.Tensor, rope_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(a + ib)(c + is) = (ac - bs) + i(as + bc) on adjacent (even, odd) pairs."""
        cos = rope_cache[..., 0]
        sin = rope_cache[..., 1]
        if _use_kernel(query):
            # interleave mode wants each angle repeated for its (even, odd) pair.
            cos2 = cos.repeat_interleave(2, dim=-1)
            sin2 = sin.repeat_interleave(2, dim=-1)
            return (
                _rotary_kernel(query, cos2, sin2, "interleave"),
                _rotary_kernel(key, cos2, sin2, "interleave"),
            )

        def rot(x: torch.Tensor) -> torch.Tensor:
            xf = x.float().reshape(*x.shape[:-1], -1, 2)
            a, b = xf[..., 0], xf[..., 1]
            out = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
            return out.flatten(-2).type_as(x)

        return rot(query), rot(key)


@override(
    target=ComplexRoPE.Config,
    exact=True,
    description="ComplexRoPE with a real-valued cache (torch_npu cannot index complex tensors)",
)
def real_cache_rope(cfg: ComplexRoPE.Config) -> AscendComplexRoPE.Config:
    return derive(cfg, AscendComplexRoPE.Config)


class AscendCosSinRoPE(CosSinRoPE):
    """Upstream ``CosSinRoPE`` (rotate-half, used by qwen3 & co.) applied with
    ``npu_rotary_mul(rotary_mode="half")``. Cache layout and numerics contract are
    upstream's; only the rotation is fused. Falls back to upstream math off-NPU."""

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        pass

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor, key: torch.Tensor, rope_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _use_kernel(query):
            return CosSinRoPE.apply_rotary_emb(query, key, rope_cache)
        head_dim = query.shape[-1]
        cos = rope_cache[..., :head_dim]
        sin = rope_cache[..., head_dim:]
        return _rotary_kernel(query, cos, sin, "half"), _rotary_kernel(key, cos, sin, "half")


@override(
    target=CosSinRoPE.Config,
    exact=True,
    description="CosSinRoPE rotation via torch_npu.npu_rotary_mul (rotate-half)",
)
def npu_rotary_cossin(cfg: CosSinRoPE.Config) -> AscendCosSinRoPE.Config:
    return derive(cfg, AscendCosSinRoPE.Config)
