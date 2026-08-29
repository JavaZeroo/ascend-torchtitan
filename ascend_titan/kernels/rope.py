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
attention override. Import is safe without torchtitan features missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torchtitan.config import derive, override
from torchtitan.models.common.rope import (
    ComplexRoPE,
    _maybe_check_max_pos,
    _maybe_wrap_positions,
)

logger = logging.getLogger(__name__)


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
