"""Fused SwiGLU feed-forward on ``torch_npu.npu_swiglu``.

Targets ``torchtitan/models/common/feed_forward.py::FeedForward.Config`` and
reuses upstream's ``torchtitan/overrides/fused_swiglu.py`` wholesale for the
parts that are not a kernel: the fused ``w13`` weight (gate/up rows
interleaved so TP row-sharding keeps both halves), its parameter init, the
FSDP/TP ``sharding_config`` and the checkpoint hooks that save/load the stock
``w1.weight``/``w3.weight`` layout. Only the activation differs: upstream calls
a Triton ``silu_and_mul`` kernel (CUDA-only, matrix ``TT-KERNEL``), we call
``npu_swiglu`` which computes ``silu(a) * b`` over the two halves of its last
dim (Meta + autograd registered in torch_npu, so compile/FSDP compose).

The interleaved ``w13`` output is regrouped into ``[gate | up]`` with one
``torch.cat`` (a ``(T, 2*hidden)`` copy) before the kernel; cheap next to the
GEMMs, and it keeps the TP-friendly weight layout untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import torch_npu

    _AVAILABLE = hasattr(torch_npu, "npu_swiglu")
except ImportError as e:
    _AVAILABLE = False
    logger.warning("[ascend_titan] torch_npu unavailable (%s); FeedForward stays on eager", e)

if _AVAILABLE:
    import torch
    from torchtitan.config import derive, override
    from torchtitan.models.common.feed_forward import FeedForward
    from torchtitan.overrides.fused_swiglu import FusedSwiGLU, _fused_swiglu_config

    class AscendFusedSwiGLU(FusedSwiGLU):
        """Upstream ``FusedSwiGLU`` with ``npu_swiglu`` as the activation."""

        @dataclass(kw_only=True, slots=True)
        class Config(FusedSwiGLU.Config):
            pass

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate, up = self.w13(x).unflatten(-1, (-1, 2)).unbind(-1)
            return self.w2(torch_npu.npu_swiglu(torch.cat((gate, up), dim=-1), dim=-1))

    @override(
        target=FeedForward.Config,
        exact=True,
        description="Fused gate+up GEMM (upstream FusedSwiGLU layout) + torch_npu.npu_swiglu",
    )
    def npu_fused_swiglu(cfg: FeedForward.Config) -> AscendFusedSwiGLU.Config:
        # Upstream builds the fused w13 Linear config (interleaved rows, per-half init,
        # sharding); `derive` there targets the class we pass in.
        fused = _fused_swiglu_config(cfg, FusedSwiGLU.Config)
        return derive(fused, AscendFusedSwiGLU.Config)
