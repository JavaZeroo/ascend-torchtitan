"""RMSNorm on ``torch_npu.npu_rms_norm``.

Targets ``torchtitan/models/common/nn_modules.py::RMSNorm.Config`` (the
``nn.RMSNorm``-based node used by every upstream decoder). The fused kernel has
Meta and autograd registrations in torch_npu, so it composes with
``torch.compile`` and FSDP without further work. Numerics: same definition as
``torch.rms_norm`` (fp32 statistics); see tests/npu/test_kernel_rms_norm.py.

Kept as a pure drop-in: parameter name (``weight``), shape and checkpoint
layout are unchanged. ``elementwise_affine=False`` falls back to the upstream
forward (the kernel requires a gamma tensor).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import torch_npu

    _AVAILABLE = hasattr(torch_npu, "npu_rms_norm")
except ImportError as e:
    _AVAILABLE = False
    logger.warning("[ascend_titan] torch_npu unavailable (%s); RMSNorm stays on torch.rms_norm", e)

if _AVAILABLE:
    import torch
    from torchtitan.config import derive, override
    from torchtitan.models.common.nn_modules import RMSNorm

    class AscendRMSNorm(RMSNorm):
        @dataclass(kw_only=True, slots=True)
        class Config(RMSNorm.Config):
            pass

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.weight is None:
                return super().forward(x)
            eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
            return torch_npu.npu_rms_norm(x, self.weight, eps)[0]

    @override(target=RMSNorm.Config, description="RMSNorm via torch_npu.npu_rms_norm")
    def npu_rms_norm(cfg: RMSNorm.Config) -> AscendRMSNorm.Config:
        return derive(cfg, AscendRMSNorm.Config)
