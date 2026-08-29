"""Kimi-K3 SiTU-GLU on the ops-nn AscendC fused kernel (``aclnnSituGlu``).

Targets ``torchtitan/models/kimi_k3/moe.py::KimiFeedForward.Config``. Upstream
computes ``beta * tanh(gate/beta) * sigmoid(gate) * [linear_beta * tanh(up/linear_beta)]``
in fp32 with ~8 elementwise ops; ops-nn fuses forward (``situ_glu``) and
backward (``situ_glu_grad``) into one kernel each.

Kernel access: the ops-nn python package ``cann_ops_nn`` registers
``torch.ops.cann_ops_nn.situ_glu(x, *, dim, beta, linear_beta, activate_left)``
(JIT-built torch extension over the installed ``aclnnSituGlu``; needs the
ops-nn run package installed under ``$ASCEND_HOME_PATH/opp/vendors`` and
``ninja``). ``x`` is ``[gate | up]`` along ``dim``; ``linear_beta <= 0`` means
pass-through, matching upstream's ``linear_beta=None``.

Build: ``scripts/build_kernels.sh ops-nn``. Without the package the override is
not registered and upstream eager runs (ADR-004).
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ops-nn's torch extension is installed as ``cann_ops_nn`` (full build) or
# ``cann_ops_nn_<vendor>`` (single-op build with TORCH_EXTENSION_OPS/VENDOR); both
# register the same ``cann_ops_nn`` torch library namespace on import.
_CANDIDATES = ("cann_ops_nn", "cann_ops_nn_ascend_titan", "cann_ops_nn_custom")
_AVAILABLE = False
_err: Exception | None = None
for _mod in _CANDIDATES:
    try:
        importlib.import_module(_mod)
        import torch

        _AVAILABLE = hasattr(torch.ops, "cann_ops_nn") and hasattr(
            torch.ops.cann_ops_nn, "situ_glu"
        )
        if _AVAILABLE:
            break
    except Exception as e:  # noqa: BLE001 - JIT builders raise many kinds of errors
        _err = e
if not _AVAILABLE:
    logger.warning(
        "[ascend_titan] ops-nn situ_glu unavailable (%s); KimiFeedForward stays on upstream eager",
        _err,
    )

if _AVAILABLE:
    from torchtitan.config import derive, override
    from torchtitan.models.kimi_k3.moe import KimiFeedForward

    class AscendKimiFeedForward(KimiFeedForward):
        @dataclass(kw_only=True, slots=True)
        class Config(KimiFeedForward.Config):
            pass

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate_up = torch.cat((self.w1(x), self.w3(x)), dim=-1)
            act = torch.ops.cann_ops_nn.situ_glu(
                gate_up,
                dim=-1,
                beta=float(self.beta),
                linear_beta=float(self.linear_beta) if self.linear_beta is not None else 0.0,
                activate_left=True,
            )
            return self.w2(act)

    @override(
        target=KimiFeedForward.Config,
        exact=True,
        description="Kimi SiTU-GLU via ops-nn aclnnSituGlu (fused fwd/bwd)",
    )
    def ops_nn_situ_glu(cfg: KimiFeedForward.Config) -> AscendKimiFeedForward.Config:
        return derive(cfg, AscendKimiFeedForward.Config)
