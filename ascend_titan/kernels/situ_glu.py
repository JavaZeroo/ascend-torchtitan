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

_MODEL_AVAILABLE = False
if _AVAILABLE:
    try:
        # kimi_k3 imports attn_gym's cute backend at module level, which needs the
        # CUDA-only `cutlass` package (TT-11). The kernel wrapper below still works;
        # only the @override registration needs the model node.
        from torchtitan.models.kimi_k3.moe import KimiFeedForward

        _MODEL_AVAILABLE = True
    except ImportError as e:
        logger.warning("[ascend_titan] kimi_k3 not importable (%s); situ_glu override skipped", e)

if _AVAILABLE:
    # ops-nn's python extension registers forward and backward as two separate ops
    # without autograd glue (backprop through it warns and yields no grad). Wrap
    # them as one differentiable custom op with a fake kernel for compile.
    @torch.library.custom_op("ascend_titan::situ_glu", mutates_args=())
    def _situ_glu_fwd(x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
        return torch.ops.cann_ops_nn.situ_glu(
            x, dim=-1, beta=beta, linear_beta=linear_beta, activate_left=True
        )

    @_situ_glu_fwd.register_fake
    def _(x, beta, linear_beta):
        return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))

    @torch.library.custom_op("ascend_titan::situ_glu_bwd", mutates_args=())
    def _situ_glu_bwd(
        grad_y: torch.Tensor, x: torch.Tensor, beta: float, linear_beta: float
    ) -> torch.Tensor:
        return torch.ops.cann_ops_nn.situ_glu_grad(
            grad_y.contiguous(), x, dim=-1, beta=beta, linear_beta=linear_beta, activate_left=True
        )

    @_situ_glu_bwd.register_fake
    def _(grad_y, x, beta, linear_beta):
        return torch.empty_like(x)

    def _setup(ctx, inputs, output):
        x, beta, linear_beta = inputs
        ctx.save_for_backward(x)
        ctx.beta, ctx.linear_beta = beta, linear_beta

    def _backward(ctx, grad_y):
        (x,) = ctx.saved_tensors
        return _situ_glu_bwd(grad_y, x, ctx.beta, ctx.linear_beta), None, None

    _situ_glu_fwd.register_autograd(_backward, setup_context=_setup)

    def situ_glu(gate_up: torch.Tensor, *, beta: float, linear_beta: float | None) -> torch.Tensor:
        """SiTU-GLU over ``[gate | up]`` (last dim), fused fwd/bwd on ops-nn."""
        return _situ_glu_fwd(gate_up, float(beta), float(linear_beta or 0.0))


if _AVAILABLE and _MODEL_AVAILABLE:
    from torchtitan.config import derive, override

    class AscendKimiFeedForward(KimiFeedForward):
        @dataclass(kw_only=True, slots=True)
        class Config(KimiFeedForward.Config):
            pass

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate_up = torch.cat((self.w1(x), self.w3(x)), dim=-1)
            return self.w2(situ_glu(gate_up, beta=self.beta, linear_beta=self.linear_beta))

    @override(
        target=KimiFeedForward.Config,
        exact=True,
        description="Kimi SiTU-GLU via ops-nn aclnnSituGlu (fused fwd/bwd)",
    )
    def ops_nn_situ_glu(cfg: KimiFeedForward.Config) -> AscendKimiFeedForward.Config:
        return derive(cfg, AscendKimiFeedForward.Config)
