"""Kimi K3 recipes.

status: 🔴 TT-11 / DEP-CUTLASS -- ``torchtitan.models.kimi_k3`` imports attn-gym's
cute backend at module level, which needs the CUDA-only ``cutlass`` package, so
the import below fails on Ascend. That is the intended behaviour (P14): the gap
is upstream's, it is recorded in ``docs/issues/torchtitan.md`` (TT-11), and the
fix sketch lives in ``patches/evidence/torchtitan/`` -- never applied, and never
filed upstream (P10: github.com/pytorch is read-only for us).

What *is* ready: the SiTU-GLU fused kernel in ``ascend_titan/kernels/situ_glu.py``
(ops-nn ``aclnnSituGlu``), which targets ``KimiFeedForward.Config``.

Full guide: ascend_titan/models/kimi_k3/README.md
"""

from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
from torchtitan.trainer import Trainer

SITU_GLU_OVERRIDE = "ascend_titan.kernels.situ_glu.ops_nn_situ_glu"


def kimi_k3_debugmodel_npu() -> Trainer.Config:
    """Upstream ``kimi_k3_debugmodel`` + the fused SiTU-GLU override."""
    config = kimi_k3_debugmodel()

    # DELTA 1: fused SiTU-GLU (ops-nn AscendC kernel) for KimiFeedForward.
    config.override.imports = [*config.override.imports, SITU_GLU_OVERRIDE]

    # DELTA 2: no checkpoint I/O in a smoke run.
    config.checkpoint.enable = False

    return config
