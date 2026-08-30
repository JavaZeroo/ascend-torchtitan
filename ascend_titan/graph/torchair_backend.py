"""Enable the torchair (GE graph) backend on a ``Trainer.Config``.

torchtitan already has the switch (``compile.enable`` + ``compile.backend``), so
this is a recipe delta, not a shim and not an override: P0 says use the switch.
What this module adds is the Ascend-specific knowledge around it -- that the
backend is called ``npu``, that it only exists in a torch_npu built with
torchair, and which components can currently be lowered.
"""

from __future__ import annotations

import importlib.util

from torchtitan.trainer import Trainer

# Components torchair can lower today; see the module docstring in __init__ for
# the measurement behind this, and docs/issues/ours.md OURS-13 for "model".
GRAPHABLE_COMPONENTS = ("loss",)


def torchair_available() -> bool:
    """True when the installed torch_npu carries torchair."""
    return importlib.util.find_spec("torch_npu.dynamo.torchair") is not None


def require_torchair() -> None:
    """Raise with the actual fix if torchair is missing (P14: no silent degrade)."""
    if torchair_available():
        return
    raise RuntimeError(
        "torch_npu was built with --disable_torchair, so graph mode is unavailable. "
        "Rebuild it with torchair and reinstall:\n"
        "  TORCHAIR=1 ./scripts/build_torch_npu.sh\n"
        "  pip install --no-deps --force-reinstall /opt/wheels/torch_npu-*.whl\n"
        "torchair's GE runtime also needs `decorator` and `scipy` in the venv."
    )


def npu_graph(
    config: Trainer.Config,
    *,
    components: list[str] | None = None,
) -> Trainer.Config:
    """Turn on ``torch.compile`` with Ascend's GE graph backend. In place.

    Args:
        components: which torchtitan components to compile. Defaults to the ones
            measured to lower on this stack; pass an explicit list to try more
            (and please record the result in docs/capability-matrix.md).
    """
    require_torchair()
    config.compile.enable = True
    config.compile.backend = "npu"
    config.compile.components = list(components or GRAPHABLE_COMPONENTS)
    return config
