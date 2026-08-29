"""ascend-torchtitan: out-of-tree Ascend NPU extension for torchtitan.

Importing this package has NO side effects. In particular it must not import
``torch``, ``torch_npu`` or ``torchtitan``: torchtitan's override mechanism
imports our ``ascend_titan.kernels.*`` modules from inside ``Trainer.__init__``
(``torchtitan/config/override.py``), and any import-time side effect here would
fire at an uncontrolled point in torchtitan's initialisation.

All side effects live behind the explicit :func:`setup` call.
"""

__version__ = "0.0.1.dev0"

__all__ = ["__version__", "setup"]


def setup(**kwargs):
    """Apply Ascend bootstrap (device backend import + compat shims).

    Thin lazy wrapper so that ``import ascend_titan`` stays side-effect free.
    See :mod:`ascend_titan._bootstrap` for the real implementation and options.
    """
    from ascend_titan._bootstrap import setup as _setup

    return _setup(**kwargs)
