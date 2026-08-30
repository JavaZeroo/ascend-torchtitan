"""L3: one package per model.

``ascend_titan/models/<model>/`` holds everything that is specific to one model
family:

    <model>/__init__.py    re-exports the recipes so ``--module ascend_titan.models.<model>`` works
    <model>/recipes.py     the supported entry points (``<model>_<flavor>_npu[_<variant>]``)
    <model>/probes.py      optional: measurement-only configs for the capability matrix
    <model>/README.md      the model's usage guide (required -- test_models_registry enforces it)

Cross-model machinery (``npu_baseline``, the matrix resolver) stays in
:mod:`ascend_titan.recipes`. Status and metadata for every model live in
:mod:`ascend_titan.models.registry`, which is plain data: importing this package
must not import torch or torchtitan (P0/F4).

Run one:  python -m ascend_titan.train --module ascend_titan.models.<model> --config <fn>
Add one:  copy ``_template/`` and follow ``README.md``.
"""

from ascend_titan.models.registry import MODELS, ModelEntry

__all__ = ["MODELS", "ModelEntry"]
