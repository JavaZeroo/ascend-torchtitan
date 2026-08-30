"""L3 machinery shared across models.

Per-model recipes do **not** live here -- they live in
``ascend_titan/models/<model>/`` (one package per model, each with its own
README). This package holds only what is model-independent:

    transforms.py   npu_baseline: the deltas every upstream config needs on NPU
    matrix.py       dynamic module that runs any upstream config + npu_baseline

Content in ``models/``, machinery in ``recipes/``.

Run a recipe:  python -m ascend_titan.train --module ascend_titan.models.<model> --config <fn>
Run any upstream config:
               python -m ascend_titan.train --module ascend_titan.recipes.matrix \
                   --config <upstream.module>__<fn>
"""
