"""L3 machinery shared across models.

Per-model recipes do **not** live here -- they live in
``ascend_titan/models/<model>/`` (one package per model, each with its own
README). This package holds only what is model-independent:

    deltas.py       the primitives a recipe applies (add_override / flex_to_varlen)
    matrix.py       the capability matrix: dynamic config resolution plus the
                    generic npu_minimal / npu_fused transforms it applies

Content in ``models/``, machinery in ``recipes/``.

Run a recipe:  python -m ascend_titan.train --module ascend_titan.models.<model> --config <fn>
Run any upstream config:
               python -m ascend_titan.train --module ascend_titan.recipes.matrix \
                   --config <upstream.module>__<fn>
"""
