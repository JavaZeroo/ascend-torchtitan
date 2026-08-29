"""Shim modules. One file per shim (or per tightly related group).

Template::

    from ascend_titan.compat import shim

    @shim(
        target="torchtitan.some.module:some_function",
        reason="one line: what breaks on NPU without this",
        upstream="https://github.com/pytorch/torchtitan/issues/NNNN",
        kind="wrap",
    )
    def my_shim(original):
        def wrapped(*args, **kwargs):
            ...
            return original(*args, **kwargs)
        return wrapped

The count of files here is a health metric: it should trend to zero.
"""
