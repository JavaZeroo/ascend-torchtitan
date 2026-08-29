"""Polyfill ``torch.distributed.set_timeout`` on torch releases that predate it.

torchtitan calls ``torch.distributed.set_timeout(timeout, group)`` from
``torchtitan/distributed/utils.py::set_pg_timeouts`` after the first train
step (adopted upstream 2026-06-27, #3764). The function is the public rename of
``torch.distributed.distributed_c10d._set_pg_timeout``; in torch nightly the
old name is a deprecated alias of the new one with identical semantics, so the
polyfill simply re-exports the old name. On a torch that already has
``set_timeout`` the shim is skipped.

Attribution: TT (torchtitan uses a nightly-only API without a feature check;
upstream has ``check_if_feature_in_pytorch`` for exactly this).
"""

from ascend_titan.compat import shim


@shim(
    target="torch.distributed:set_timeout",
    reason="torchtitan's set_pg_timeouts needs torch.distributed.set_timeout, "
    "absent in torch<=2.13 (public rename of _set_pg_timeout)",
    upstream="draft:docs/issues/torchtitan.md#set-timeout",
    kind="polyfill",
)
def dist_set_timeout(original):
    assert original is None
    from torch.distributed.distributed_c10d import _set_pg_timeout

    def set_timeout(timeout, group=None):
        return _set_pg_timeout(timeout, group)

    set_timeout.__doc__ = "Polyfill of torch.distributed.set_timeout (ascend_titan)."
    return set_timeout
