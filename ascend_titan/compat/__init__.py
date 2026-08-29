"""L0 compat layer: governed monkeypatches ("shims") for torchtitan code that the
override mechanism cannot reach.

Rules (docs/PRINCIPLES.md P0, P1, P3, P4):
  * P0  If torchtitan already exposes a config switch, use it; do not shim.
  * P1  Never shim around a torch_npu defect. File a torch_npu issue instead.
  * P3  Prefer ``kind="wrap"`` (call the original, add behaviour around it) over
        ``kind="replace"``; a replacement must state ``why_not_wrap``.
  * P4  Every shim carries an ``upstream`` issue/PR link. It is a debt with a
        due date: delete the shim when upstream fixes it.

Shim modules live in ``ascend_titan/compat/shims/`` and are imported by
``apply_all()``; importing this package alone registers nothing.
"""

from ascend_titan.compat.registry import (
    Applied,
    Shim,
    ShimError,
    apply_all,
    list_shims,
    shim,
)

__all__ = ["Applied", "Shim", "ShimError", "apply_all", "list_shims", "shim"]
