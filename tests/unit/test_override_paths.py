"""The ``*_OVERRIDE`` path constants in ``ascend_titan.kernels`` are the single
source for override targets (P11). This test pins them to reality: each must
point at a ``def`` that actually exists in this repo, so moving or renaming a
factory without updating the constant fails on CPU, not in a training job.

Pure text check on purpose: importing the kernel modules needs torch_npu.
"""

import re
from pathlib import Path

import ascend_titan.kernels as kernels

KERNELS_DIR = Path(kernels.__file__).parent


def _override_constants() -> dict[str, str]:
    consts = {n: v for n, v in vars(kernels).items() if n.endswith("_OVERRIDE")}
    assert consts, "no *_OVERRIDE constants exported by ascend_titan.kernels"
    return consts


def test_override_paths_resolve_to_real_factories():
    for name, target in _override_constants().items():
        prefix, mod, fn = target.rsplit(".", 2)
        assert prefix == "ascend_titan.kernels", f"{name}: unexpected package {prefix!r}"
        src_file = KERNELS_DIR / f"{mod}.py"
        assert src_file.is_file(), f"{name}: {src_file} does not exist"
        # an optional-addon factory may live under an `if <probe>:` guard, indented
        assert re.search(rf"^\s*def {fn}\(", src_file.read_text(), re.MULTILINE), (
            f"{name}: no `def {fn}` in kernels/{mod}.py -- the factory moved or was "
            "renamed; update the constant in ascend_titan/kernels/__init__.py"
        )


def test_override_paths_are_not_redefined_elsewhere():
    """No other file may spell out an override path literal (P11)."""
    pkg_root = KERNELS_DIR.parent
    offenders = []
    for py in pkg_root.rglob("*.py"):
        if py == KERNELS_DIR / "__init__.py":
            continue
        if '"ascend_titan.kernels.' in py.read_text():
            offenders.append(str(py.relative_to(pkg_root)))
    assert not offenders, (
        f"override path literals outside kernels/__init__.py: {offenders}; "
        "import the *_OVERRIDE constants from ascend_titan.kernels instead"
    )
