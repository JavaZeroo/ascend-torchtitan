"""`import ascend_titan` must not import torch / torch_npu / torchtitan (F4)."""

import subprocess
import sys

CODE = """
import sys
import ascend_titan
import ascend_titan.compat, ascend_titan.kernels, ascend_titan.models, ascend_titan.recipes  # noqa
bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'torch_npu', 'torchtitan')]
print(",".join(sorted(bad)))
"""


def test_import_has_no_side_effects():
    out = subprocess.check_output([sys.executable, "-c", CODE], text=True).strip()
    assert out == "", f"import ascend_titan pulled in: {out}"
