#!/usr/bin/env bash
# 从 NIGHTLY 环境派生出 TRITON 环境（NIGHTLY + Triton-Ascend），让 inductor 在昇腾上可用。
#   ./scripts/install_triton.sh                  # 就地装进 /opt/venv-nightly（默认环境）
#   DST=/opt/venv-triton ./scripts/install_triton.sh   # 或派生一个环境用于对照
#
# Triton-Ascend 是 inductor 在昇腾上的后端：没有它 `torch.compile(backend="inductor")`
# 报 "0 active drivers"，torch.compile / CP / 模型级 flex 全部不可用。它是基线的一部分。
#
# 踩过的两个坑，都在下面处理掉了：
#   1) PyPI 的 triton 不卸干净 -> triton/__init__.py 还是 PyPI 那份，backends 里只有
#      amd/nvidia，报 "0 active drivers"。
#   2) `cp -a` 克隆 venv 后 bin/* 的 shebang 若没改，pip 会装回**源** venv。
set -euo pipefail
SRC=${SRC:-/opt/venv-nightly}
DST=${DST:-$SRC}          # 默认就地安装；DST != SRC 时先克隆一份
VERSION=${VERSION:-3.2.2}
INDEX=${INDEX:-https://triton-ascend.osinfra.cn/pypi/simple}

[ -d "$SRC" ] || { echo "源环境不存在：$SRC"; exit 2; }
if [ "$DST" != "$SRC" ]; then
  [ -e "$DST" ] && { echo "$DST 已存在；删掉它再跑，或换个 DST=" >&2; exit 2; }
  cp -a "$SRC" "$DST"
  sed -i "s|$SRC|$DST|g" "$DST/pyvenv.cfg" "$DST"/bin/activate*
  for f in "$DST"/bin/*; do
    [ -f "$f" ] && head -c2 "$f" 2>/dev/null | grep -q '#!' && sed -i "1s|$SRC|$DST|" "$f"
  done
fi

# pip 必须打在目标环境上——先自证，再动手
LOC=$("$DST/bin/python" -c 'import pip, os; print(os.path.dirname(os.path.dirname(pip.__file__)))')
case "$LOC" in "$DST"/*) ;; *) echo "pip 指向 $LOC，不是 $DST；shebang 没改干净" >&2; exit 1 ;; esac

"$DST/bin/python" -m pip uninstall -y triton || true
rm -rf "$DST"/lib/python*/site-packages/triton*
"$DST/bin/python" -m pip install --no-deps "triton-ascend==$VERSION" --extra-index-url "$INDEX"
"$DST/bin/python" -m pip install pybind11          # backends/ascend/utils.py 需要

# 验收：有昇腾驱动，且 inductor 真能编出前反向内核（不是只 import 成功）
"$DST/bin/python" - <<'PY'
import torch, torch_npu  # noqa: F401
from triton.runtime import driver
assert driver.active is not None, "triton 没有可用后端"
print("triton driver:", driver.active)
m = torch.nn.Linear(64, 64).npu()
x = torch.randn(8, 64, device="npu", requires_grad=True)
torch.compile(lambda t: torch.nn.functional.silu(m(t)).sum(), backend="inductor")(x).backward()
assert x.grad is not None
print("inductor 前反向编译通过")
PY
echo "TRITON 环境就绪：$DST（对照 constraints/npu-triton.txt）"
