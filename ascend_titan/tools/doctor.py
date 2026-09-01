"""Environment probe: the M0 tool.

``ascend-titan-doctor`` or ``python -m ascend_titan.tools.doctor``.

Reports the four-way version tuple (torch / torch_npu / CANN / torchtitan),
whether torch autoloads torch_npu, and which CUDA-only dependencies leaked into
the environment. Exit code is 0 unless ``--strict`` and a hard requirement is
missing. Runs fine on a CPU box (it will simply say so).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field


@dataclass
class Report:
    python: str = sys.version.split()[0]
    torch: str | None = None
    torch_npu: str | None = None
    torch_npu_autoload: bool | None = None
    npu_available: bool | None = None
    npu_count: int | None = None
    cann_version: str | None = None
    torchtitan: str | None = None
    torchtitan_sha: str | None = None
    torchtitan_importable: bool | None = None
    torchtitan_import_error: str | None = None
    cuda_only_packages: list[str] = field(default_factory=list)
    inductor_backend: str | None = None
    """triton 的昇腾后端；None = inductor 用不了（torch.compile 会报 0 active drivers）。"""
    shims_registered: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _dist_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _cann_version() -> str | None:
    """CANN toolkit version. Layout (CANN 9.x): $ASCEND_HOME_PATH/compiler/version.info
    holds ``Version=...``; older toolkits ship version.cfg at the root."""
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        return None
    for rel in ("compiler/version.info", "version.cfg", "../version.cfg"):
        p = os.path.join(home, rel)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    if line.lower().startswith("version"):
                        return line.strip().split("=", 1)[-1]
    return f"(ASCEND_HOME_PATH={home}, no version file found)"


def _git_sha(pkg_name: str) -> str | None:
    try:
        mod = importlib.import_module(pkg_name)
    except Exception:
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def probe(*, import_torchtitan: bool = True) -> Report:
    r = Report()
    r.torch = _dist_version("torch")
    r.torch_npu = _dist_version("torch_npu")
    r.cann_version = _cann_version()

    try:
        import torch

        pre = torch._C._get_privateuse1_backend_name()
        r.torch_npu_autoload = pre == "npu"
        if r.torch_npu and not r.torch_npu_autoload:
            try:
                importlib.import_module("torch_npu")
            except Exception as e:  # noqa: BLE001
                r.notes.append(f"torch_npu import failed: {e!r}")
        if hasattr(torch, "npu"):
            r.npu_available = bool(torch.npu.is_available())
            r.npu_count = torch.npu.device_count() if r.npu_available else 0
    except Exception as e:  # noqa: BLE001
        r.notes.append(f"torch import failed: {e!r}")

    cuda_only = (
        "nvidia-cutlass-dsl",
        "flash-attn-4",
        "nvidia-cudnn-cu12",
        "nvidia-cudnn-cu13",
    )
    for pkg in cuda_only:
        if _dist_version(pkg):
            r.cuda_only_packages.append(f"{pkg}=={_dist_version(pkg)}")

    # inductor 的昇腾后端。缺了它 torch.compile 报 "0 active drivers"——这是环境坏了，
    # 不是已知限制（装法：scripts/install_triton.sh）。
    try:
        from triton.runtime import driver

        # driver.active 是个惰性代理：碰一下属性才会真正构造驱动，
        # 而构造本身就是"这台机器有没有可用后端"的答案。
        target = driver.active.get_current_target()
        r.inductor_backend = getattr(target, "backend", None) or str(target)
    except Exception as e:  # noqa: BLE001 - 报告缺什么正是 doctor 的职责
        r.notes.append(f"triton has no usable backend ({type(e).__name__}); inductor unavailable")

    r.torchtitan = _dist_version("torchtitan")
    if import_torchtitan:
        try:
            importlib.import_module("torchtitan.trainer")
            r.torchtitan_importable = True
            r.torchtitan_sha = _git_sha("torchtitan")
        except Exception as e:  # noqa: BLE001
            r.torchtitan_importable = False
            r.torchtitan_import_error = f"{type(e).__name__}: {e}"

    try:
        from ascend_titan.compat import list_shims

        r.shims_registered = [s.name for s in list_shims()]
    except Exception as e:  # noqa: BLE001
        r.notes.append(f"shim discovery failed: {e!r}")

    if r.torch_npu is None:
        r.notes.append("torch_npu not installed: this is not an NPU environment.")
    if r.torch_npu is not None and r.inductor_backend is None:
        r.notes.append(
            "no Ascend triton backend: torch.compile(backend='inductor') will fail with "
            "'0 active drivers'. Install it with scripts/install_triton.sh."
        )
    if r.cuda_only_packages:
        r.notes.append(
            "CUDA-only packages present; torchtitan was probably installed with the "
            "attn-gym[linear] extra. See constraints/README.md."
        )
    return r


def render(r: Report) -> str:
    rows = [
        ("python", r.python),
        ("torch", r.torch),
        ("torch_npu", r.torch_npu),
        ("torch_npu autoload", r.torch_npu_autoload),
        ("npu available / count", f"{r.npu_available} / {r.npu_count}"),
        ("CANN", r.cann_version),
        ("torchtitan", r.torchtitan),
        ("torchtitan sha", r.torchtitan_sha),
        ("torchtitan importable", r.torchtitan_importable),
        ("inductor backend", r.inductor_backend or "-（缺 Triton-Ascend）"),
        ("cuda-only packages", ", ".join(r.cuda_only_packages) or "-"),
        ("shims registered", ", ".join(r.shims_registered) or "-"),
    ]
    width = max(len(k) for k, _ in rows)
    out = ["ascend-titan doctor", "=" * (width + 30)]
    out += [f"{k:<{width}}  {v}" for k, v in rows]
    if r.torchtitan_import_error:
        out.append(f"torchtitan import error: {r.torchtitan_import_error}")
    out += [f"note: {n}" for n in r.notes]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 1 if torch_npu/torchtitan missing")
    p.add_argument("--no-titan", action="store_true", help="do not import torchtitan")
    a = p.parse_args(argv)
    r = probe(import_torchtitan=not a.no_titan)
    print(json.dumps(asdict(r), indent=2) if a.json else render(r))
    if a.strict and (r.torch_npu is None or r.torchtitan_importable is False):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
