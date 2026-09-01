"""The case set: upstream's own integration-test lists, read from the pinned checkout.

We clone the case set instead of vendoring it (``docs/matrix/README.md``), so a new
upstream test shows up in our matrix automatically.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Case:
    suite: str
    name: str
    descr: str
    ngpu: int
    configs: list[str]  # encoded matrix config names
    override_args: list[list[str]]
    skip: str | None = None  # reason, if not runnable by construction


def fused_is_a_no_op(config_name: str) -> bool:
    """True when no fused kernel targets any node of this config.

    Such a case would run, succeed, and report a number identical to ``minimal`` --
    a fused measurement of nothing (P7/P12), which is why the caller skips it.

    A config that cannot be built here is *not* reported as a no-op: we simply do
    not know, so the case runs and the runner attributes whatever it hits.
    """
    from ascend_titan.recipes.matrix import SEP, npu_fused, resolve

    suffix = SEP + "fused"
    base = config_name[: -len(suffix)] if config_name.endswith(suffix) else config_name
    try:
        config = resolve(base)()
    except Exception:  # noqa: BLE001 - undecidable here; let the case run
        return False
    return npu_fused(config).is_noop


def load_cases(titan_dir: Path, suites: list[str], *, mode: str = "minimal") -> list[Case]:
    sys.path.insert(0, str(titan_dir))
    from ascend_titan.recipes.matrix import encode

    builders = {
        "features": ("tests.integration_tests.features", "build_features_test_list"),
        "models": ("tests.integration_tests.models", "build_model_tests_list"),
        "h100": ("tests.integration_tests.h100", "build_h100_tests_list"),
    }
    cases: list[Case] = []
    for suite in suites:
        mod_name, fn_name = builders[suite]
        mod = __import__(mod_name, fromlist=[fn_name])
        for d in getattr(mod, fn_name)():
            skip = None
            if d.disabled:
                skip = "disabled upstream"
            elif not d.configs:
                skip = "legacy override_args-only case (no config fn)"
            elif d.required_cuda_capabilities:
                skip = f"TT-GATE: requires CUDA capability {list(d.required_cuda_capabilities)}"
            elif mode == "fused" and all(fused_is_a_no_op(encode(fn)) for fn in d.configs):
                # ⚪ not 🔴: nothing is broken, the case is simply not a fused
                # measurement (P2 -- "not applicable" may not look like "failed").
                skip = "fused 无可融合节点：与 minimal 完全等价，不作为融合结果上报"
            cases.append(
                Case(
                    suite=suite,
                    name=d.test_name,
                    descr=d.test_descr,
                    ngpu=d.ngpu,
                    configs=[encode(fn, mode=mode) for fn in d.configs],
                    override_args=[list(a) for a in d.override_args],
                    skip=skip,
                )
            )
    return cases
