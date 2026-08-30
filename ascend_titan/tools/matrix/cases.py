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
