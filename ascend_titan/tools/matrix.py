"""Capability-matrix sweep: run upstream integration-test configs on NPU, triage
every failure to an attribution code, and write a report.

    python -m ascend_titan.tools.matrix --suites features,models --cards 0-7 \
        --out outputs/matrix/$(date +%F)

Reads the upstream test lists (``tests/integration_tests/{features,models}.py``
at ``TITAN_DIR``) so the case set follows upstream automatically; applies
``npu_baseline`` to each config through ``ascend_titan.recipes.matrix``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---- attribution ------------------------------------------------------------
# (code, regex, note). First match wins; keep the specific ones first.
TRIAGE: list[tuple[str, str, str]] = [
    ("NPU-2", r"No backend type associated with device type npu", "fake PG lacks npu"),
    ("NPU-1", r"aten::_flash_attention_forward", "no NPU kernel for _flash_attention_forward"),
    ("TORCH-1", r"FlexAttention is only supported on", "flex device whitelist"),
    ("TT-2", r"has no attribute 'set_timeout'", "nightly-only set_timeout"),
    ("TT-1", r"No module named 'triton'", "unconditional import triton"),
    (
        "TT-8",
        r"Dimension specified as 0 but tensor has no dimensions",
        "PP step(arg_mbs=) nightly-only contract",
    ),
    (
        "DEP",
        r"No module named '(helion|fla|deep_ep|torchao|cutlass|deepep|hybridep)",
        "CUDA-only dependency missing",
    ),
    (
        "DEP",
        r"torchao is not installed",
        "torchao missing (float8/mx/nvfp4 need it; CUDA-only formats anyway)",
    ),
    (
        "NPU-OP",
        r"NPU function error: call (aclnn\w+) failed, error code is (\d+)",
        "torch_npu op-plugin kernel failure (see note)",
    ),
    (
        "TT-9",
        r"'torch\._C\.Tag' has no attribute",
        "upstream override uses a nightly-only torch.Tag (fused_mla)",
    ),
    (
        "TT-5",
        r"only supports spmd_backend='spmd_types'",
        "model requires spmd_types, which fails on NPU (TT-5)",
    ),
    ("DEP", r"`helion` is not installed|helion_rope", "helion (CUDA-only kernel DSL) missing"),
    ("DEP", r"No module named 'torchvision'", "torchvision missing (installable)"),
    (
        "COMPILE",
        r"when making fake tensor call|torch\._dynamo\.exc\.",
        "torch.compile path failed (see note)",
    ),
    ("TT-4", r"data is not allocated yet", "ChunkedLossWrapper / manual unshard backward"),
    ("TT-5", r"all parameters must be\s+DTensors", "spmd_types + fully_shard(dp_mesh_dims)"),
    (
        "TT-5",
        r"requires parallelism\.spmd_backend='spmd_types'",
        "feature requires spmd_types, which fails on NPU (TT-5)",
    ),
    (
        "TT-GATE",
        r"has_cuda_capability|only supported on Hopper|compute capability|CUDA capability",
        "explicit CUDA capability gate",
    ),
    ("DEP", r"symmetric_memory|symm_mem|_SymmetricMemory", "symmetric memory (CUDA-only)"),
    (
        "TT-CUDA",
        r"torch\.cuda\.|\bcuda\b.*(not available|is_available)|Torch not compiled with CUDA",
        "hard-coded torch.cuda",
    ),
    (
        "OURS",
        r"NotImplementedError: AscendFusionAttention",
        "limitation of the Ascend attention override",
    ),
    # ERR99999 is CANN's generic "application exception" wrapper printed on any crash -- not a CANN failure.
    ("CANN", r"\b(EZ|EI|EE|EJ)\d{4}\b|ERR(?!99999)\d{5}", "CANN error code"),
    ("NPU", r'File "[^"]*torch_npu/', "traceback frame inside torch_npu"),
    ("HANG", r"__TIMEOUT__", "exceeded per-test timeout"),
    (
        "CLI",
        r"For full helptext, run|unrecognized arguments|error: argument",
        "CLI / tyro parse error (harness)",
    ),
]
_ERR_LINE = re.compile(r"([A-Za-z_.]*(?:Error|Exception)): ([^\n]*)")


def triage(log: str) -> tuple[str, str]:
    for code, pat, note in TRIAGE:
        m = re.search(pat, log)
        if m:
            if code == "NPU-OP":
                note = f"torch_npu op-plugin: {m.group(1)} failed, error {m.group(2)}"
            return code, note
    errs = [
        f"{name}: {msg.strip()}"
        for name, msg in _ERR_LINE.findall(log)
        if "ChildFailedError" not in name and "error_file" not in msg
    ]
    return "UNKNOWN", (errs[-1][:200] if errs else "no error line found")


# ---- cases ------------------------------------------------------------------
@dataclass
class Case:
    suite: str
    name: str
    descr: str
    ngpu: int
    configs: list[str]  # encoded matrix config names
    override_args: list[list[str]]
    skip: str | None = None  # reason, if not runnable by construction


@dataclass
class Result:
    suite: str
    name: str
    ngpu: int
    state: str  # green | red | skip
    code: str = ""
    note: str = ""
    seconds: float = 0.0
    log: str = ""
    runs: list[dict] = field(default_factory=list)


def load_cases(titan_dir: Path, suites: list[str], *, stock: bool) -> list[Case]:
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
                    configs=[encode(fn, stock=stock) for fn in d.configs],
                    override_args=[list(a) for a in d.override_args],
                    skip=skip,
                )
            )
    return cases


# ---- execution --------------------------------------------------------------
class CardPool:
    def __init__(self, cards: list[int]):
        self._free = list(cards)
        self._cv = threading.Condition()

    def acquire(self, n: int) -> list[int]:
        with self._cv:
            while len(self._free) < n:
                self._cv.wait()
            got, self._free = self._free[:n], self._free[n:]
            return got

    def release(self, cards: list[int]) -> None:
        with self._cv:
            self._free.extend(cards)
            self._cv.notify_all()


def run_case(case: Case, cards: list[int], out: Path, repo: Path, timeout: int) -> Result:
    r = Result(suite=case.suite, name=case.name, ngpu=case.ngpu, state="red")
    test_out = out / case.name
    test_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    full_log = ""
    for i, (cfg, args) in enumerate(zip(case.configs, case.override_args, strict=True)):
        args = [a.replace("{test_output_dir}", str(test_out)) for a in args]
        env = os.environ.copy()
        env.update(
            {
                "MODULE": "ascend_titan.recipes.matrix",
                "CONFIG": cfg,
                "NPU": str(case.ngpu),
                "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, cards)),
                "LOG_RANK": ",".join(map(str, range(case.ngpu))),
                "TORCHTITAN_TEST_OUTPUT_DIR": str(test_out),
            }
        )
        env.pop("COMM_MODE", None)
        cmd = [str(repo / "scripts/run_train.sh"), "--dump_folder", str(test_out), *args]
        log_path = test_out / f"run{i}.log"
        with open(log_path, "w") as f:
            try:
                p = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout
                )
                rc = p.returncode
            except subprocess.TimeoutExpired:
                f.write("\n__TIMEOUT__\n")
                rc = -1
        text = log_path.read_text(errors="replace")
        full_log += text
        r.runs.append({"config": cfg, "args": args, "rc": rc, "log": str(log_path)})
        if rc != 0:
            break
    r.seconds = round(time.time() - t0, 1)
    r.log = str(test_out)
    if all(run["rc"] == 0 for run in r.runs):
        r.state = "green"
    else:
        r.code, r.note = triage(full_log)
    return r


def sweep(
    cases: list[Case], cards: list[int], out: Path, repo: Path, timeout: int, jobs: int
) -> list[Result]:
    results: list[Result] = []
    lock = threading.Lock()
    pool = CardPool(cards)
    runnable = [c for c in cases if c.skip is None and c.ngpu <= len(cards)]
    for c in cases:
        if c.skip is not None:
            results.append(Result(c.suite, c.name, c.ngpu, "skip", note=c.skip))
        elif c.ngpu > len(cards):
            results.append(
                Result(
                    c.suite, c.name, c.ngpu, "skip", note=f"needs {c.ngpu} cards, have {len(cards)}"
                )
            )
    runnable.sort(key=lambda c: -c.ngpu)  # big ones first, like upstream

    def worker(c: Case):
        got = pool.acquire(c.ngpu)
        try:
            res = run_case(c, got, out, repo, timeout)
        finally:
            pool.release(got)
        with lock:
            results.append(res)
            print(
                f"[{res.state:5}] {c.suite}/{c.name} ngpu={c.ngpu} {res.code} {res.note} ({res.seconds}s)",
                flush=True,
            )

    threads: list[threading.Thread] = []
    sem = threading.Semaphore(jobs)

    def guarded(c: Case):
        with sem:
            worker(c)

    for c in runnable:
        t = threading.Thread(target=guarded, args=(c,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results


# ---- report -----------------------------------------------------------------
def tuple_tag() -> str:
    import importlib.metadata as md

    def v(n):
        try:
            return md.version(n).split("+")[0]
        except md.PackageNotFoundError:
            return "none"

    return f"torch{v('torch')}_npu{v('torch_npu')}"


def render(results: list[Result], *, title: str) -> str:
    icon = {"green": "🟢", "red": "🔴", "skip": "⚪"}
    lines = [
        f"# {title}",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · tuple `{tuple_tag()}`",
        "",
    ]
    counts = {s: sum(1 for r in results if r.state == s) for s in ("green", "red", "skip")}
    lines.append(f"**{counts['green']} 🟢 · {counts['red']} 🔴 · {counts['skip']} ⚪ (skipped)**")
    lines.append("")
    by_code: dict[str, int] = {}
    for r in results:
        if r.state == "red":
            by_code[r.code] = by_code.get(r.code, 0) + 1
    if by_code:
        lines.append(
            "Red by attribution: " + ", ".join(f"`{k}`×{v}" for k, v in sorted(by_code.items()))
        )
        lines.append("")
    for suite in sorted({r.suite for r in results}):
        lines += [
            f"## {suite}",
            "",
            "| case | ngpu | state | attribution | note | s |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted((x for x in results if x.suite == suite), key=lambda x: x.name):
            note = r.note.replace("|", "\\|")
            lines.append(
                f"| {r.name} | {r.ngpu} | {icon[r.state]} | {r.code} | {note} | {r.seconds:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_cards(spec: str) -> list[int]:
    cards: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            cards += list(range(int(a), int(b) + 1))
        elif part:
            cards.append(int(part))
    return cards


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suites", default="features,models")
    p.add_argument("--cards", default="0-7", help="NPU ids, e.g. 0-7 or 0,1,4")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--titan-dir", default=os.environ.get("TITAN_DIR", str(repo.parent / "torchtitan"))
    )
    p.add_argument("--timeout", type=int, default=900, help="seconds per run")
    p.add_argument("--jobs", type=int, default=8, help="max concurrent cases")
    p.add_argument("--filter", default=None, help="regex on case name")
    p.add_argument("--stock", action="store_true", help="run upstream configs unmodified")
    p.add_argument("--list", action="store_true")
    p.add_argument("--retriage", default=None, help="re-run triage over an existing sweep dir")
    a = p.parse_args(argv)

    if a.retriage:
        out = Path(a.retriage)
        results = [Result(**r) for r in json.loads((out / "results.json").read_text())]
        for r in results:
            if r.state != "red":
                continue
            log = "".join(
                Path(run["log"]).read_text(errors="replace")
                for run in r.runs
                if Path(run["log"]).exists()
            )
            r.code, r.note = triage(log)
        (out / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
        (out / "report.md").write_text(render(results, title="Matrix sweep (re-triaged)"))
        print(render(results, title="Matrix sweep (re-triaged)"))
        return 0

    cases = load_cases(Path(a.titan_dir), a.suites.split(","), stock=a.stock)
    if a.filter:
        cases = [c for c in cases if re.search(a.filter, c.name)]
    if a.list:
        for c in cases:
            print(f"{c.suite:8} {c.name:45} ngpu={c.ngpu} {'SKIP: ' + c.skip if c.skip else ''}")
        return 0
    out = Path(
        a.out or repo / "outputs" / "matrix" / f"{time.strftime('%Y%m%d-%H%M')}_{tuple_tag()}"
    )
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} cases -> {out}", flush=True)
    results = sweep(cases, parse_cards(a.cards), out, repo, a.timeout, a.jobs)
    (out / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    title = f"Matrix sweep {'(stock upstream)' if a.stock else '(npu_baseline)'}"
    (out / "report.md").write_text(render(results, title=title))
    print(render(results, title=title))
    return 0


if __name__ == "__main__":
    sys.exit(main())
