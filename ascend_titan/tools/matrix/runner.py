"""Execution: schedule cases across NPU cards and run them through run_train.sh.

One card pool, ``--jobs`` concurrent cases, big cases first (like upstream). A
card is never handed to two cases at once: two HCCL jobs on one card collide
(EI0020), which the triage table knows as HARNESS.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ascend_titan.tools.matrix.cases import Case
from ascend_titan.tools.matrix.triage import triage


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


class CardPool:
    def __init__(self, cards: list[int]):
        self._free = list(cards)
        self._cv = threading.Condition()

    def acquire(self, n: int) -> list[int]:
        with self._cv:
            while len(self._free) < n:
                self._cv.wait()
            got, self._free = self._free[:n], self._free[n:]
            # ASCEND_RT_VISIBLE_DEVICES must be ascending: with an unsorted list
            # (e.g. "4,5,0,1", which this pool produces after a release) torch_npu
            # reports device_count()==0 and torch.npu.is_available() False, and the
            # run dies far away with a confusing device_type error. Measured on
            # 910B2 / CANN 9.1.0; also filed as NPU-10.
            return sorted(got)

    def release(self, cards: list[int]) -> None:
        with self._cv:
            self._free.extend(cards)
            self._cv.notify_all()


# Above HCCL's default base port and above this host's ephemeral range.
_HCCL_PORT_BASE = 61000
_HCCL_PORTS_PER_CARD = 8


def hccl_base_port(cards: list[int]) -> int:
    """A per-card-set HCCL base port, so concurrent cases never collide."""
    return _HCCL_PORT_BASE + min(cards) * _HCCL_PORTS_PER_CARD


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
                # HCCL binds a listening socket at HCCL_IF_BASE_PORT (default 60000) on
                # the host IP. On a shared box that collides with somebody else's job --
                # "The IP address ... and port 60001 have already been bound" -- and the
                # case dies before step 1 (attributed HARNESS). Give each card set its
                # own range, above the default, so our cases never collide with each
                # other and we do not fight the next person either.
                "HCCL_IF_BASE_PORT": str(hccl_base_port(cards)),
                "MODULE": "ascend_titan.recipes.matrix",
                "CONFIG": cfg,
                "NPU": str(case.ngpu),
                "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, cards)),
                "LOG_RANK": ",".join(map(str, range(case.ngpu))),
                "TORCHTITAN_TEST_OUTPUT_DIR": str(test_out),
                # The cases must run on the interpreter this tool runs on. Left to
                # PATH, ``run_train.sh`` picks up whatever ``torchrun`` the ambient
                # shell has, which on this box is the system python -- no
                # Triton-Ascend, and every case comes back red with "0 active
                # drivers". A whole sweep of false HARNESS reds, measured
                # 2026-09-02. Name the interpreter instead of hoping for it.
                "PYTHON": sys.executable,
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
                f"[{res.state:5}] {c.suite}/{c.name} ngpu={c.ngpu} "
                f"{res.code} {res.note} ({res.seconds}s)",
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
