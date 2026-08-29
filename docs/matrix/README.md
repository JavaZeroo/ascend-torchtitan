# Matrix sweeps

`python -m ascend_titan.tools.matrix` runs the **upstream** integration-test configs
(`tests/integration_tests/{features,models}.py` at the pinned torchtitan) on NPU, with
`npu_baseline` applied to each config (`ascend_titan/recipes/transforms.py`), triages every
failure to an attribution code and writes `report.md` + `results.json`.

```bash
TITAN_DIR=../torchtitan python -m ascend_titan.tools.matrix --list            # what would run
python -m ascend_titan.tools.matrix --cards 0-7 --jobs 4 --out outputs/matrix/$(date +%F)
python -m ascend_titan.tools.matrix --filter 'pp_|cp' --cards 0-3            # subset
python -m ascend_titan.tools.matrix --stock --filter default                  # upstream config untouched
```

Reports are committed here as `<date>_<track>.md` (copy of `report.md`). The
summary table in `docs/capability-matrix.md` is curated by hand from them: a sweep
tells you *which case* failed with *which code*; the matrix says what that means per axis.

Attribution codes: see `CLAUDE.md`; `TRIAGE` in `ascend_titan/tools/matrix.py` is the regex
table — when a new failure signature appears, add it there **and** to `docs/issues/`.
