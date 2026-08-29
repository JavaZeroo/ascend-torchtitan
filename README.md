# ascend-torchtitan

Out-of-tree **Ascend NPU** extension for [torchtitan](https://github.com/pytorch/torchtitan).
Install it next to a pinned torchtitan and torchtitan runs on NPUs; opt into Ascend fused kernels,
parallel strategies and graph mode per recipe. torchtitan itself is never forked or patched on disk.

> Status: **M0/M1 scaffold**. Nothing has been validated on an NPU yet — see
> `docs/capability-matrix.md` (all ⚪) and `docs/roadmap.md`.

## Install

```bash
git clone <this repo> ascend-torchtitan && cd ascend-torchtitan
TITAN_DIR=../torchtitan ./scripts/install.sh    # torchtitan @ pinned SHA, no CUDA-only extras
ascend-titan-doctor                              # prints the version tuple and what is missing
```

## Run

```bash
MODULE=ascend_titan.recipes.qwen3 CONFIG=qwen3_debugmodel_npu NPU=8 ./scripts/run_train.sh
COMM_MODE=fake_backend NPU=8 ./scripts/run_train.sh        # single device, fake process groups
```

`python -m ascend_titan.train` is `python -m torchtitan.train` with `ascend_titan.setup()` run
first (torch_npu import + compat shims). Everything else is upstream.

## How it is put together

| layer | package | mechanism |
|---|---|---|
| L0 compat | `ascend_titan.compat` | governed monkeypatches, each tied to an upstream issue — count should reach zero |
| L1 kernels | `ascend_titan.kernels` | torchtitan `@override` factories wrapping AscendC / Triton-Ascend kernels |
| L2 parallel / graph | `ascend_titan.parallel`, `.graph` | `ModelSpec.parallelize_fn` replacements, torchair backend |
| L3 recipes | `ascend_titan.recipes` | upstream config + deltas + `override.imports` |
| L4 tools | `ascend_titan.tools` | `doctor`, provenance |

Design: `docs/design/2026-08-29-ascend-torchtitan-design.md`. Principles: `docs/PRINCIPLES.md`.
Decisions: `docs/adr/`. Working with Claude Code: `CLAUDE.md` and `.claude/skills/`.

## Develop

```bash
pytest tests/unit -x          # CPU
ruff check .
./scripts/probe_compat.sh     # how far forward the torchtitan pin can move
```
