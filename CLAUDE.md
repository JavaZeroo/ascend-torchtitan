# ascend-torchtitan — Development Guide

Out-of-tree Ascend NPU extension for [torchtitan](https://github.com/pytorch/torchtitan).
We **extend, never fork**: torchtitan is an installed dependency pinned by commit SHA.

Read first: `docs/PRINCIPLES.md` (P0–P7, cited in review), `docs/glossary.md`
(override ≠ OverrideDefinitions, recipe ≠ torchtitan_recipes), `docs/roadmap.md` (M0–M5).

## Layout

| path | layer | what lives here |
|---|---|---|
| `ascend_titan/_bootstrap.py`, `train.py` | entry | `setup()` = the only side effects; must run before any `import torchtitan` |
| `ascend_titan/compat/` | L0 | shim registry + `shims/` (governed monkeypatches; count → 0) |
| `ascend_titan/kernels/` | L1 | `@override` factories wrapping Ascend kernels |
| `ascend_titan/parallel/`, `graph/` | L2 | parallel strategies, torchair |
| `ascend_titan/recipes/` | L3 | `Trainer.Config` = upstream registry fn + deltas |
| `ascend_titan/tools/` | L4 | `doctor` (env probe), provenance |
| `constraints/` | — | pinned torchtitan SHA + pip constraints (**source of truth for versions**) |
| `docs/capability-matrix.md` | — | 🟢/🔴/⚪ per feature, with attribution |
| `docs/upstream-tracking.md` | — | shim ↔ issue table, upstream asks |
| `docs/issues/` | — | problem lists by owner (torch_npu / pytorch / torchtitan / ours) — paste-ready |
| `docs/baseline.md` | — | validated version tuples (NEXT / STABLE) and M1 numbers |
| `tests/assets/losses/npu/` | — | frozen golden loss curves (`scripts/check_golden.sh`) |

The sibling checkout `../torchtitan` is the pinned upstream; read it freely, never edit it.

## Commands

```bash
WITH_TORCH=1 ./scripts/install.sh         # torch+torch_npu (NEXT track) + torchtitan @ pinned SHA + this pkg
CONSTRAINTS=constraints/npu-stable.txt WITH_TORCH=1 ./scripts/install.sh   # STABLE track
ascend-titan-doctor                       # env probe (works on CPU)
pytest tests/unit -x                      # CPU unit tests (titan-marked tests need torchtitan)
ruff check . && ruff format --check .
./scripts/probe_compat.sh                 # how far can the pinned SHA move forward
MODULE=ascend_titan.recipes.qwen3 CONFIG=qwen3_debugmodel_npu NPU=8 ./scripts/run_train.sh
COMM_MODE=fake_backend NPU=8 ./scripts/run_train.sh   # 1 device, fake PG (currently 🔴 NPU-2)
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 ./scripts/check_golden.sh qwen3_debugmodel_npu   # deterministic vs golden
python -m ascend_titan.tools.matrix --cards 0-7 --jobs 4   # sweep upstream test configs on NPU (docs/matrix/)
# tyro: subcommands like `activation-checkpoint:none` must come LAST, after all --flags
```

## Hard rules

1. **`import ascend_titan` has no side effects** (`tests/unit/test_import_purity.py`). torchtitan
   imports `ascend_titan.kernels.*` from inside `Trainer.__init__`; import-time side effects would
   fire mid-initialisation.
2. **Never shim around torch_npu** (P1). Attribution `NPU` ⇒ file an issue, mark 🔴, stop.
3. **Config before shim** (P0). Check `torchtitan/config/configs.py` for a switch first.
4. **Shims wrap** (P3) and **carry an upstream issue** (P4); the registry enforces both.
5. **Overrides target existing `Configurable` nodes only** (P6). No node ⇒ upstream ask, not a
   parent-block replacement.
6. **Recipes are deltas** (`tests/unit/test_recipes.py::test_recipe_is_delta_not_copy`).
7. **Version bumps are PRs with a matrix run** (P5). Never edit `constraints/torchtitan.sha` casually.
8. **No speculative fallbacks.** Same rule as upstream: only validate explicit contracts.

## Attributing a failure (used constantly)

| traceback's first non-framework frame | code | action |
|---|---|---|
| `torch_npu/...` | NPU | torch_npu issue, 🔴, **no workaround** |
| `torchtitan/...` near `cuda`/`nccl` strings | TT | upstream issue + wrap-shim candidate |
| CANN error code (`EZ9999`, `EI0002`, …) | CANN | record, stop |
| `attn_gym`/`helion`/`deep_ep`/`cutlass` | DEP | record; Ascend replacement is an L1 task |
| `torch/` itself (device whitelist, missing public API) | TORCH | pytorch issue; polyfill shim only for a pure rename/alias |

## Validating numerics
Same bar as upstream: non-computation changes must give **identical** loss with
`--debug.seed=42 --debug.deterministic`; computation changes (kernels) need an alignment test
against upstream's eager path plus `torch.library.opcheck`. Never use `--debug.deterministic_warn_only`.

## Dev container
`ascend-titan-dev` on the NPU host (image cann:9.1.0-910b-ubuntu22.04-py3.12-devel): system python = NEXT track, `/opt/venv213` was the probe venv. `/data` is the shared NFS; the repo lives at the same path inside. All 8 cards are usually shared with other jobs — pick cards with `ASCEND_RT_VISIBLE_DEVICES`.

## Skills
`.claude/skills/`: `compat-probe` (M0), `shim-authoring`, `override-authoring`,
`capability-matrix` (record a cell + attribution), `upstream-sync` (bump the SHA).
Rules in `.claude/rules/` apply by path.
