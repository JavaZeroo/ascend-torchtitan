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

The sibling checkout `../torchtitan` is the pinned upstream; read it freely, never edit it.

## Commands

```bash
./scripts/install.sh                      # torchtitan @ pinned SHA, no CUDA extras, + this pkg
ascend-titan-doctor                       # env probe (works on CPU)
pytest tests/unit -x                      # CPU unit tests (titan-marked tests need torchtitan)
ruff check . && ruff format --check .
./scripts/probe_compat.sh                 # how far can the pinned SHA move forward
MODULE=ascend_titan.recipes.qwen3 CONFIG=qwen3_debugmodel_npu NPU=8 ./scripts/run_train.sh
COMM_MODE=fake_backend NPU=8 ./scripts/run_train.sh   # 1 device, fake PG
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
7. **Version bumps are PRs with a matrix run** (P5). Never edit `torchtitan_sha=` casually.
8. **No speculative fallbacks.** Same rule as upstream: only validate explicit contracts.

## Attributing a failure (used constantly)

| traceback's first non-framework frame | code | action |
|---|---|---|
| `torch_npu/...` | NPU | torch_npu issue, 🔴, **no workaround** |
| `torchtitan/...` near `cuda`/`nccl` strings | TT | upstream issue + wrap-shim candidate |
| CANN error code (`EZ9999`, `EI0002`, …) | CANN | record, stop |
| `attn_gym`/`helion`/`deep_ep`/`cutlass` | DEP | record; Ascend replacement is an L1 task |

## Validating numerics
Same bar as upstream: non-computation changes must give **identical** loss with
`--debug.seed=42 --debug.deterministic`; computation changes (kernels) need an alignment test
against upstream's eager path plus `torch.library.opcheck`. Never use `--debug.deterministic_warn_only`.

## Skills
`.claude/skills/`: `compat-probe` (M0), `shim-authoring`, `override-authoring`,
`capability-matrix` (record a cell + attribution), `upstream-sync` (bump the SHA).
Rules in `.claude/rules/` apply by path.
