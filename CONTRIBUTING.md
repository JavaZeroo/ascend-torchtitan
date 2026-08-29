# Contributing

Thanks for helping torchtitan run well on Ascend. Read `docs/PRINCIPLES.md` first — reviews cite P0–P7 by number.

## Ground rules
- **Never edit torchtitan.** It is an installed dependency pinned by commit (`constraints/torchtitan.sha`). Upstream changes go to pytorch/torchtitan.
- **Attribute before fixing.** Every failure gets a code (TT / NPU / CANN / DEP / TORCH, see `CLAUDE.md`). `NPU` failures get an issue, not a workaround (P1).
- **Shims are last resort.** Config switch first (P0), override second (P6), shim only for code neither reaches — wrapper or polyfill, never a copy (P3), always with an upstream link (P4).
- **Recipes are deltas** on the upstream `config_registry` function.

## Workflow
```bash
./scripts/install.sh            # pinned torchtitan + this package
pre-commit install
pytest tests/unit -x            # CPU; must pass before pushing
ASCEND_RT_VISIBLE_DEVICES=0 pytest tests/npu -x   # if you have an NPU
```
- Branch from `main`; one topic per PR; keep the diff reviewable.
- PR template asks for: attribution of what you fixed, matrix cells changed, shims added/removed, validated tuple.
- A torchtitan SHA bump is its own PR (`.claude/skills/upstream-sync`), never folded into a feature PR.

## Adding things
| you want to | read |
|---|---|
| fix a failure on NPU | `.claude/skills/capability-matrix` then `shim-authoring` |
| add a fused kernel | `.claude/skills/override-authoring`, `ascend_titan/kernels/README.md` |
| add a model recipe | `.claude/rules/recipes.md` |
| bump torchtitan | `.claude/skills/upstream-sync` |

## Code style
`ruff` (config in `pyproject.toml`), type hints on public functions, docstrings say *why* and cite the upstream file:line they depend on.

## Reporting problems
Use the issue templates. For a failure on NPU, attach `ascend-titan-doctor --json` output and the first non-framework traceback frame.
