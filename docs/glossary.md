# Glossary

| term | meaning here | not to be confused with |
|---|---|---|
| **override** | torchtitan's `@override` mechanism (`torchtitan/config/override.py`): swap a `Configurable.Config` node at config time. Our L1 layer. | `OverrideDefinitions` — upstream's *integration-test* case dataclass. |
| **shim** | A governed monkeypatch registered with `@shim` in `ascend_titan.compat`. Our L0 layer. | torchtitan's `ModelConfigConverter` (float8/LoRA), which is a config-tree transform. |
| **recipe** | A function in `ascend_titan/recipes/` returning a `Trainer.Config`, built as *upstream registry fn + deltas*. | `torchtitan_recipes` — upstream's package of test configs. |
| **pinned SHA** | The torchtitan commit in `constraints/npu.txt`. | torchtitan PyPI releases (stale). |
| **matrix** | `docs/capability-matrix.md`, three-state per cell. | upstream CI's test list. |
| **provenance** | The table of *which backend actually ran* for each overridable node in a given run. | the override log lines (those are one input to it). |
| **L0–L4** | compat / kernels / parallel+graph / recipes / tools. See design doc. | |
| **M0–M5** | milestones. See docs/roadmap.md. | |
