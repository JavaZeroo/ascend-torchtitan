---
description: Rules for L3 recipes
globs: ascend_titan/recipes/**
---
# Recipe rules
- A recipe calls the upstream `config_registry` function and mutates the result. Never construct
  `Trainer.Config(...)` from scratch (`test_recipe_is_delta_not_copy`).
- Every delta gets a `# DELTA n:` comment naming the upstream default it changes and the matrix
  cell it corresponds to.
- Overrides are activated in the recipe via `config.override.imports = [...]`, so the assembly is
  readable in one place and auditable in the override log.
- Keep the `validated:` header line; CI rewrites it, humans don't.
