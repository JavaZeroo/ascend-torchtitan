---
description: L3 recipe 规则
globs: ascend_titan/recipes/**
---
# recipe 规则
- recipe 调用上游 `config_registry` 函数并修改其结果。绝不从零构造 `Trainer.Config(...)`（`test_recipe_is_delta_not_copy`）。
- 每个增量加一条 `# DELTA n:` 注释，写明它改变的上游默认值和对应的矩阵格子。
- override 在 recipe 里通过 `config.override.imports = [...]` 激活，装配一处可见，并可在 override 日志里审计。
- 通用变换放 `transforms.py`（`npu_baseline`），供矩阵扫描对任意上游配置复用。
- 保留 `validated:` 头行；CI 改写它，人不改。
