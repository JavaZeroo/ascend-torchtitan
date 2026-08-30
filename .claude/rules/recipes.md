---
description: L3 recipe / 模型包规则
globs: ascend_titan/models/**, ascend_titan/recipes/**
---
# recipe 规则
- **内容在 `models/`，机制在 `recipes/`**：每个模型一个包 `models/<model>/{__init__,recipes,[probes]}.py + README.md`；`recipes/` 只放跨模型的 `transforms.py` 与 `matrix.py`。新模型从 `models/_template/` 复制，并在 `models/registry.py` 登记（`tests/unit/test_models_registry.py` 强制 README + 登记）。
- **recipe 与探针分开**：`recipes.py` 是给人跑的支持入口，`probes.py` 是只为矩阵测量、预期可能 🔴 的配置。别混。
- 命名 `<model>_<flavor>_npu[_<variant>]`；`<flavor>` 取上游 config registry 的名字。
- recipe 调用上游 `config_registry` 函数并修改其结果。绝不从零构造 `Trainer.Config(...)`（`test_recipe_is_delta_not_copy`）。
- 每个增量加一条 `# DELTA n:` 注释，写明它改变的上游默认值和对应的矩阵格子。
- override 在 recipe 里通过 `config.override.imports = [...]` 激活，装配一处可见，并可在 override 日志里审计。
- 通用变换放 `transforms.py`（`npu_baseline`），供矩阵扫描对任意上游配置复用。每条增量只允许"不加就跑不起来"的内容，注释写 issue ID，用**特性探测**（不是版本号）决定何时消失；性能 override 不进 baseline（P12）。绝不用 baseline 绕过 torch_npu 缺陷（P9）。
- 保留 `validated:` 头行；CI 改写它，人不改。
