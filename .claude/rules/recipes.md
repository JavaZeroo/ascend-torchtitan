---
description: L3 recipe / 模型包规则
globs: ascend_titan/models/**, ascend_titan/recipes/**
---
# recipe 规则
- **内容在 `models/`，机制在 `recipes/`**：每个模型一个包 `models/<model>/{__init__,recipes,[probes]}.py + README.md`；`recipes/` 只放跨模型的 `deltas.py`（原语）与 `matrix.py`（矩阵）。新模型从 `models/_template/` 复制，并在 `models/registry.py` 登记（`tests/unit/test_models_registry.py` 强制 README + 登记）。
- **recipe 与探针分开**：`recipes.py` 是给人跑的支持入口，`probes.py` 是只为矩阵测量、预期可能 🔴 的配置。别混。
- **debug 用的配置不进仓库**：为查一个具体故障临时凑的配置（调 lr 看会不会发散这类）是调试脚手架，不是测量基线——放 `outputs/`（已 gitignore），**结论**写进模型 README 或 `docs/issues/`，配置本身不提交。`probes.py` 只放矩阵长期要测的格子。
- 命名 `<model>_<flavor>_npu[_<variant>]`；`<flavor>` 取上游 config registry 的名字。
- recipe 调用上游 `config_registry` 函数并修改其结果。绝不从零构造 `Trainer.Config(...)`（`test_recipe_is_delta_not_copy`）。
- 每个增量加一条 `# DELTA n:` 注释，写明它改变的上游默认值和对应的矩阵格子。
- override 在 recipe 里通过 `config.override.imports = [...]` 激活，装配一处可见，并可在 override 日志里审计。目标路径**只用** `from ascend_titan.kernels import *_OVERRIDE` 常量，不写字符串字面量——常量唯一定义在 `kernels/__init__.py`（P11，`tests/unit/test_override_paths.py` 强制）。
- **recipe 用原语，不用通用变换**：recipe 从 `recipes/deltas.py` 取 `add_override` / `swap_override` / `flex_to_varlen`，一条 DELTA 一次调用。`npu_minimal` / `npu_fused` 住在 `recipes/matrix.py`（矩阵是它们唯一的调用者，给没有 recipe 的上游配置用），**recipe 里不许出现**（`test_recipes_spell_out_their_own_deltas`）。矩阵口径不变：minimal 只允许"不加就跑不起来"的内容、挂 issue ID、用特性探测决定何时消失；性能项永远只进 fused（P12）。绝不用任何一个绕过 torch_npu 缺陷（P9）。
- 保留 `validated:` 头行；CI 改写它，人不改。
