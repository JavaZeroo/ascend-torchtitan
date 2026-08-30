---
description: L1 override 模块规则
globs: ascend_titan/kernels/**
---
# override 规则
- **基础依赖硬导入（P14 / ADR-007）**：`import torch`、`import torch_npu`、`import torchtitan.*` 一律直写，**绝不 try**、不设 `_AVAILABLE` 开关。算子探测统一用 `from ascend_titan.kernels._probe import require_op, torch_npu`；`require_op("npu_xxx")` 缺失时抛 `MissingNpuOpError`（算子缺失 = 昇腾侧缺口，走 P9，不在本仓降级）。
- 真正可选的加速包（ops-nn `cann_ops_nn`、Triton-Ascend——需要单独构建、不在基线内）才用 `_probe.optional_module(*candidates)` + WARNING 降级（ADR-004），且 WARNING 必须写明降级了什么。
- CPU 单测用 `tests/conftest.py` 的 `npu_stub` fixture 提供假 `torch_npu`；负面用例用 `no_torch_npu` / `npu_stub_missing_op`。
- 只针对已有的上游 `Configurable.Config`（P6）。docstring 里引用节点所在的上游 file:line。
- 用 `derive(cfg, NewConfig, **deltas)` 构造替换配置；绝不手工逐字段复制。
- 按实例选择通过 `@override` 的 `fqns=` 表达，不在工厂里检查字段。
- 自定义内核用 `torch.library.custom_op` 封装，带 `register_fake` 与 `register_autograd`；附 `opcheck` 测试和对上游 eager 模块的对齐测试。
- 替换实现若改变参数布局，用 `register_state_dict_post_hook` / `register_load_state_dict_pre_hook` 桥接 checkpoint（见上游 `torchtitan/overrides/fused_swiglu.py`）。
- 写第一个 override 前读 `../torchtitan/torchtitan/overrides/README.md`。
