---
description: L1 override 模块规则
globs: ascend_titan/kernels/**
---
# override 规则
- 模块在没装算子依赖时也必须能 import：依赖探测放在模块级 try/except 里，打 WARNING，失败时跳过 `@override` 注册（ADR-004）。绝不在 import 时抛错。
- 只针对已有的上游 `Configurable.Config`（P6）。docstring 里引用节点所在的上游 file:line。
- 用 `derive(cfg, NewConfig, **deltas)` 构造替换配置；绝不手工逐字段复制。
- 按实例选择通过 `@override` 的 `fqns=` 表达，不在工厂里检查字段。
- 自定义内核用 `torch.library.custom_op` 封装，带 `register_fake` 与 `register_autograd`；附 `opcheck` 测试和对上游 eager 模块的对齐测试。
- 替换实现若改变参数布局，用 `register_state_dict_post_hook` / `register_load_state_dict_pre_hook` 桥接 checkpoint（见上游 `torchtitan/overrides/fused_swiglu.py`）。
- 写第一个 override 前读 `../torchtitan/torchtitan/overrides/README.md`。
