# kernels（L1）—— 融合算子 override

每个文件把一个昇腾算子（AscendC / Triton-Ascend / torch_npu 融合算子）封装成 torchtitan 的 `@override` 工厂，目标是一个**已存在**的 `Configurable.Config` 节点。

规则：
- 只针对上游已有的节点（设计文档 §"Configurable 节点判据"）。计算没有节点时**不要**替换父块；在 `docs/upstream-tracking.md` 记为上游 ask。
- 自定义内核用 `torch.library.custom_op` 注册，带 `register_fake` 与 `register_autograd`；附 `opcheck` 测试。
- 依赖缺失 ⇒ 打响亮的 WARNING 且不注册（上游 eager 路径就是回退）。绝不在没有日志和 provenance 的情况下静默换成慢路径。

现有：`attention.py`（`npu_fusion_attention`，替换 VarlenAttention）、`rope.py`（ComplexRoPE 的实数缓存实现）。
M3+ 计划：`situ_glu.py`、`kda.py`、`causal_conv1d.py`、`rmsnorm.py`。
