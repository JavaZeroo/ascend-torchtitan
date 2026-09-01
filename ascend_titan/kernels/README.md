# kernels（L1）—— 融合算子 override

每个文件把一个昇腾算子（AscendC / Triton-Ascend / torch_npu 融合算子）封装成 torchtitan 的 `@override` 工厂，目标是一个**已存在**的 `Configurable.Config` 节点。

规则：
- 只针对上游已有的节点（设计文档 §"Configurable 节点判据"）。计算没有节点时**不要**替换父块；在 `docs/upstream-tracking.md` 记为上游 ask。
- 自定义内核用 `torch.library.custom_op` 注册，带 `register_fake` 与 `register_autograd`；附 `opcheck` 测试。
- **`torch_npu` 是基础依赖（P14 / ADR-007）**：模块无条件 `import torch_npu`，算子探测走 `_probe.require_op()`；缺模块或缺算子一律在 import 处抛错，绝不降级。
- 只有真正可选的加速包（ops-nn `cann_ops_nn`、Triton-Ascend）用 `_probe.optional_module()` + 响亮 WARNING 降级（ADR-004），且必须记 provenance。绝不在没有日志和 provenance 的情况下静默换成慢路径。

| 文件 | 算子 | 目标节点 |
|---|---|---|
| `_probe.py` | —— | 依赖探测：`require_op()`（硬）/ `optional_module()`（可选包） |
| `attention.py` | `npu_fusion_attention(_grad)` | `VarlenAttention.Config` |
| `rope.py` | `npu_rotary_mul` | `ComplexRoPE.Config`（实数缓存）、`CosSinRoPE.Config` |
| `rms_norm.py` | `npu_rms_norm` | `RMSNorm.Config` |
| `swiglu.py` | `npu_swiglu` | `FeedForward.Config`（上游 FusedSwiGLU 布局） |
| `situ_glu.py` | ops-nn `situ_glu` | `KimiFeedForward.Config`（kimi_k3） |
| `gdn.py` | 纯 torch chunk 递推 | `GatedDeltaKernel.Config` + `InnerGatedDeltaNet.Config`（qwen3_5，单一 override 认领两个叠置节点） |
| `gdn_fla.py` | fla-npu AscendC 融合（R5，opt-in） | 同上节点（兄弟 override，与 `gdn.py` 互斥） |
| `kda.py` | attn_gym reference + 自研 causal conv1d | `KDAKernel.Config` + `InnerKDA.Config`（kimi_k3） |
