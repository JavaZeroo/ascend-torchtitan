# 原则

评审时按编号引用。

| # | 原则 | 理由 |
|---|---|---|
| **P0** | **先用配置，再打补丁。** torchtitan 已有开关（`--compile.backend`、`--training.disable_cuda_graphs`、`attn_backend=`）就用开关；shim 只留给没有开关的代码。 | 每条 shim 都是负债，一行配置不是。对上游的静态分析表明大部分"写死 CUDA"的地方本来就有开关。 |
| **P1** | **绝不绕过 torch_npu 的缺陷。** 归因为 torch_npu 的失败 → 提 issue + 矩阵标 🔴。本仓不做 workaround。 | 绕过会比它所掩盖的 bug 活得更久，还会让 torch_npu 看不到真实需求。这是项目红线。 |
| **P2** | **范围收缩是排期，不是排除。** 多模态、低精度、更多模型都在路线图上。矩阵三态：🟢 / 🔴（附归因）/ ⚪ 未评估。 | "还没测"和"测了不行"绝不能混成一格。 |
| **P3** | **包装，不替换。** shim 应调用原函数、在外层加行为，从而自动继承上游变更。`kind="replace"` 必须写 `why_not_wrap`。 | 替换型 shim 会静默丢掉上游的改进，包装型不会。 |
| **P4** | **每条 shim 必须挂上游 issue。** 注册表在 import 时强制。shim 是有到期日的债务：上游修好即删。 | shim 数量是健康度指标，应趋近于零。 |
| **P5** | **版本以 commit SHA 为单位，升级走 PR。** `constraints/torchtitan.sha` 保存 torchtitan commit；升级 PR 必须附带全量矩阵结果。 | 上游 release 落后 main 半年以上，且缺少我们依赖的特性。 |
| **P6** | **只 override 上游已有的 `Configurable` 节点。** 融合算子通过 `@override` 替换现有 `Config` 节点；计算没有节点时，请上游抽一个出来，而不是替换父块。 | 替换父块等于 fork 了那个块，上游每次改动都变成我们的合并。 |
| **P7** | **响亮降级，记录 provenance。** 算子依赖缺失时退回上游 eager，打 WARNING 并记 provenance。没有 provenance 表的 benchmark 不收。 | 静默降级会污染性能数据；响亮降级保证一切可跑。 |
