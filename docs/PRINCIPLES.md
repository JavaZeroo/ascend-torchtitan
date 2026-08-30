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
| **P8** | **nightly-first。** 开发、验证、门禁的基线 = torch nightly（日期取自 torch_npu master 的 `requirements_<line>.txt`）+ torch_npu master 源码构建 + torchtitan main SHA（ADR-006）。只在正式版 torch 上出现、nightly 上不存在的问题**不是问题**：不写 shim、不写补丁、不记 issue。 | torchtitan 与 torch_npu 的 main 都面向 nightly；追正式版等于自己制造一层接口补丁（2026-08-30 评审：7/11 个补丁、14 个红格都是版本差）。 |
| **P9** | **torch_npu 的问题只能修，不能绕；修好才算数。** 归因 NPU 的失败走完整流程：最小复现（`tests/repro/`）→ 在 `../ascend-pytorch`（或 op-plugin）的 `fix/<ID>` 分支修 → `scripts/build_torch_npu.sh` 重建 → NPU 验证（对齐测试 / opcheck / golden）→ `patches/torch_npu/` 存 format-patch → `gitcode-pr-rfc-pipeline` 在 gitcode.com/Ascend 提 issue + PR → `STATUS.md` 记 URL → 合入后删补丁、升 `torch_npu.sha`。在 torch_npu 之外的任何位置（本仓、recipe、baseline、shim、"换个 loss"）绕过 torch_npu 缺陷都是违规。 | P1 的执行细则。没有流程的红线会被"临时绕一下"侵蚀——TT-4 被 `npu_baseline` 展开 loss 就是例子。 |
| **P10** | **上游边界。** 可操作的远端只有 `gitcode.com/ascend/*`。`github.com/pytorch/*` 只读：不提 issue、不提 PR、不评论。TT/TORCH 归因的问题先按 P8 确认在 nightly 上存在；存在则记入 `docs/issues/`，修复方案存 `patches/evidence/`，永不应用于安装路径。 | 授权范围。 |
| **P11** | **单一事实来源。** 版本元组只在 `constraints/`；问题状态只在 `docs/issues/STATUS.md`；其它文档只引用 ID，不复述状态。两处不一致时先修事实来源。 | 2026-08-30 评审在 5 个文档里发现同一前提的 3 种说法。 |
| **P12** | **baseline 最小化。** 矩阵默认施加 `npu_minimal`，只允许"不加就跑不起来"的增量，每条挂 issue ID 与消失条件（特性探测，不是版本号）；性能 override 走 `npu_fused`（opt-in）。同样适用于 recipe：能用上游默认就用上游默认。目标：`npu_minimal` = identity。 | 否则矩阵红格分不清"上游问题"与"我们的内核问题"。2026-08-30 据此删掉了 qwen3 参考 recipe 的两条版本差增量（loss、spmd_backend）。 |
| **P13** | **构建可复现，验证先于断言。** 源码构建的组件都有 `scripts/build_*.sh` + SHA 锁 + 产物元数据；不在 NFS 上构建。任何 🟢 / "已修复" 必须附命令与输出，且在 NIGHTLY 上跑过才算数。 | 正式版上"已验证"的结论在 nightly 上有一半不再适用。 |
| **P14** | **基础依赖硬导入，绝不 try。** `torch`、`torch_npu`、`torchtitan` 是基础依赖：一律 `import`，不写 `try/except ImportError`，不设 `_AVAILABLE` 开关，不因缺失而降级——缺了就在导入处抛错。`torch_npu` 缺某个算子同样抛错（走 P9 修上游），错误信息里写清楚"这是昇腾侧缺口"。唯一例外：(a) 真正可选的加速包（ops-nn 的 `cann_ops_nn`、Triton-Ascend 等需要单独构建、不在基线内的包）走 `kernels/_probe.optional_module` + WARNING（ADR-004）；(b) 诊断工具（`ascend-titan-doctor`、`tests/repro/probe_*.py`）的职责就是报告缺什么，允许探测。 | `try: import torch_npu` 把"环境装错了"变成一次安静的 eager 运行：融合算子没生效、性能回归无人发现、golden 还是绿的。本项目**只**为在 NPU 上跑 torchtitan 而存在，没有"没有昇腾后端也能用"的模式。见 ADR-007。 |
