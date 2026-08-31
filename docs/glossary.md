# 术语表

| 术语 | 本仓含义 | 别和它混淆 |
|---|---|---|
| **override** | torchtitan 的 `@override` 机制（`torchtitan/config/override.py`）：在配置期替换某个 `Configurable.Config` 节点。我们的 L1 层。 | `OverrideDefinitions`——上游*集成测试*的用例数据类。 |
| **shim** | 用 `@shim` 注册在 `ascend_titan.compat` 的受治理 monkeypatch。我们的 L0 层。 | torchtitan 的 `ModelConfigConverter`（float8/LoRA），那是配置树变换。 |
| **recipe** | `ascend_titan/recipes/` 里返回 `Trainer.Config` 的函数，写法是*上游 registry 函数 + 增量*。 | `torchtitan_recipes`——上游测试配置包。 |
| **pinned SHA** | `constraints/torchtitan.sha` 里的 torchtitan commit。 | torchtitan 的 PyPI release（已陈旧）。 |
| **matrix / 矩阵** | `docs/capability-matrix.md`，每格三态。 | 上游 CI 的用例列表。 |
| **provenance** | 一次运行中每个可 override 节点*实际生效的后端*表。 | override 的日志行（那只是它的输入之一）。 |
| **NIGHTLY** | torch nightly + torch_npu master 源码构建 + torchtitan main（`constraints/nightly.txt` + 两个 `.sha`）。唯一的 track。 | torchtitan 自己的 nightly/stable wheel。 |
| **golden** | `tests/assets/losses/npu/` 里冻结的确定性 loss/grad_norm 曲线，由 `scripts/check_golden.sh` 校验。 | 上游的 `tests/assets/losses/*_a10g.txt`（GPU，配置不同）。 |
| **L0–L4** | compat / kernels / parallel+graph / recipes / tools。见设计文档。 | |
| **M0–M5** | 里程碑。见 docs/roadmap.md。 | |
