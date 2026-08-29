# 贡献指南

感谢你帮助 torchtitan 在昇腾上跑好。先读 `docs/PRINCIPLES.md`——评审按 P0–P7 编号引用。

## 基本规则
- **绝不修改 torchtitan。** 它是按 commit 固定的已安装依赖（`constraints/torchtitan.sha`）。上游改动请提到 pytorch/torchtitan。
- **先归因再修。** 每个失败都有代码（TT / NPU / CANN / DEP / TORCH，见 `CLAUDE.md`）。`NPU` 类失败提 issue，不做 workaround（P1）。
- **shim 是最后手段。** 先找配置开关（P0），再考虑 override（P6），两者都够不着才用 shim——包装或 polyfill，绝不复制一份（P3），并且必须挂上游链接（P4）。
- **recipe 是增量**，建立在上游 `config_registry` 函数之上。

## 工作流
```bash
./scripts/install.sh            # 固定 SHA 的 torchtitan + 本包
pre-commit install
pytest tests/unit -x            # CPU；push 前必须通过
ASCEND_RT_VISIBLE_DEVICES=0 pytest tests/npu -x   # 有 NPU 时
```
- 从 `main` 拉分支；一个 PR 一个主题；diff 保持可评审。
- PR 模板会问：修了什么、归因是什么、矩阵格子变化、shim 增减、验证过的版本元组。
- torchtitan SHA 升级单独成 PR（`.claude/skills/upstream-sync`），不要夹在功能 PR 里。

## 想做什么，读什么
| 你想 | 读 |
|---|---|
| 修一个 NPU 上的失败 | `.claude/skills/capability-matrix`，然后 `shim-authoring` |
| 加一个融合算子 | `.claude/skills/override-authoring`、`ascend_titan/kernels/README.md` |
| 加一个模型 recipe | `.claude/rules/recipes.md` |
| 升级 torchtitan | `.claude/skills/upstream-sync` |

## 代码风格
`ruff`（配置在 `pyproject.toml`），公开函数写类型标注，docstring 说*为什么*并引用所依赖的上游 file:line。

## 报告问题
使用 issue 模板。NPU 上的失败请附 `ascend-titan-doctor --json` 输出和 traceback 中第一个非框架帧。
