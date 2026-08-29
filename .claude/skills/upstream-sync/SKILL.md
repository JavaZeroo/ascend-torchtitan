---
name: upstream-sync
description: 安全地升级 constraints/torchtitan.sha 里固定的 torchtitan SHA——探测可 import 性、重跑 CPU 测试、跑 NPU 矩阵、更新 shim 与文档。用于 CI 的 main 腿显示漂移、需要的上游特性已落地、或定期同步时。
---
# upstream-sync

升级是一个 PR（P5）。绝不在此流程之外改 `constraints/torchtitan.sha`。

1. `./scripts/probe_compat.sh` → 候选 SHA = 最远可 import 的（或你需要的特定 commit，若它能 import）。
2. `git -C ../torchtitan log --oneline <old>..<new> -- torchtitan/config torchtitan/protocols torchtitan/models/common`——读每个触及扩展面的 commit。对每条 shim 目标和每个 override 目标确认仍存在（目标移动时 `apply_all()` 会报错；override 在 build 时失败）。
3. 更新 `constraints/torchtitan.sha`；`./scripts/install.sh`；`pytest tests/unit -x`。
4. 把 `constraints/titan-deps.txt` 与 `../torchtitan/.ci/docker/requirements.txt` 同步（去掉 `attn-gym[linear]`）。
5. NPU：`python -m ascend_titan.tools.matrix` 全量扫描；把 `docs/capability-matrix.md` 的 diff 贴进 PR。新增的 🔴 合入前必须归因。
6. 检查 `docs/upstream-tracking.md`：上游 issue 已关闭的 shim ⇒ 在本 PR 里删除。
7. PR 标题：`sync: torchtitan <old7> → <new7>`；正文 = 探测报告 + 矩阵 diff + 删除的 shim。
