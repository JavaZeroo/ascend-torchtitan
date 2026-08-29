# ADR-003：按 commit SHA 固定 torchtitan

## 状态
已采纳（2026-08-29）

## 背景
见 docs/upstream-tracking.md：release 陈旧，main 追 torch nightly。（原文『torch_npu 追 torch 正式版』已被 2026-08-30 的实测推翻——torch_npu master 也追 nightly；见 ADR-006。本 ADR 的决定不变：torchtitan 仍按 SHA 固定，且与 torch_npu SHA 一起升级。）

## 决定
`constraints/torchtitan.sha` 保存 commit。`scripts/install.sh` 检出它并以 `--no-deps` 加我们的依赖列表安装。升级是一个附带全量矩阵结果的 PR。`scripts/probe_compat.sh` 报告最远能前进到哪。

## 后果
- 仅 `pip install ascend-torchtitan` 不够；安装脚本是产品的一部分。
- CI 两条腿：pinned（门禁）与 main（漂移探针，允许失败）。
