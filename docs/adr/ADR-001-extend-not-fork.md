# ADR-001：树外扩展 torchtitan，绝不 fork

## 状态
已采纳（2026-08-29）

## 背景
torchtitan 每月约 170 个 commit，并提供了官方扩展点：完整路径的 `--module`、子类化 `Trainer.Config` 后的 `config.build()`、`ModelSpec` 的可调用字段，以及 `@override` 机制——其 README 明确面向硬件厂商，并说 vendor 内核应放在外部包里。

## 决定
`ascend-torchtitan` 是一个独立的包。它导入 torchtitan，从不修改其源码树，分层为：L0 受治理 shim（只给扩展点够不着的代码）、L1 override、L2 并行/图模式、L3 recipe、L4 工具。

## 备选
- **fork**：上游每个 commit 都变成一次合并；否决。
- **钉死某个 release 的自洽发行版**：release 落后 main 半年以上，且没有 override 机制和 kimi_k3；否决。

## 后果
- 我们依赖上游 `Config` 的形状；`derive()` 和 CPU 上的 recipe 测试是预警系统。
- 有些计算（kimi_k3 attn_res）不改上游就够不着；这些成为上游 ask，而不是替换父块（P6）。
