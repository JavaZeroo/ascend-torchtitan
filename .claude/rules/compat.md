---
description: L0 shim 层规则
globs: ascend_titan/compat/**
---
# shim 规则
- 写 shim 前先在 `../torchtitan/torchtitan/config/configs.py` 里 grep 有没有开关（P0）。有开关就写 recipe 增量，不写 shim。
- 先归因（CLAUDE.md 的表）。只有 `TT`/`TORCH` 类失败可以变成 shim，**且必须先在 NIGHTLY 基线上复现**（P8）——只在正式版 torch 上出现的缺口不写 shim。`NPU` 永远不行（P1/P9）。
- 默认 `kind="wrap"`；缺失的 API 用 `kind="polyfill"`（属性已存在时自动跳过）。`kind="replace"` 需要 `why_not_wrap=`，并在 `docs/upstream-tracking.md` 说明什么上游改动能让我们删掉它。
- **polyfill 是版本债，不是禁区**：作为 plugin 必须保留"上游缺 API 就补上"的能力，但每条都随版本演进退场——目标属性一旦在当前版本上已存在，`test_polyfills_are_gone_once_upstream_provides_the_api` 会要求删除它。NIGHTLY 基线上正常应为 0 条（P8）。
- `shims/` 下一个文件一条 shim（或一组紧密相关的）；文件名 = shim 函数名。
- 每条 shim 配一个 CPU 单测：patch 一个假目标，断言包装后仍调用原函数。
- 同一个 PR 里更新 `docs/upstream-tracking.md` 的表。
- 删 shim 是值得庆祝的 PR。上游 issue 一关就删。
