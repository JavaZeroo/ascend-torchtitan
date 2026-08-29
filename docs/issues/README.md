# 问题清单

所有阻止 torchtitan 原样在昇腾上运行的问题，按**谁负责修**分类。每条都写成可直接贴进 issue 跟踪器的形式。**状态只在 `STATUS.md` 维护**（P11）；只在正式版 torch 上出现、NIGHTLY 上不存在的问题按 P8 关闭。torch_npu / op-plugin 的问题走 P9 流程提到 gitcode；torchtitan / pytorch 的问题不提上游（P10），修复方案存 `patches/evidence/`。

| 归属 | 文件 | 内容 |
|---|---|---|
| torch_npu / op-plugin | [torch_npu.md](torch_npu.md) | torch_npu 缺失的算子 / 后端注册 / 与 nightly 的漂移（P1/P9：本仓绝不绕过，修好提 PR） |
| pytorch | [pytorch.md](pytorch.md) | torch 核心里的设备白名单与扩展点 |
| torchtitan | [torchtitan.md](torchtitan.md) | 无特性检查地使用 nightly-only API、无条件导入 CUDA-only 依赖、缺失的 `Configurable` 节点 |
| ascend-torchtitan | [ours.md](ours.md) | 本仓已知的缺口 |

归因代码（CLAUDE.md）：**TT** torchtitan · **NPU** torch_npu · **CANN** · **DEP** 第三方 CUDA-only 依赖 · **TORCH** pytorch 核心。
