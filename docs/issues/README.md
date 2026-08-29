# 问题清单

所有阻止 torchtitan 原样在昇腾上运行的问题，按**谁负责修**分类。每条都写成可直接贴进 issue 跟踪器的形式；提交后把 URL 替换掉代码/文档里的 `draft:` 指针。状态：`draft`（文本就绪，未提交）· `filed`（附链接）· `fixed@<版本>` · `wontfix`。

| 归属 | 文件 | 内容 |
|---|---|---|
| torch_npu | [torch_npu.md](torch_npu.md) | torch_npu 缺失的算子 / 后端注册（P1：本仓绝不绕过） |
| pytorch | [pytorch.md](pytorch.md) | torch 核心里的设备白名单与扩展点 |
| torchtitan | [torchtitan.md](torchtitan.md) | 无特性检查地使用 nightly-only API、无条件导入 CUDA-only 依赖、缺失的 `Configurable` 节点 |
| ascend-torchtitan | [ours.md](ours.md) | 本仓已知的缺口 |

归因代码（CLAUDE.md）：**TT** torchtitan · **NPU** torch_npu · **CANN** · **DEP** 第三方 CUDA-only 依赖 · **TORCH** pytorch 核心。
