# ADR-004：算子依赖缺失时响亮地退回上游 eager

## 状态
已采纳（2026-08-29）

## 背景
融合算子依赖单独安装的包（triton-ascend-kernels、fla-npu、ops-nn、ops-transformer）。选项：硬失败、静默回退、显式三档 impl 选择。

## 决定
回退 = 不注册 override，于是上游自己的 torch 实现运行。打 WARNING 并记 provenance。benchmark 必须附 provenance 表（P7）。

## 备选
- 硬失败：在部分环境下完全跑不起来。
- 每个算子显式 `impl=ascendc|triton|eager`（MindSpeed-MM 风格）：代码更多（每个算子三份实现）；eager 路径上游本来就有，免费。

## 后果
- 性能数字只有带 provenance 才可信；nightly 性能基线是第二道防线。
