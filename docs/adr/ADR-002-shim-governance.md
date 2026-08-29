# ADR-002：shim 是受治理的债务，不是兼容层

## 状态
已采纳（2026-08-29）

## 背景
override 机制只能触及 `Configurable.Config` 节点。有些写死 CUDA 的代码在自由函数里（例如 AC 的 D2H 策略）。monkeypatch 是树外唯一的工具，而 monkeypatch 会静默腐烂。

## 决定
一个 shim 注册表（`ascend_titan.compat`），在 import 时拒绝任何没有 `reason` 和 `upstream` 的 shim（P4），以及没有 `why_not_wrap` 的 `replace` 型 shim（P3）。shim 只由 `setup()` 应用。`doctor` 报告 shim 数量，期望趋近于零。`polyfill` 型只在属性缺失时添加，新版 torch 自带后自动 no-op。

## 备选
- 被补丁函数的源码指纹：推迟；包装型 shim 自动继承上游变更，指纹只对替换型有意义，而替换型应当罕见。
- vendor 无关的 shim 框架：否决（单一 vendor 抽不出好抽象；设备无关的接缝应放上游）。

## 后果
- 加 shim 前必须先提上游 issue。这是有意设置的摩擦。
