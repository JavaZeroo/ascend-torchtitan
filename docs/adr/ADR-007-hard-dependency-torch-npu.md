# ADR-007：torch_npu 是基础依赖，硬导入

## 状态
已采纳（2026-08-30），细化 ADR-004 的适用范围。

## 背景
L1 的 5 个内核模块与 `_bootstrap.py` 都写着

```python
try:
    import torch_npu
    _AVAILABLE = hasattr(torch_npu, "npu_xxx")
except ImportError as e:
    _AVAILABLE = False
    logger.warning("torch_npu unavailable (%s); ... stays on eager", e)
```

这是把 ADR-004（"算子依赖缺失时响亮降级"）套用到了**设备后端**上。两者不是一回事：

* ops-nn / Triton-Ascend 是**可选加速包**——需要单独的 run 包与 JIT 构建，不在 NIGHTLY 基线里，装不上是常态，退回上游 eager 仍然是一次有意义的运行。
* `torch_npu` 是**基础依赖**——本仓存在的唯一目的就是在昇腾上跑 torchtitan。没有它，`device_type` 不是 `npu`，`@override` 不注册，训练要么在别处以更难懂的错误死掉，要么静悄悄地跑了一次没有融合算子的 eager，而 WARNING 淹没在 torchtitan 的日志里。**"绿的但测的不是我们要测的东西"比红的更糟。**

`hasattr(torch_npu, "npu_xxx")` 同样有问题：算子缺失是**昇腾侧的缺口**，按 P1/P9 必须去修 torch_npu / op-plugin，而不是在本仓静默换条路。

## 决定
1. `torch`、`torch_npu`、`torchtitan` 一律无条件 `import`；缺失时 `ImportError` 直接向上抛。
2. 算子级探测统一走 `ascend_titan/kernels/_probe.py::require_op(name)`：缺失抛 `MissingNpuOpError`，消息里写明版本、缺的算子、以及"去 gitcode.com/Ascend 修并提 issue/PR（P9），不要在本仓绕过"。
3. `setup()` 去掉 `require_npu` 参数——它本来就不该是可选的。
4. 真正可选的加速包走 `_probe.optional_module(*candidates)`，这是 ADR-004 保留的唯一"WARNING + 降级"通道，且必须在 WARNING 里写清楚降级了什么。
5. 诊断工具例外：`ascend-titan-doctor`、`tests/repro/probe_*.py` 的职责就是报告缺什么。
6. CPU 单测通过 `tests/conftest.py` 的 `npu_stub` / `no_torch_npu` / `npu_stub_missing_op` fixture 提供假的 `torch_npu`——测试要提供依赖，而不是去测一条已经不存在的降级路径。

## 后果
- `tests/unit/test_kernel_import_safety.py` 从"没有 torch_npu 也能安全 import"翻转为"没有 torch_npu 必须抛错"。
- 在没装 torch_npu 的机器上 `import ascend_titan.kernels.attention` 会失败——这是有意的；`import ascend_titan` 本身仍然无副作用（P0/F4 不变）。
- ADR-004 收窄到可选加速包；P7 的"响亮降级"同样只对它们生效。
