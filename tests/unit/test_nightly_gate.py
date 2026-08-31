"""NIGHTLY gate (P8)：不为正式版 torch 保留兼容代码。

只在正式版 torch 上出现、nightly 上不存在的问题不写 shim，也不留 shim。可检查的代理：
**没有一条 shim 允许以 `torch.*` 为目标**——nightly 上 torch 自身的缺口是 torch 的 bug，
按 P10 记录、不 shim；合法的 shim 全都指向 torchtitan 的设计缺口（无条件 `torch.compile`、
硬编码的设备判断这类没有开关的地方）。

历史上违反这条的两个例子都已删除：`dist_set_timeout`（polyfill 补 `torch.distributed`
缺的 API）与 `pp_step_presplit`（包 `torch.distributed.pipelining`，只为 torch ≤ 2.13）。
"""

import importlib
import pkgutil

from ascend_titan.compat import registry


def _registered():
    """重新执行每个 shim 模块的装饰器。

    不能只调 `_discover()`：别的测试跑过之后模块已在 `sys.modules` 里，import 是缓存命中，
    装饰器不会再跑，注册表会是空的。
    """
    import ascend_titan.compat.shims as pkg

    registry.reset_for_tests()
    for info in pkgutil.iter_modules(pkg.__path__):
        importlib.reload(importlib.import_module(f"{pkg.__name__}.{info.name}"))
    return dict(registry._REGISTRY)


def test_no_shim_targets_torch_itself():
    shims = _registered()
    assert shims, "shim 注册表是空的——discovery 坏了"
    offenders = {
        name: s.target for name, s in shims.items() if s.target.split(":")[0].startswith("torch.")
    }
    assert not offenders, (
        f"这些 shim 以 torch 自身为目标：{offenders}。"
        "nightly 上 torch 的缺口按 P10 记录并推动上游，不在本仓 shim。"
    )
    registry.reset_for_tests()


def test_every_shim_targets_torchtitan():
    shims = _registered()
    for name, s in shims.items():
        module = s.target.split(":")[0]
        assert module.startswith("torchtitan."), f"{name} 指向 {module}，不是 torchtitan 的缺口"
    registry.reset_for_tests()


def test_no_polyfills_remain():
    """polyfill 只用来补旧 torch 缺的 API——nightly-first 之后不该再有。"""
    shims = _registered()
    polyfills = [name for name, s in shims.items() if s.kind == "polyfill"]
    assert not polyfills, f"这些是给旧 torch 补 API 的 polyfill，应当删除：{polyfills}"
    registry.reset_for_tests()
