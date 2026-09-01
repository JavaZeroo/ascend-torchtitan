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

import pytest

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


def test_polyfills_are_gone_once_upstream_provides_the_api():
    """polyfill 是**版本债**，不是禁令。

    作为 plugin 我们必须保留"上游缺这个 API 就补上"的能力（`kind="polyfill"`），
    但每一条都要随版本演进退场：一旦装好的 torch / torchtitan 自己有了这个属性，
    这条 polyfill 就只是死重量，运行时会被静默跳过、没人会发现。这里替运行时把它
    喊出来——目标已存在 ⇒ 删掉它。

    NIGHTLY 基线上通常一条 polyfill 都不该有（P8：我们永远在最新的 torch 上），
    所以这条测试现在是空过；它是给 polyfill 真的出现那天准备的。
    """
    import importlib

    pytest.importorskip("torchtitan")
    stale = []
    for name, s in _registered().items():
        if s.kind != "polyfill":
            continue
        owner = s.owner(importlib.import_module(s.module))
        if hasattr(owner, s.attr):
            stale.append(f"{name} -> {s.target}")
    assert not stale, f"这些 polyfill 的目标在当前版本上已经存在，上游补上了，应当删除：{stale}"
    registry.reset_for_tests()
