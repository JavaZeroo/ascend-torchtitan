import importlib
import sys
import types

import pytest

from ascend_titan.compat import registry


def _fake_upstream(monkeypatch, compiled_marker):
    """搭一个最小的 torchtitan 表面：set_determinism + FlexAttention._compiled_flex_attn。

    不依赖真实 torchtitan——shim 规则要求每条 shim 有一个 patch 假目标的 CPU 单测。
    """
    calls = []

    utils = types.ModuleType("torchtitan.distributed.utils")

    def set_determinism(*args, **kwargs):
        calls.append((args, kwargs))
        # 这正是上游在非 ROCm 分支上做的事：把 _compiled_flex_attn 换成编译版
        attention.FlexAttention._compiled_flex_attn = compiled_marker
        return "upstream-return"

    utils.set_determinism = set_determinism

    attention = types.ModuleType("torchtitan.models.common.attention")

    class FlexAttention:
        _compiled_flex_attn = compiled_marker

    attention.FlexAttention = FlexAttention

    for name, mod in (
        ("torchtitan", types.ModuleType("torchtitan")),
        ("torchtitan.distributed", types.ModuleType("torchtitan.distributed")),
        ("torchtitan.distributed.utils", utils),
        ("torchtitan.models", types.ModuleType("torchtitan.models")),
        ("torchtitan.models.common", types.ModuleType("torchtitan.models.common")),
        ("torchtitan.models.common.attention", attention),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return utils, attention, calls


def _reload_shim(clean_registry):
    """导入并只注册一次。

    首次导入本身会跑一遍 @shim 装饰器，之后 reload 再跑一遍就会撞重名——
    已有的 shim 测试没踩到，只是因为它们的模块早被别的测试导入过。
    """
    import ascend_titan.compat.shims.flex_eager_when_deterministic as m

    clean_registry.reset_for_tests()
    importlib.reload(m)
    return m


def test_restores_eager_flex_after_upstream_compiles_it(clean_registry, monkeypatch):
    """包装后仍调用原函数，并把 _compiled_flex_attn 复位成 eager。"""
    torch_flex = pytest.importorskip("torch.nn.attention.flex_attention")

    compiled_marker = object()  # 冒充 torch.compile(flex_attention) 的返回值
    utils, attention, calls = _fake_upstream(monkeypatch, compiled_marker)

    m = _reload_shim(clean_registry)
    assert m.__name__.endswith("flex_eager_when_deterministic")
    applied = registry.apply_all()
    assert [a.name for a in applied] == ["flex_eager_when_deterministic"]

    result = utils.set_determinism("debug-config", world="mesh")

    # 原函数照常被调用，返回值透传
    assert calls == [(("debug-config",), {"world": "mesh"})]
    assert result == "upstream-return"
    # 上游把它换成了编译版，shim 再换回 eager
    assert attention.FlexAttention._compiled_flex_attn is torch_flex.flex_attention


def test_leaves_an_already_eager_attribute_alone(clean_registry, monkeypatch):
    """上游若已经走了 eager 分支（ROCm），shim 不该再动它，也不该重复打日志。"""
    torch_flex = pytest.importorskip("torch.nn.attention.flex_attention")

    utils, attention, _ = _fake_upstream(monkeypatch, torch_flex.flex_attention)

    _reload_shim(clean_registry)
    registry.apply_all()
    utils.set_determinism("debug-config")

    assert attention.FlexAttention._compiled_flex_attn is torch_flex.flex_attention
