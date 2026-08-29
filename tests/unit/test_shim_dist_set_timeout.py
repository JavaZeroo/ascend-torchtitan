import importlib

import torch.distributed as dist

from ascend_titan.compat import registry


def test_polyfill_provides_set_timeout(clean_registry, monkeypatch):
    # reload so the @shim decorator re-registers into the clean registry
    import ascend_titan.compat.shims.dist_set_timeout as m

    importlib.reload(m)
    had_it = hasattr(dist, "set_timeout")
    applied = registry.apply_all()
    assert hasattr(dist, "set_timeout")
    if had_it:
        # newer torch: polyfill must be a no-op, nothing applied
        assert applied == []
    else:
        assert [a.name for a in applied] == ["dist_set_timeout"]
        assert dist.set_timeout.__ascend_shim__ == "dist_set_timeout"
        monkeypatch.delattr(dist, "set_timeout")  # leave torch as we found it
