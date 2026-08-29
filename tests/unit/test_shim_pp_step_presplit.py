import importlib
import inspect

import pytest

from ascend_titan.compat import registry


class _Stage:
    def __init__(self, is_last):
        self.is_last = is_last
        self.has_backward = None
        self.cleared = 0
        self.output_chunks = ["o"]

    def clear_runtime_states(self):
        self.cleared += 1


class _Sched:
    """Stand-in for a torch 2.13 PipelineScheduleSingle."""

    _has_backward = True

    def __init__(self):
        self._stage = _Stage(is_last=True)
        self.calls = []

    def step(
        self, *args, target=None, losses=None, return_outputs=True, loss_kwargs=None, **kwargs
    ):
        self.calls.append(("orig", args, kwargs))
        return "orig"

    def _step_microbatches(
        self, arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs, loss_kwargs=None
    ):
        self.calls.append(("mbs", arg_mbs, kwarg_mbs, target_mbs, losses, loss_kwargs))

    def _merge_outputs(self, chunks):
        return "merged"


def test_wrapper_routes_presplit_and_passes_through():
    from ascend_titan.compat.shims.pp_step_presplit import _wrap_step

    _Sched.step = _wrap_step(_Sched.step)
    s = _Sched()
    assert s.step("x", mask=1) == "orig"
    losses = []
    out = s.step(
        arg_mbs=[("a",)],
        kwarg_mbs=[{"m": 1}],
        target_mbs=["t"],
        losses=losses,
        loss_kwargs={"g": 2},
        return_outputs=True,
    )
    assert out == "merged"
    assert s.calls[-1] == ("mbs", [("a",)], [{"m": 1}], ["t"], losses, {"g": 2})
    assert s._stage.has_backward is True and s._stage.cleared == 1
    with pytest.raises(ValueError, match="cannot be mixed"):
        s.step("x", arg_mbs=[("a",)])


def test_loss_kwargs_bound_when_step_microbatches_lacks_them():
    from ascend_titan.compat.shims.pp_step_presplit import _wrap_step

    class Old(_Sched):
        def __init__(self):
            super().__init__()
            self._loss_fn = lambda out, tgt, **kw: ("loss", kw)
            self.seen = None

        def _step_microbatches(self, arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs):
            self.seen = self._loss_fn("o", "t")

    Old.step = _wrap_step(_Sched.__dict__["step"])
    s = Old()
    s.step(
        arg_mbs=[()],
        kwarg_mbs=[{}],
        target_mbs=["t"],
        losses=[],
        loss_kwargs={"g": 7},
        return_outputs=False,
    )
    assert s.seen == ("loss", {"g": 7})
    assert s._loss_fn("o", "t") == ("loss", {})  # restored


def test_wrap_is_noop_when_torch_already_supports_it():
    from ascend_titan.compat.shims.pp_step_presplit import _wrap_step

    def step(self, *args, arg_mbs=None, **kwargs):
        return "native"

    assert _wrap_step(step) is step


def test_registered_against_real_torch(clean_registry):
    import torch.distributed.pipelining.schedules as sch

    import ascend_titan.compat.shims.pp_step_presplit as m

    importlib.reload(m)
    native = "arg_mbs" in inspect.signature(sch.PipelineScheduleSingle.step).parameters
    applied = registry.apply_all()
    assert {a.name for a in applied} == {"pp_step_presplit_single", "pp_step_presplit_multi"}
    assert ("arg_mbs" in inspect.signature(sch.PipelineScheduleSingle.step).parameters) or native
