"""NIGHTLY gate (P8): on a torch that already has the APIs our shims polyfill/wrap, the shims
must be no-ops -- the target objects stay torch's own. A shim that still takes effect on the
NIGHTLY baseline is a version-gap shim that should not exist."""

import importlib
import inspect

import pytest
import torch
import torch.distributed as dist


def _nightly_features_present() -> bool:
    from torch.distributed.pipelining.schedules import PipelineScheduleSingle

    return hasattr(dist, "set_timeout") and (
        "arg_mbs" in inspect.signature(PipelineScheduleSingle.step).parameters
    )


@pytest.mark.skipif(not _nightly_features_present(), reason="torch predates the NIGHTLY baseline")
def test_shims_are_noops_on_nightly_torch():
    from torch.distributed.pipelining.schedules import PipelineScheduleMulti, PipelineScheduleSingle

    from ascend_titan.compat import registry

    registry.reset_for_tests()
    for mod in (
        "ascend_titan.compat.shims.dist_set_timeout",
        "ascend_titan.compat.shims.pp_step_presplit",
    ):
        importlib.reload(importlib.import_module(mod))
    orig_timeout = dist.set_timeout
    orig_single, orig_multi = PipelineScheduleSingle.step, PipelineScheduleMulti.step
    registry.apply_all()
    assert dist.set_timeout is orig_timeout, "polyfill replaced torch's own set_timeout"
    assert not hasattr(dist.set_timeout, "__ascend_shim__")
    assert (
        PipelineScheduleSingle.step is orig_single and PipelineScheduleMulti.step is orig_multi
    ), "pp_step_presplit wrapped a torch that already accepts arg_mbs"
    assert torch.__version__  # keep torch referenced for the skip condition
    registry.reset_for_tests()
