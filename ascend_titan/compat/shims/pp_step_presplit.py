"""Accept pre-split microbatches in ``PipelineSchedule.step`` on torch <= 2.13.

torchtitan's ``pp_forward_backward_step`` calls
``schedule.step(arg_mbs=..., kwarg_mbs=..., target_mbs=..., losses=..., loss_kwargs=...)``
(pre-split inputs). That keyword contract exists only in torch nightly; released
torch treats the ``*_mbs`` names as model kwargs and tries to split them into
microbatches -> ``IndexError: Dimension specified as 0 but tensor has no dimensions``
(the 0-dim ``global_valid_tokens``).

Released torch already has everything else: ``_step_microbatches(arg_mbs,
kwarg_mbs, target_mbs, losses, return_outputs, loss_kwargs)`` with the same
parameters. The wrapper below forwards pre-split calls to it, reproducing the
few preamble lines of the original ``step`` (grad check, stage flags, runtime
state reset); calls without ``*_mbs`` go to the original untouched. On a torch
whose ``step`` already accepts ``arg_mbs`` the wrapper is not installed.

Attribution: TT (nightly-only pipelining API, no feature check; see
docs/issues/torchtitan.md#pp-step).
"""

from __future__ import annotations

import inspect

from ascend_titan.compat import shim

_UPSTREAM = "draft:docs/issues/torchtitan.md#pp-step"
_REASON = (
    "torchtitan passes pre-split microbatches (arg_mbs/kwarg_mbs/target_mbs) to "
    "PipelineSchedule.step, a nightly-only keyword contract"
)


def _wrap_step(original):
    if "arg_mbs" in inspect.signature(original).parameters:
        return original  # torch already supports pre-split inputs

    import torch

    def step(
        self,
        *args,
        target=None,
        losses=None,
        return_outputs=True,
        loss_kwargs=None,
        arg_mbs=None,
        kwarg_mbs=None,
        target_mbs=None,
        **kwargs,
    ):
        if arg_mbs is None and kwarg_mbs is None and target_mbs is None:
            return original(
                self,
                *args,
                target=target,
                losses=losses,
                return_outputs=return_outputs,
                loss_kwargs=loss_kwargs,
                **kwargs,
            )
        if args or kwargs or target is not None:
            raise ValueError(
                "pre-split inputs (arg_mbs/kwarg_mbs/target_mbs) cannot be mixed with "
                "positional args, model kwargs or target="
            )
        # --- preamble mirrored from torch 2.12/2.13 step() ---
        if (
            self._has_backward
            and getattr(self, "_backward_requires_autograd", True)
            and not torch.is_grad_enabled()
        ):
            raise RuntimeError(
                "step() requires gradients to be enabled for backward computation; "
                "it should not be used under torch.no_grad() context. "
                "Please call eval() instead."
            )
        stages = getattr(self, "_stages", None) or [self._stage]
        for stage in stages:
            stage.has_backward = self._has_backward
        for stage in stages:
            stage.clear_runtime_states()
        # --- pre-split path: what nightly's _get_microbatch_inputs would hand over ---
        self._step_microbatches(
            arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs, loss_kwargs=loss_kwargs
        )
        if return_outputs:
            for stage in stages:
                if stage.is_last:
                    return self._merge_outputs(stage.output_chunks)
        return None

    step.__doc__ = (original.__doc__ or "") + "\n\n[ascend_titan] accepts *_mbs pre-split inputs."
    return step


@shim(
    target="torch.distributed.pipelining.schedules:PipelineScheduleSingle",
    reason=_REASON,
    upstream=_UPSTREAM,
    kind="wrap",
)
def pp_step_presplit_single(original):
    original.step = _wrap_step(original.step)
    return original


@shim(
    target="torch.distributed.pipelining.schedules:PipelineScheduleMulti",
    reason=_REASON,
    upstream=_UPSTREAM,
    kind="wrap",
)
def pp_step_presplit_multi(original):
    original.step = _wrap_step(original.step)
    return original
