"""Measurement-only Qwen3.5 configs. NOT for training.

A probe isolates one variable so a red cell can be blamed on that variable and
nothing else. Keeping them out of ``recipes.py`` keeps "what we support" and
"what we measure" apart -- a recipe is meant to be run, a probe is an experiment
whose expected result may well be 🔴.

Run one exactly like a recipe::

    MODULE=ascend_titan.models.qwen3_5.probes CONFIG=qwen35_0_8b_lr_5e4 \\
        NPU=1 ./scripts/run_train.sh --training.steps 10 --metrics.log_freq 1
"""

from torchtitan.trainer import Trainer

from ascend_titan.models.qwen3_5.recipes import qwen35_0_8b_npu


def _with_lr(config: Trainer.Config, lr: float) -> Trainer.Config:
    """Override the learning rate of every parameter group.

    There is no ``optimizer.lr`` to set from the CLI: the rate lives inside
    ``param_groups[*].optimizer_kwargs``, which tyro cannot address.
    """
    for group in config.optimizer.param_groups:
        group.optimizer_kwargs = {**group.optimizer_kwargs, "lr": lr}
    return config


def qwen35_0_8b_lr_5e4() -> Trainer.Config:
    """Is the step-4 non-finite loss the learning rate, or our GDN?

    ``qwen35_0_8b_npu`` goes non-finite around step 4 from random init. The
    kernel is already ruled out: ``ascend_chunk_gdn`` matches attn_gym's
    reference to 1.6e-7 at seq 16384 even with the worst-case gates
    (``tests/unit/test_kernel_gdn.py``). The other thing our recipe changes is
    the *data* -- upstream's 0.8B trains on multimodal cc12m at lr 5e-3, and we
    swapped in C4 text. This probe backs the rate off by 10x; if it trains, the
    divergence is ours to explain in the recipe, not Ascend's.
    """
    return _with_lr(qwen35_0_8b_npu(), 5e-4)


def qwen35_0_8b_lr_1e3() -> Trainer.Config:
    """Same question, half the back-off, to bracket where it breaks."""
    return _with_lr(qwen35_0_8b_npu(), 1e-3)
