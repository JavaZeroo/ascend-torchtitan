"""Recipes must build on CPU: this is the cheapest drift detector we have.

It already paid for itself: it caught upstream's removal of the ``sdpa`` LM
attention backend before anyone ran on an NPU.
"""

import inspect

import pytest

pytestmark = pytest.mark.titan


def test_qwen3_npu_recipes_build():
    from ascend_titan.recipes.qwen3 import (
        ATTENTION_OVERRIDE,
        qwen3_debugmodel_npu,
        qwen3_debugmodel_npu_fsdp2,
        qwen3_debugmodel_stock_flex,
        qwen3_debugmodel_stock_varlen,
    )

    cfg = qwen3_debugmodel_npu()
    assert cfg.training.steps == 10
    assert cfg.checkpoint.enable is False
    assert cfg.override.imports[0] == ATTENTION_OVERRIDE
    assert cfg.parallelism.spmd_backend == "partial_dtensor"
    assert type(cfg.loss).__qualname__ == "CrossEntropyLoss.Config"

    assert qwen3_debugmodel_npu_fsdp2().parallelism.data_parallel_shard_degree == 2
    assert qwen3_debugmodel_stock_flex().override.imports == []
    assert qwen3_debugmodel_stock_varlen().override.imports == []


def test_upstream_lm_attention_backends_are_what_we_think():
    """Pin the fact the recipes depend on; fails loudly if upstream changes it again."""
    from torchtitan.models.common.config_utils import get_attention_config

    with pytest.raises(ValueError, match="sdpa is no longer supported"):
        get_attention_config("sdpa")
    assert get_attention_config("flex") is not None
    assert get_attention_config("varlen") is not None


def test_recipe_is_delta_not_copy():
    """Guard the 'deltas only' rule: recipe must call the upstream registry fn."""
    from ascend_titan.recipes import qwen3

    src = inspect.getsource(qwen3.qwen3_debugmodel_npu)
    assert "qwen3_debugmodel()" in src
    assert "Trainer.Config(" not in src
