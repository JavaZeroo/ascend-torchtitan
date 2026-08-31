"""Recipes must build on CPU: this is the cheapest drift detector we have.

It already paid for itself: it caught upstream's removal of the ``sdpa`` LM
attention backend before anyone ran on an NPU.
"""

import inspect

import pytest

pytestmark = pytest.mark.titan


def test_qwen3_npu_recipes_build():
    from torchtitan.models.qwen3.config_registry import qwen3_debugmodel

    from ascend_titan.models.qwen3 import (
        qwen3_debugmodel_npu,
        qwen3_debugmodel_npu_fsdp2,
        qwen3_debugmodel_stock_flex,
        qwen3_debugmodel_stock_varlen,
    )
    from ascend_titan.models.qwen3.recipes import ATTENTION_OVERRIDE

    cfg = qwen3_debugmodel_npu()
    assert cfg.training.steps == 10
    assert cfg.checkpoint.enable is False
    assert cfg.override.imports[0] == ATTENTION_OVERRIDE
    assert cfg.parallelism.spmd_backend == qwen3_debugmodel().parallelism.spmd_backend
    assert type(cfg.loss).__qualname__ == "ChunkedLossWrapper.Config"
    assert type(cfg.loss) is type(qwen3_debugmodel().loss)

    assert qwen3_debugmodel_npu_fsdp2().parallelism.data_parallel_shard_degree == 2
    assert qwen3_debugmodel_stock_flex().override.imports == []
    assert qwen3_debugmodel_stock_varlen().override.imports == []


def test_reference_recipe_keeps_only_two_deltas():
    """P12: every delta needs a reason to exist. Two are left, and both are named
    in models/qwen3/README.md with their disappearance condition."""
    from torchtitan.models.qwen3.config_registry import qwen3_debugmodel

    from ascend_titan.models.qwen3 import qwen3_debugmodel_npu

    up, cfg = qwen3_debugmodel(), qwen3_debugmodel_npu()
    assert type(cfg.loss) is type(up.loss)
    assert cfg.parallelism.spmd_backend == up.parallelism.spmd_backend
    assert cfg.optimizer == up.optimizer and cfg.lr_scheduler == up.lr_scheduler
    assert cfg.training == up.training
    # delta 1 (attention node + override) and delta 2 (no checkpoint I/O)
    assert cfg.override.imports and cfg.checkpoint.enable is False


def test_upstream_lm_attention_backends_are_what_we_think():
    """Pin the fact the recipes depend on; fails loudly if upstream changes it again."""
    from torchtitan.models.common.config_utils import get_attention_config

    with pytest.raises(ValueError, match="sdpa is no longer supported"):
        get_attention_config("sdpa")
    assert get_attention_config("flex") is not None
    assert get_attention_config("varlen") is not None


def test_recipe_is_delta_not_copy():
    """Guard the 'deltas only' rule: recipe must call the upstream registry fn."""
    from ascend_titan.models.qwen3 import recipes

    src = inspect.getsource(recipes.qwen3_debugmodel_npu)
    assert "qwen3_debugmodel()" in src
    assert "Trainer.Config(" not in src


def test_probes_are_not_recipes():
    """probes.py is measurement-only: it must not leak into the supported entry points."""
    from ascend_titan.models.qwen3 import probes, recipes

    assert not set(vars(probes)) & {
        "qwen3_debugmodel_npu_fsdp2",
        "qwen3_debugmodel_npu_fused",
    }
    assert "stock" not in " ".join(n for n in vars(recipes) if n.startswith("qwen3_"))


def test_llama3_stock_recipe_has_no_overrides():
    """The zero-override reference path: any override here defeats its purpose."""
    from ascend_titan.models.llama3 import (
        llama3_debugmodel_stock_npu,
        llama3_debugmodel_stock_npu_fsdp2,
    )

    cfg = llama3_debugmodel_stock_npu()
    assert cfg.override.imports == []
    assert cfg.checkpoint.enable is False
    assert llama3_debugmodel_stock_npu_fsdp2().parallelism.data_parallel_shard_degree == 2


def test_ce_loss_probe_is_the_only_place_that_unchunks_the_loss():
    """Chunked loss is supported and default; the plain-CE path stays a probe."""
    from torchtitan.models.qwen3.config_registry import qwen3_debugmodel

    from ascend_titan.models.qwen3 import (
        qwen3_debugmodel_npu,
        qwen3_debugmodel_npu_ce_loss,
        recipes,
    )

    assert "config.loss" not in inspect.getsource(recipes.qwen3_debugmodel_npu)
    assert type(qwen3_debugmodel_npu().loss) is type(qwen3_debugmodel().loss)
    assert type(qwen3_debugmodel_npu_ce_loss().loss).__qualname__ == "CrossEntropyLoss.Config"


@pytest.mark.titan
def test_release_recipes_are_deltas_on_the_real_upstream_configs(monkeypatch, tmp_path):
    """R1: a release recipe must start from the upstream *real-size* config, not
    from debugmodel and not from a hand-built Trainer.Config."""
    import inspect

    from ascend_titan.models.qwen3 import recipes as q3

    checked = [(q3.qwen3_0_6b_npu, "qwen3_0_6b()"), (q3.qwen3_8b_npu_pp2, "qwen3_14b()")]
    # qwen3_5 needs fla-core, which is an extra (see models/qwen3_5/README.md)
    pytest.importorskip("fla", reason="pip install fla-core (qwen3_5 extra)")
    from ascend_titan.models.qwen3_5 import recipes as q35

    checked.append((q35.qwen35_0_8b_npu, "qwen35_0_8b()"))
    for fn, upstream_call in checked:
        src = inspect.getsource(fn)
        assert upstream_call in src
        assert "Trainer.Config(" not in src


def test_assets_helper_refuses_to_guess(tmp_path, monkeypatch):
    """A missing asset must say how to get it, not silently fall back to the toy one."""
    from ascend_titan.models import assets

    monkeypatch.setenv("ASCEND_TITAN_ASSETS", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_assets.sh tokenizer"):
        assets.hf_assets_path("Qwen3-0.6B")
    with pytest.raises(FileNotFoundError, match="fetch_assets.sh c4"):
        assets.c4_shards()

    (tmp_path / "hf" / "Qwen3-0.6B").mkdir(parents=True)
    (tmp_path / "hf" / "Qwen3-0.6B" / "tokenizer.json").write_text("{}")
    assert assets.hf_assets_path("Qwen3-0.6B").endswith("Qwen3-0.6B")


def test_c4_subset_is_real_text_not_the_toy_file(tmp_path, monkeypatch):
    """The subset is cut from a real shard; it must keep whole C4 records."""
    import gzip
    import json

    from ascend_titan.models import assets

    monkeypatch.setenv("ASCEND_TITAN_ASSETS", str(tmp_path))
    shard = tmp_path / "c4" / "en" / "c4-train.00000-of-01024.json.gz"
    shard.parent.mkdir(parents=True)
    with gzip.open(shard, "wt", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"document {i}", "url": "u"}) + "\n")

    path = assets.c4_subset(docs=4)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert len(rows) == 4
    assert rows[0]["text"] == "document 0"
    assert assets.c4_subset(docs=4) == path  # cached, not rebuilt
