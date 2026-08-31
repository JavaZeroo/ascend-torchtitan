import pytest

from ascend_titan.tools.matrix import parse_cards, triage


def test_triage_priority_and_unknown():
    assert (
        triage("blah\nRuntimeError: No backend type associated with device type npu")[0] == "NPU-2"
    )
    assert (
        triage("NotImplementedError: Could not run 'aten::_flash_attention_forward'")[0] == "NPU-1"
    )
    assert (
        triage("ValueError: FlexAttention is only supported on CUDA, CPU or HPU devices.")[0]
        == "TORCH-1"
    )
    assert triage('  File "/x/torch_npu/foo.py", line 1\nRuntimeError: boom')[0] == "NPU"
    assert triage("something\n__TIMEOUT__\n")[0] == "HANG"
    assert triage("ERR99999 UNKNOWN applicaiton exception\nRuntimeError: x")[0] != "CANN"
    assert triage("[ERROR] EZ9999 op failed")[0] == "CANN"
    code, note = triage("NPU function error: call aclnnIndex failed, error code is 161002")
    assert code == "NPU-OP" and "aclnnIndex" in note and "161002" in note
    code, note = triage(
        "[rank0]:[rank0]: ZeroDivisionError: division by zero\nChildFailedError: \n"
    )
    assert code == "UNKNOWN" and note.startswith("ZeroDivisionError")


def test_parse_cards():
    assert parse_cards("0-3,6") == [0, 1, 2, 3, 6]


@pytest.mark.titan
def test_npu_minimal_transform_on_upstream_recipes():
    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.models.common.attention import FlexAttention, VarlenAttention
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    from ascend_titan.recipes.transforms import (
        ATTENTION_OVERRIDE,
        _flex_attention_is_usable,
        npu_minimal,
    )

    cfg = llama3_debugmodel()
    assert any(True for _ in cfg.traverse(FlexAttention.Config))
    a = npu_minimal(cfg)
    if _flex_attention_is_usable():
        # flex is usable here (device whitelist lifted *and* inductor has a
        # backend), so the upstream default stays.
        assert a.flex_to_varlen == 0
        assert any(True for _ in cfg.traverse(FlexAttention.Config))
        assert ATTENTION_OVERRIDE not in cfg.override.imports
    else:
        assert a.flex_to_varlen > 0
        assert not any(True for _ in cfg.traverse(FlexAttention.Config))
        assert any(True for _ in cfg.traverse(VarlenAttention.Config))
        assert ATTENTION_OVERRIDE in cfg.override.imports
    assert "ascend_titan.kernels.rope.real_cache_rope" in cfg.override.imports
    # A perf kernel must NOT be in the measurement baseline (P12): a red matrix
    # cell has to mean "this upstream feature fails on NPU", never "our drop-in
    # RMSNorm broke it".
    assert "ascend_titan.kernels.rms_norm.npu_rms_norm" not in cfg.override.imports
    assert cfg.parallelism.spmd_backend == "spmd_types"
    # TT-4 is gone on the NIGHTLY track; the upstream default loss wrapper stays in place.
    assert isinstance(cfg.loss, ChunkedLossWrapper.Config)
    # idempotent
    before = list(cfg.override.imports)
    b = npu_minimal(cfg)
    assert b.flex_to_varlen == 0 and cfg.override.imports == before


@pytest.mark.titan
def test_matrix_module_resolves_upstream_recipe():
    import ascend_titan.recipes.matrix as m

    fn = getattr(m, "torchtitan.models.llama3.config_registry__llama3_debugmodel")
    cfg = fn()
    assert cfg.parallelism.spmd_backend == "spmd_types"
    assert "ascend_titan.kernels.rms_norm.npu_rms_norm" not in cfg.override.imports
    stock = getattr(m, "torchtitan.models.llama3.config_registry__llama3_debugmodel__stock")()
    assert stock.override.imports == []
    fused = getattr(m, "torchtitan.models.llama3.config_registry__llama3_debugmodel__fused")()
    assert "ascend_titan.kernels.rms_norm.npu_rms_norm" in fused.override.imports
    with pytest.raises(AttributeError):
        _ = m.nonsense


@pytest.mark.titan
def test_npu_minimal_skips_rope_when_upstream_block_override_present():
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    from ascend_titan.recipes.transforms import ROPE_OVERRIDE, npu_minimal

    cfg = llama3_debugmodel()
    cfg.override.imports = ["torchtitan.overrides.fused_mla.fused_mla"]
    a = npu_minimal(cfg)
    assert ROPE_OVERRIDE not in cfg.override.imports and not a.rope_override


@pytest.mark.titan
def test_npu_fused_is_opt_in_and_idempotent():
    """The perf transform is separate from the baseline and safe to re-apply."""
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    from ascend_titan.recipes.transforms import RMSNORM_OVERRIDE, npu_fused, npu_minimal

    cfg = llama3_debugmodel()
    npu_minimal(cfg)
    assert RMSNORM_OVERRIDE not in cfg.override.imports
    a = npu_fused(cfg)
    assert a.rms_norm_override and RMSNORM_OVERRIDE in cfg.override.imports
    npu_fused(cfg)
    assert cfg.override.imports.count(RMSNORM_OVERRIDE) == 1


def test_encode_rejects_unknown_mode():
    from ascend_titan.recipes.matrix import encode

    with pytest.raises(ValueError, match="mode must be one of"):
        encode(lambda: None, mode="turbo")


def test_repo_root_points_at_the_checkout():
    """Guards a whole class of silent bugs: a wrong root makes the sweep import
    the wrong ``tests.integration_tests`` and shell out to a script that is not there."""
    from ascend_titan.tools.matrix import repo_root

    root = repo_root()
    assert (root / "scripts" / "run_train.sh").is_file()
    assert (root / "ascend_titan").is_dir()


def test_triage_rules_are_data_and_valid():
    """Every rule compiles, and every attribution code is one we document."""
    import re

    from ascend_titan.tools.matrix import rules

    known_prefixes = (
        "NPU",
        "NPU-OP",
        "TORCH",
        "TT",
        "DEP",
        "CANN",
        "OURS",
        "HARNESS",
        "HANG",
        "CLI",
        "COMPILE",
        "UNKNOWN",
    )
    assert len(rules()) > 20
    for code, pattern, note in rules():
        re.compile(pattern)
        assert note, f"{code}: a rule without a note is useless in the report"
        assert code.split("-")[0] in known_prefixes or code in known_prefixes, code


def test_provenance_section_orders_ascend_first():
    """The audit table has to put our own overrides where a reader looks first."""
    from ascend_titan.tools.matrix.report import render_provenance

    assert render_provenance({}) == []
    md = "\n".join(
        render_provenance(
            {
                "torchtitan.components.loss.CrossEntropyLoss.Config": ("upstream", 7),
                "ascend_titan.kernels.rms_norm.AscendRMSNorm.Config": ("ascend", 2),
                "torchtitan.overrides.fused_swiglu.FusedSwiGLU.Config": ("upstream-override", 1),
            }
        )
    )
    rows = [line for line in md.splitlines() if line.startswith("| `")]
    assert "ascend_titan" in rows[0] and "| ascend |" in rows[0]
    assert "upstream-override" in rows[1]
    assert "CrossEntropyLoss" in rows[2]


def test_bench_parses_and_takes_steady_state():
    """Step 1 carries warm-up; the baseline must not report it as throughput."""
    from ascend_titan.tools.bench import parse_steps, render, steady_state

    log = (
        "step:  1  loss:  7.53909  grad_norm:  9.7176  memory:  2.38GiB(3.90%)  "
        "tps: 10,000  tflops: 5.00  mfu: 1.00%\n"
        "step:  2  loss:  6.88108  grad_norm:  6.4899  memory:  2.38GiB(3.90%)  "
        "tps: 50,000  tflops: 29.00  mfu: 9.00%\n"
        "step:  3  loss:  5.10291  grad_norm:  3.3062  memory:  2.38GiB(3.90%)  "
        "tps: 52,000  tflops: 30.00  mfu: 9.50%\n"
    )
    steps = parse_steps(log)
    assert [s["step"] for s in steps] == [1, 2, 3]
    s = steady_state(steps)
    assert s["tps"] >= 50_000  # warm-up step excluded
    assert s["loss"] == 5.10291  # loss is always the last step's

    from ascend_titan.tools.bench import Row

    md = render([Row(module="m", config="c", ngpu=1, note="boom")], "torchX_npuY")
    assert "🔴" in md and "provenance" in md


def test_card_pool_hands_out_ascending_lists():
    """ASCEND_RT_VISIBLE_DEVICES must be ascending: torch_npu reports zero devices
    for an unsorted list (NPU-10), and the pool naturally produces one after a release."""
    from ascend_titan.tools.matrix import CardPool

    pool = CardPool([0, 1, 2, 3])
    first = pool.acquire(2)
    second = pool.acquire(2)
    pool.release(first)
    assert pool.acquire(2) == first == sorted(first)
    assert second == sorted(second)

    pool = CardPool([4, 5, 0, 1])
    assert pool.acquire(4) == [0, 1, 4, 5]


def test_hccl_base_port_is_per_card_set_and_out_of_the_default_range():
    """HCCL's default port collides with other jobs on a shared box (HARNESS reds)."""
    from ascend_titan.tools.matrix import hccl_base_port

    assert hccl_base_port([0, 1]) != hccl_base_port([2, 3])
    assert hccl_base_port([4, 5]) == hccl_base_port([4, 5, 6, 7])  # keyed on the lowest card
    assert all(hccl_base_port([c]) >= 61000 for c in range(8))


@pytest.mark.titan
def test_npu_minimal_skips_both_attention_overrides_under_a_block_override(npu_stub):
    """OURS-9: torchtitan rejects an override claiming a descendant of another's node.

    `fused_mla` claims `layers.N.attention`, which is an ancestor of both the inner
    attention node and the RoPE node, so neither of our overrides may be added.
    """
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    from ascend_titan.recipes.transforms import ATTENTION_OVERRIDE, ROPE_OVERRIDE, npu_minimal

    cfg = llama3_debugmodel()
    cfg.override.imports = ["torchtitan.overrides.fused_mla.fused_mla"]
    a = npu_minimal(cfg)
    assert ATTENTION_OVERRIDE not in cfg.override.imports
    assert ROPE_OVERRIDE not in cfg.override.imports
    assert not a.attention_override and not a.rope_override
    assert any("attention block" in n for n in a.notes)


def test_bench_normalises_the_card_spec():
    """``--cards 0-7`` must become "0,1,...,7", not reach the env verbatim.

    ``ASCEND_RT_VISIBLE_DEVICES`` understands a comma list only, and it must be
    ascending or torch_npu reports zero devices (NPU-10). Passing the matrix's
    range syntax straight through is silently fatal: every multi-card row comes
    back "rc=1, 0 steps parsed" with nothing pointing at the cause.
    """
    from ascend_titan.tools.matrix import parse_cards

    cards = sorted(parse_cards("0-7"))
    assert ",".join(str(c) for c in cards[:8]) == "0,1,2,3,4,5,6,7"
    assert ",".join(str(c) for c in cards[:1]) == "0"
    assert sorted(parse_cards("4,5,0,1")) == [0, 1, 4, 5]
