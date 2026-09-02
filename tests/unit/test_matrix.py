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

    from ascend_titan.recipes.deltas import _flex_attention_is_usable
    from ascend_titan.recipes.matrix import ATTENTION_OVERRIDE, npu_minimal

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

    from ascend_titan.recipes.matrix import ROPE_OVERRIDE, npu_minimal

    cfg = llama3_debugmodel()
    cfg.override.imports = ["torchtitan.overrides.fused_mla.fused_mla"]
    a = npu_minimal(cfg)
    assert ROPE_OVERRIDE not in cfg.override.imports and not a.rope_override


@pytest.mark.titan
def test_npu_fused_is_opt_in_and_idempotent():
    """The perf transform is separate from the baseline and safe to re-apply."""
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    from ascend_titan.recipes.matrix import RMSNORM_OVERRIDE, npu_fused, npu_minimal

    cfg = llama3_debugmodel()
    npu_minimal(cfg)
    assert RMSNORM_OVERRIDE not in cfg.override.imports
    a = npu_fused(cfg)
    assert a.rms_norm_override and RMSNORM_OVERRIDE in cfg.override.imports
    npu_fused(cfg)
    assert cfg.override.imports.count(RMSNORM_OVERRIDE) == 1


def test_fused_that_fuses_nothing_is_loud_and_not_reportable():
    """P7/P12: a config no fused kernel targets must not yield a "fused" measurement.

    qwen3_5 is the real case: it carries no ``models.common.nn_modules.RMSNorm``
    node at all, so ``--mode fused`` used to produce a config byte-identical to
    ``minimal`` and report the number as if the fused kernels had run.
    """
    from ascend_titan.recipes.matrix import TransformReport, npu_fused

    empty = TransformReport()
    assert empty.is_noop
    assert not TransformReport(rms_norm_override=True).is_noop

    class _NothingToFuse:
        """A config tree with no nodes and no active overrides."""

        class _Override:
            imports: list[str] = []

        override = _Override()

        def traverse(self, _cls):
            return iter(())

    report = npu_fused(_NothingToFuse())
    assert report.is_noop
    assert any("measures exactly what 'minimal' does" in n for n in report.notes)


def test_undecidable_is_not_the_same_as_nothing_to_fuse(monkeypatch):
    """A config that cannot be built here must still run, not be skipped as a no-op.

    Skipping on a build failure would hide a real red cell behind a ⚪.
    """
    import ascend_titan.recipes.matrix as matrix_mod
    from ascend_titan.tools.matrix.cases import fused_is_a_no_op

    def unbuildable(_name):
        raise RuntimeError("this config needs a package we do not have")

    monkeypatch.setattr(matrix_mod, "resolve", unbuildable)
    assert fused_is_a_no_op("some.module__some_fn") is False

    monkeypatch.setattr(matrix_mod, "resolve", lambda _n: lambda: object())
    monkeypatch.setattr(matrix_mod, "npu_fused", lambda _c: matrix_mod.TransformReport())
    assert fused_is_a_no_op("some.module__some_fn__fused") is True


def test_swap_override_replaces_sibling_idempotently():
    """Two overrides claiming one node swap, never stack; re-swap is a no-op."""
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    from ascend_titan.recipes.matrix import GDN_FUSED_OVERRIDE, GDN_OVERRIDE, swap_override

    cfg = llama3_debugmodel()
    cfg.override.imports = ["ascend_titan.kernels.attention.npu_fusion_attention", GDN_OVERRIDE]

    assert swap_override(cfg, remove=GDN_OVERRIDE, add=GDN_FUSED_OVERRIDE)
    assert GDN_OVERRIDE not in cfg.override.imports
    assert GDN_FUSED_OVERRIDE in cfg.override.imports
    # attention override is untouched; sibling relationship is ordered, not positional
    assert cfg.override.imports[0] == "ascend_titan.kernels.attention.npu_fusion_attention"

    # Idempotent: the target is gone, so a second swap is a no-op that keeps the
    # existing order and does not append a duplicate.
    assert not swap_override(cfg, remove=GDN_OVERRIDE, add=GDN_FUSED_OVERRIDE)
    assert cfg.override.imports.count(GDN_FUSED_OVERRIDE) == 1


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

    from ascend_titan.recipes.matrix import ATTENTION_OVERRIDE, ROPE_OVERRIDE, npu_minimal

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


def test_cases_run_on_the_interpreter_the_tool_runs_on():
    """A sweep must not inherit which python it runs on from the ambient PATH.

    Measured 2026-09-02: launching the tool as
    ``/opt/venv-nightly/bin/python -m ascend_titan.tools.matrix`` without that
    venv on PATH made ``run_train.sh`` pick the system ``torchrun``, which has no
    Triton-Ascend, and all 13 cases came back red with "0 active drivers" --
    a full sweep of false HARNESS attributions.
    """
    import inspect
    import pathlib
    import sys

    from ascend_titan.tools.matrix import runner

    source = inspect.getsource(runner)
    assert '"PYTHON": sys.executable' in source, "the runner must pin the interpreter"

    script = (pathlib.Path(runner.__file__).parents[3] / "scripts" / "run_train.sh").read_text()
    assert "PYTHON=${PYTHON:-python}" in script
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("torchrun "), (
            f"bare torchrun resolves through PATH, not PYTHON: {stripped}"
        )
    assert sys.executable  # the value the runner forwards


def test_a_harness_flake_is_retried_once_before_it_becomes_a_red_cell(monkeypatch):
    """HARNESS means the run measured the box, not the code, so it must not stand.

    Measured 2026-09-02: a serial CP sweep lost four cases to
    "port 16666 have already been bound" left behind by the case before them,
    and the triage note said "rerun" while nothing reran.
    """
    from ascend_titan.tools.matrix import runner

    calls = []

    def fake(case, cards, out, repo, timeout):
        calls.append(1)
        if len(calls) == 1:
            return runner.Result(
                "features", "cp", 4, "red", code="HARNESS", note="HCCL port conflict"
            )
        return runner.Result("features", "cp", 4, "green")

    monkeypatch.setattr(runner, "_run_case_once", fake)
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)

    r = runner.run_case(runner.Case("features", "cp", "", 4, [], []), [0], None, None, 1)
    assert len(calls) == 2, "a HARNESS red must be retried exactly once"
    assert r.state == "green"
    assert "HCCL port conflict" in r.retried_after


def test_a_real_red_is_never_retried(monkeypatch):
    """Only HARNESS is a flake. Retrying a genuine failure just doubles the sweep."""
    from ascend_titan.tools.matrix import runner

    calls = []

    def fake(case, cards, out, repo, timeout):
        calls.append(1)
        return runner.Result("features", "cp", 4, "red", code="CANN", note="no kernel")

    monkeypatch.setattr(runner, "_run_case_once", fake)
    r = runner.run_case(runner.Case("features", "cp", "", 4, [], []), [0], None, None, 1)
    assert len(calls) == 1
    assert r.code == "CANN"
    assert r.retried_after == ""
