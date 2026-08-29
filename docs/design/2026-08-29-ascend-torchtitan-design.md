# ascend-torchtitan — Design

Status: accepted 2026-08-29 (v1). Owner: ascend-torchtitan maintainers.
Companion documents: `docs/PRINCIPLES.md`, `docs/roadmap.md`, `docs/adr/`, `docs/upstream-tracking.md`.

## 1. Goals and non-goals

**Goals**
1. `pip install` this package + `scripts/install.sh` ⇒ torchtitan runs in eager mode on Ascend NPUs.
2. Host Ascend performance work — fused kernels (KDA, causal_conv1d, SituGLU, attn_res, fusion attention, norms), parallel strategies (moonep EP, CP), graph mode (torchair) — as **pluggable, opt-in** pieces.
3. Do so **without forking torchtitan**, and survive its ~170 commits/month.
4. Stay readable: a newcomer can tell, from one recipe file, exactly which Ascend pieces a run uses.

**Non-goals**
- A vendor-agnostic middle layer (ADR-005).
- Working around torch_npu defects (P1).
- Tracking torchtitan `experiments/` (upstream itself does not gate core on them).

## 2. What upstream already provides (verified against torchtitan @ 13da2d77c)

| extension point | evidence | what it lets us do without patching |
|---|---|---|
| `--module <full.path>` | `torchtitan/config/manager.py:126-135` | out-of-tree `config_registry` modules → our recipes |
| `config.build()` resolves `Config._owner` | `train.py:43`, `docs/extension.md` | subclass `Trainer`/`Trainer.Config` if ever needed |
| `override.imports` | `torchtitan/config/override.py`, `overrides/README.md` | replace **any** `Configurable.Config` node; per-instance `fqns`; per-node conflict detection; `derive()` for field-robust config construction; per-entry kwargs |
| `ModelSpec` callables | `protocols/model_spec.py:33` | swap `parallelize_fn`, `pipelining_fn`, `state_dict_adapter` |
| device detection | `tools/utils.py:54-60`, `distributed/utils.py:498` | `device_type` from `torch._utils._get_available_device_type()` (npu via privateuse1); backend from `Backend.default_device_backend_map` (→ hccl) |
| already-guarded CUDA paths | `distributed/cudagraph.py:342`, `configs.py:296-306`, quantization modules | CUDA graphs fall back to eager on non-CUDA; `compile.backend` is a plain string; float8/mx/nvfp4 are opt-in |

Upstream's override README states the intent directly: vendor kernels belong in an external package activated by `override.imports`; `experiments/README.md` §4 forbids vendor code in-tree. Our layout is the shape upstream asked for.

## 3. Architecture

```
python -m ascend_titan.train  ──setup()──►  torch_npu import + L0 shims  ──►  torchtitan.train.main()
                                                                                    │
       recipe (L3): cfg = upstream_registry_fn(); deltas; cfg.override.imports=[L1 modules]
                                                                                    │
                       Trainer.__init__: apply_overrides → imports ascend_titan.kernels.* (L1)
                                          ModelSpec.parallelize_fn ← ascend_titan.parallel (L2)
                                          compile.backend="torchair" ← ascend_titan.graph (L2)
```

| layer | package | mechanism | health metric |
|---|---|---|---|
| L0 compat | `ascend_titan.compat` | governed monkeypatch registry; applied only by `setup()` | shim count → 0 |
| L1 kernels | `ascend_titan.kernels` | `@override` factories + `torch.library.custom_op` | each op: opcheck + alignment test |
| L2 parallel / graph | `ascend_titan.parallel`, `.graph` | `ModelSpec` callables, `compile.backend` | end-to-end perf baseline |
| L3 recipes | `ascend_titan.recipes` | upstream registry fn + deltas | number of 🟢 cells with baselines |
| L4 tools | `ascend_titan.tools` | doctor, provenance, alignment helpers | — |

### 3.1 Bootstrap and the import-order constraint (F4)
`torchtitan/tools/utils.py:60` evaluates `device_type` at import time. `setup()` therefore runs before any torchtitan import; `ascend_titan.train` exists solely to guarantee that order. `setup()` probes whether torch autoloaded `torch_npu` (`torch._import_device_backends`) and reports it; the entry point is required regardless because L0 shims must precede torchtitan imports too.

`import ascend_titan` is side-effect free (tested) because torchtitan imports our L1 modules from inside `Trainer.__init__`.

### 3.2 Configurable-node criterion (P6)
For every computation we want to fuse: does upstream have a `Configurable.Config` node whose `forward` contains it?

| kimi_k3 op | upstream node | verdict |
|---|---|---|
| `chunk_kda` | `kda.py:48 KDAKernel.Config` | override directly |
| SituGLU | `moe.py:41 SiTUFeedForward` / `:60 SiTUGroupedExperts` | override the parent node (same granularity as upstream `fused_swiglu.py`) |
| `causal_conv1d` | inside `InnerKDA.forward` (`kda.py:150`) | override `InnerKDA.Config`; coarse but self-contained |
| attn_res | `model.py:135 _apply_attention_residual`, free function | **no node** → upstream ask (extracting a `Module` also resolves upstream's `TODO: Add TP Support`) |
| inner attention (all LMs) | `attention.py:222 FlexAttention.Config`, `:126 VarlenAttention.Config` | override directly with an `npu_fusion_attention`-backed module; **may be needed in M1** because upstream removed the eager `sdpa` LM path (`config_utils.py:97`) |

### 3.3 Shim governance (ADR-002)
`@shim(target, reason, upstream, kind, why_not_wrap)`; registry rejects missing `upstream` or replace-without-reason at import. Applied once, idempotently, with a log line per shim. `doctor` lists registered shims. Wrapper-type shims inherit upstream changes; replace-type shims are the only ones that would justify source fingerprints (deferred).

### 3.4 Degradation and provenance (ADR-004, P7)
Missing kernel dependency ⇒ the L1 module logs a WARNING and does not register its override ⇒ upstream eager runs. Provenance (M3) records per node which backend ran; benchmarks must carry it.

## 4. Dependency and version management (F1–F3) — the least-validated part

Measured facts:
- torchtitan main targets PyTorch **nightly**; even releases pin nightly dates (`docs/release.md:10-12`). torch_npu targets torch **releases**.
- Last release v0.2.2 (2026-02-20) lacks the override mechanism (2026-06-10) and kimi_k3 (2026-08-24). ⇒ pin by SHA (ADR-003).
- `attn-gym[linear]==0.0.5` is hard-pinned; the extra pulls `nvidia-cutlass-dsl[cu13]`. ⇒ `scripts/install.sh` uses `--no-deps` + `constraints/titan-deps.txt`. attn_gym's KDA has a `naive` torch fallback, so kimi_k3 still runs without the extra.
- nightly-only APIs in use: `torch.nn.attention.varlen`, `torch.distributed._symmetric_memory`.

Mechanisms:
- `constraints/npu.txt` = the four-way tuple + pinned SHA (a deliverable).
- `scripts/probe_compat.sh` = furthest importable SHA / first breaking commit.
- CI legs: **pinned** (gate) and **main** (drift probe, may fail). On NPU, the main leg is replaced by the furthest-importable-SHA leg, because main may be red purely for torch-version reasons.
- Reading the two legs: 🟢/🟢 fine · 🟢/🔴 upstream drift (fix before next bump) · 🔴/🔴 our bug · 🔴/🟢 upstream already fixed it.

**M0 exists to turn this section from measured-on-CPU into validated-on-NPU.** Go/no-go is decided there.

## 5. Testing strategy

| tier | where | runs on | gate? |
|---|---|---|---|
| unit: import purity, shim registry, doctor, recipe construction | `tests/unit` | CPU, every PR, both CI legs | yes (pinned leg) |
| per-shim / per-kernel CPU tests | `tests/unit` | CPU | yes |
| upstream integration suites (`features`, `models`; 57 cases) | upstream `tests/integration_tests/run_tests.py` from the pinned checkout, device env var adapted | NPU nightly, 64 cards / 8 h budget (~20 % used by the matrix) | pinned leg yes |
| numerics: opcheck + alignment vs upstream eager; loss vs `qwen3_a10g.txt` (loose) and NPU golden (tight) | `tests/npu` | NPU nightly | yes once golden exists |
| perf baselines with provenance | `benchmarks/` (M4) | NPU nightly | threshold alert |
| automatic bisect on nightly red | script (M4) | NPU, ~4 card-hours per red | — |

Budget priority when the 8 h window overruns: smoke → pinned matrix → furthest-SHA matrix → bisect → perf → version matrix; tail gets cut and the report says so.

## 6. Collaboration model
One upstream, many independent vendor packages. Ownership test: *would a second vendor need it?* Yes ⇒ upstream (device capability query, device-graph abstraction, attention backend registry, `Module` extraction of attn_res). No ⇒ here. Upstream PRs are filed **only when not filing means carrying a fork or a patch indefinitely**; otherwise contribute bug reports from the drift leg, which cost upstream no review effort.

## 7. Risks

| risk | likelihood | mitigation |
|---|---|---|
| No torch version satisfies both torchtitan@SHA and torch_npu (F1) | medium | M0 probe; fall back to an older SHA; file torch_npu API issues; **no-go if empty intersection** |
| Neither `flex` nor `varlen` attention runs on NPU (no eager LM attention upstream) | medium-high | inner-attention override becomes the first L1 module, pulled into M1; discovered by the CPU recipe test on day one |
| kimi_k3 upstream churn (days old, eager-only) | high | no kimi_k3 overrides before M4; attn_res upstream ask waits for stabilisation |
| FSDP2/DTensor over HCCL broadly red | unknown | M1 path ③ answers this; if red with NPU attribution, roadmap pauses at M1 (P1) |
| Silent perf degradation via loud-but-ignored fallback | medium | provenance mandatory in benchmarks; nightly perf thresholds |
| Shim count creeps up | medium | P4 import-time enforcement, doctor reporting, `upstream-sync` skill step 6 |

## 8. Open items (deferred with data)
- Source fingerprints for replace-type shims — only if any appear.
- `AscendTrainer` subclass — no known need.
- Whether attn_gym's Triton KDA fwd compiles under Triton-Ascend — 1–2 day probe in M3.
- Repository hosting / licence — decide before first external release.
