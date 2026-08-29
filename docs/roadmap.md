# Roadmap

Each milestone introduces only the mechanism it needs. Directories exist from day one (with a README) so later work has a home; code does not.

| M | goal | new mechanism | acceptance |
|---|---|---|---|
| **M0** ✅ 2026-08-29 | Compatibility probe | `tools/doctor.py`, `scripts/probe_compat.sh`, `constraints/` | Four-way version tuple recorded in `constraints/npu.txt`; furthest-importable torchtitan SHA known; torch_npu autoload answered; torch_npu missing-API list filed as issues. **Go/no-go decided here.** |
| **M1** ✅ 2026-08-29 | Qwen3 runs | `setup()`, shim registry (with ≥0 shims), `recipes/qwen3.py` | `qwen3_debugmodel_npu` 10 steps: ① 1 NPU eager ② `--comm.mode=fake_backend` ③ FSDP2 ×2. Loss within loose tolerance of upstream `tests/assets/losses/*/qwen3_a10g.txt`. NPU golden loss frozen. **Happened as predicted:** flex and varlen both fail on NPU, so the inner-attention override (`kernels/attention.py`) shipped in M1. fake_backend path is 🔴 NPU-2 and stays so until torch_npu registers the fake backend. |
| **M2** | Capability matrix | upstream `tests/integration_tests` runner on NPU, matrix three-state, nightly CI pinned + furthest-SHA, attribution auto-triage | Matrix has no ⚪ in the M2 axes (parallel × attention × AC × compile). Every 🔴 has an attribution. |
| **M3** | Override mechanism hardened | provenance table, `opcheck`, second override (SituGLU or RMSNorm), LSE epilogue for attention (CP) | Provenance in every run log; two overrides with alignment tests. |
| **M4** | Kimi-K3 + fused ops | `parallel/` (moonep, CP), automatic bisect on nightly reds, perf baselines | kimi_k3 recipe with KDA / conv1d / SituGLU / attn_res, perf baseline recorded with provenance. |
| **M5** | Graph mode / low precision / multimodal | `graph/` (torchair), FP8 override on the post-converter tree, multimodal axes | Matrix axes extended; no new mechanism types. |

Explicitly deferred decisions (revisit with data): shim source fingerprints (only if replace-type shims appear), `AscendTrainer` subclass (no known need), vendor-agnostic middle layer (no).
