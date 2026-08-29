# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/); versions follow SemVer once 0.1.0 is cut.

## [Unreleased]
### Added
- Package skeleton: side-effect-free import, `setup()` bootstrap, `python -m ascend_titan.train` entry.
- Shim registry with `wrap` / `replace` / `polyfill` kinds; P3/P4 enforced at import time.
- `ascend-titan-doctor` environment probe.
- First L1 override: `ascend_titan.kernels.attention.npu_fusion_attention` (VarlenAttention → `torch_npu.npu_fusion_attention`).
- First shim: polyfill of `torch.distributed.set_timeout` for torch ≤ 2.13.
- Qwen3 recipes (`qwen3_debugmodel_npu`, `_fsdp2`, matrix variants).
- Capability matrix, problem lists (`docs/issues/`), ADR-001..005, Claude Code skills and rules.
- Baseline (`docs/baseline.md`): NEXT (torch 2.13.0 / torch_npu 2.13.0rc1) and STABLE (2.12.0 / 2.12.0) tracks, torchtitan `13da2d77c`; golden loss curves and `scripts/check_golden.sh`.
- Open-source scaffolding: Apache-2.0, CONTRIBUTING, CoC, SECURITY, issue/PR templates, pre-commit, CPU CI, NPU nightly workflow skeleton.
