# 变更日志

所有值得记录的变更都在这里。格式：[Keep a Changelog](https://keepachangelog.com/)；0.1.0 之后遵循 SemVer。

## [Unreleased]
### 新增
- 包骨架：无副作用的 import、`setup()` 引导、`python -m ascend_titan.train` 入口。
- shim 注册表，支持 `wrap` / `replace` / `polyfill` 三种类型；P3/P4 在 import 时强制。
- `ascend-titan-doctor` 环境探测。
- L1 override：`kernels/attention.py`（VarlenAttention → `torch_npu.npu_fusion_attention`）、`kernels/rope.py`（ComplexRoPE 改为实数缓存，torch_npu 不能索引复数张量）。
- shim：`torch.distributed.set_timeout` 的 polyfill（torch ≤ 2.13）；PP `step(arg_mbs=...)` 预切分 microbatch 转发。
- Qwen3 recipe（`qwen3_debugmodel_npu`、`_fsdp2`、矩阵变体）；`recipes/transforms.py::npu_baseline` 通用变换；`recipes/matrix.py` 动态 recipe 模块。
- `tools/matrix.py`：把上游集成用例搬到 NPU 上扫描并自动归因。
- 基线（`docs/baseline.md`）：NEXT（torch 2.13.0 / torch_npu 2.13.0rc1）与 STABLE（2.12.0 / 2.12.0），torchtitan `13da2d77c`；golden loss 曲线与 `scripts/check_golden.sh`。
- 能力矩阵、问题清单（`docs/issues/`）、ADR-001..005、Claude Code skill 与规则。
- 开源工程：Apache-2.0、CONTRIBUTING、行为准则、SECURITY、issue/PR 模板、pre-commit、CPU CI、NPU nightly workflow 骨架。
