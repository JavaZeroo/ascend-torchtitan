"""L1: fused-kernel overrides (torchtitan ``@override`` factories).

Modules here are imported by torchtitan's override mechanism. The only import-time
side effect allowed is an optional-addon probe that logs a loud WARNING when the
accelerator package is absent (ADR-004); no module may load an NPU op or reach
fla_npu.ops.ascendc at import. See .claude/skills/override-authoring.
"""
