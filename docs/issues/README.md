# Problem lists

Everything that stops torchtitan from running as-is on Ascend, split by **who owns the fix**. Each entry
is written so it can be pasted into an issue tracker; once filed, the URL replaces the `draft:` pointer
in code/docs. Status legend: `draft` (text ready, not filed) · `filed` (link) · `fixed@<version>` · `wontfix`.

| owner | file | what goes here |
|---|---|---|
| torch_npu | [torch_npu.md](torch_npu.md) | missing ops / backend registrations in torch_npu (P1: never worked around here) |
| pytorch | [pytorch.md](pytorch.md) | device whitelists and extension points in torch core |
| torchtitan | [torchtitan.md](torchtitan.md) | nightly-only API use without feature checks, unconditional CUDA-only imports, missing `Configurable` nodes |
| ascend-torchtitan | [ours.md](ours.md) | known gaps in this repository |

Attribution codes (CLAUDE.md): **TT** torchtitan · **NPU** torch_npu · **CANN** · **DEP** third-party CUDA-only dep · **TORCH** pytorch core.
