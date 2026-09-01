# tests/repro —— 最小复现与探测脚本

不是 pytest 用例（文件名不以 `test_` 开头），是提 issue 时附带的**最小复现**和基线探测：

| 脚本 | 用途 | 挂靠 |
|---|---|---|
| `probe_torch_apis.py` | CPU 上探测 torch API 缺口（TORCH-1/7/8、NPU-2 的设备表）——升级 torch nightly 日期前先跑 | 基线升级流程 |
| `probe_npu_gaps.py` | NPU 上逐项探测 NPU-1/2/3/6、TORCH-1（flex eager）、TT-4；每行 `[OK ]` / `[ERR]` 即矩阵归因 | docs/baseline.md、STATUS.md |
| `probe_flex_deterministic.py` | 确定性模式下编译版 flex 在昇腾上的两种败因（TT-12 shim 的证据） | docs/capability-matrix.md |
| `probe_solve_tri_crosscheck.py` | fla-npu `solve_tri` 对独立第二实现的互证（R5） | docs/roadmap.md |

规则：

- 归因 NPU 的问题在这里先写出最小复现，再走 P9（修复 → 构建 → 验证 → gitcode issue + PR）；issue 正文里的复现代码从这里复制。
- **每个探针有生命周期**：表里必须写清它挂靠的 issue / 文档；挂靠的 issue 关闭或文档不再引用时，探针随之删除（与 shim、patch 同一条规矩）。
- 一次性的探索脚本**不进本目录**（也不进 git）——放 `outputs/`（已 gitignore），得出的结论落到 `docs/`，脚本本身不留。
