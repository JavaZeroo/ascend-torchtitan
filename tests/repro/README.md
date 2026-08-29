# tests/repro —— 最小复现与探测脚本

不是 pytest 用例（文件名不以 `test_` 开头），是提 issue 时附带的**最小复现**和基线探测：

| 脚本 | 用途 |
|---|---|
| `probe_torch_apis.py` | CPU 上探测 torch 版本缺口（TT-2/8/9、TT-5、TORCH-1/7/8、NPU-2 的设备表）——升级 torch nightly 日期前先跑 |
| `probe_npu_gaps.py` | NPU 上逐项探测 NPU-1/2/3/6、TORCH-1（flex eager）、TT-4；每行 `[OK ]` / `[ERR]` 即矩阵归因 |

规则：归因 NPU 的问题在这里先写出最小复现，再走 P9（修复 → 构建 → 验证 → gitcode issue + PR）；issue 正文里的复现代码从这里复制。
