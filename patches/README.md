# patches —— 本地保存的上游修复方案

按用户决定（2026-08-29）：torchtitan / pytorch 的问题**不向上游提 issue**，先在这里保存可 review 的修复方案；
torch_npu 的问题解决并验证后，可以向 `gitcode.com/Ascend/pytorch` 提 issue 和 PR。

| 目录 | 基线 | 生成方式 |
|---|---|---|
| `torchtitan/` | `constraints/torchtitan.sha`（13da2d77c） | 在隔离克隆 `/opt/build/titan-work`（分支 `ascend-patches`）上修改，`git format-patch` 导出 |
| `pytorch/` | 已安装的 torch 2.12.0 / 2.13.0 源码（wheel 展开） | 手写 diff，在 site-packages 副本上验证 |
| `torch_npu/` | `../ascend-pytorch`（gitcode Ascend/pytorch main） | 修改后 `git format-patch`；验证方式见各补丁头部 |

每个补丁文件头部写明：对应问题编号、根因、验证命令与结果。状态汇总在 `docs/issues/STATUS.md`。

应用方式（验证用）：
```bash
git -C /opt/build/titan-work am patches/torchtitan/*.patch     # 或 git apply
pip install --no-deps -e /opt/build/titan-work                 # 在实验 venv 里
```
