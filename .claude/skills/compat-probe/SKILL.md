---
name: compat-probe
description: M0 流程——在 NPU 机器上确定 torch / torch_npu / CANN / torchtitan 版本元组，找到最远可 import 的 torchtitan SHA，回答 torch 是否自动加载 torch_npu，并为 torch_npu 缺失的 API 提 issue。用于搭建新环境、`ascend-titan-doctor` 显示不匹配、或升级 torchtitan SHA 之前。
---
# compat-probe

目标：用事实而不是希望填满 `constraints/`。产出是一份贴进 PR 的短报告。

## 步骤
1. 在 NPU 机器上 `ascend-titan-doctor --json > /tmp/doctor.json`。记录 torch、torch_npu、CANN、`torch_npu_autoload`。若 autoload 为 `false`，`python -m ascend_titan.train` 就是强制的（F4）——报告里写明。
2. `WITH_TORCH=1 ./scripts/install.sh`（使用固定 SHA）。若 `import torchtitan.trainer` 失败：
   - 读 traceback；按 CLAUDE.md 的归因表分类；
   - 缺失的 `torch.*` API ⇒ 这就是 torch 版本缺口（F1）。记下 API 和引入它的上游 commit：`git -C ../torchtitan log -S '<api name>' --oneline | tail -1`。
3. `./scripts/probe_compat.sh`——记录最远可 import 的 SHA 和第一个破坏性 commit。
4. 冒烟：`ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 ./scripts/check_golden.sh qwen3_debugmodel_npu`，然后 FSDP2×2。
5. 填 `docs/capability-matrix.md` 的"运行路径"行和 `constraints/` 里的 pin。torch_npu 每个缺失 API 在 `docs/issues/torch_npu.md` 起草。不做 workaround（P1）。

## 报告模板
```
tuple: torch=… torch_npu=… CANN=… torchtitan=<sha>（固定）furthest=<sha>
autoload: yes|no
import torchtitan.trainer: ok | 在 <module> 失败: <api>（由 <commit> 引入）
golden 1-NPU: 🟢|🔴 <归因>
golden FSDP2: 🟢|🔴 <归因>
起草的 issue: …
go/no-go: …
```
