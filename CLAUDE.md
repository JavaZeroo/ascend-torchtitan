# ascend-torchtitan —— 开发指南

[torchtitan](https://github.com/pytorch/torchtitan) 的昇腾 NPU 树外扩展。我们**扩展，不 fork**：torchtitan 是按 commit SHA 固定的已安装依赖。

先读：`docs/PRINCIPLES.md`（P0–P7，评审按编号引用）、`docs/glossary.md`（override ≠ OverrideDefinitions，recipe ≠ torchtitan_recipes）、`docs/roadmap.md`（M0–M5）、`docs/baseline.md`（已验证的版本元组）。

## 布局

| 路径 | 层 | 内容 |
|---|---|---|
| `ascend_titan/_bootstrap.py`、`train.py` | 入口 | `setup()` = 唯一的副作用点；必须先于任何 `import torchtitan` |
| `ascend_titan/compat/` | L0 | shim 注册表 + `shims/`（受治理的 monkeypatch；数量 → 0） |
| `ascend_titan/kernels/` | L1 | 封装昇腾算子的 `@override` 工厂（attention = custom_op、rope、rms_norm） |
| `ascend_titan/parallel/`、`graph/` | L2 | 并行策略、torchair |
| `ascend_titan/recipes/` | L3 | `Trainer.Config` = 上游 registry 函数 + 增量；`transforms.npu_baseline`；`matrix.py` 动态 recipe |
| `ascend_titan/tools/` | L4 | `doctor`（环境探测）、`matrix`（扫描 + 归因）、`provenance`（实际实现审计） |
| `constraints/` | — | 固定的 torchtitan SHA + pip 约束（**版本的唯一事实来源**） |
| `docs/capability-matrix.md` | — | 每个特性 🟢/🔴/⚪，带归因 |
| `docs/upstream-tracking.md` | — | shim ↔ issue 表、上游 ask |
| `docs/issues/` | — | 按归属分的问题清单（torch_npu / pytorch / torchtitan / ours）——可直接贴去提 issue |
| `docs/baseline.md` | — | 已验证的版本元组（NEXT / STABLE）与 M1 数据 |
| `tests/assets/losses/npu/` | — | 冻结的 golden loss 曲线（`scripts/check_golden.sh`） |

同级检出 `../torchtitan` 是固定的上游；随便读，绝不改。

## 命令

```bash
WITH_TORCH=1 ./scripts/install.sh         # torch+torch_npu（NEXT track）+ 固定 SHA 的 torchtitan + 本包
CONSTRAINTS=constraints/npu-stable.txt WITH_TORCH=1 ./scripts/install.sh   # STABLE track
ascend-titan-doctor                       # 环境探测（CPU 上也能跑）
pytest tests/unit -x                      # CPU 单测（标记 titan 的测试需要 torchtitan）
ruff check . && ruff format --check .
./scripts/probe_compat.sh                 # 固定 SHA 最远能前进到哪
MODULE=ascend_titan.recipes.qwen3 CONFIG=qwen3_debugmodel_npu NPU=8 ./scripts/run_train.sh
COMM_MODE=fake_backend NPU=8 ./scripts/run_train.sh   # 单设备，fake 进程组（目前 🔴 NPU-2）
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 ./scripts/check_golden.sh qwen3_debugmodel_npu   # 确定性运行 vs golden
python -m ascend_titan.tools.matrix --cards 0-7 --jobs 4   # 把上游测试配置搬到 NPU 上扫描（docs/matrix/）
ascend-titan-provenance --module ascend_titan.recipes.qwen3 --config qwen3_debugmodel_npu   # 每个节点实际用的实现
# tyro：`activation-checkpoint:none` 之类的子命令必须放在所有 --flag 之后
```

## 硬规则

1. **`import ascend_titan` 无副作用**（`tests/unit/test_import_purity.py`）。torchtitan 会在 `Trainer.__init__` 内部导入 `ascend_titan.kernels.*`；import 期副作用会在初始化中途触发。
2. **绝不绕过 torch_npu**（P1）。归因为 `NPU` ⇒ 提 issue、标 🔴、停。
3. **先配置再 shim**（P0）。先查 `torchtitan/config/configs.py` 有没有开关。
4. **shim 用包装**（P3）并**挂上游 issue**（P4）；注册表两者都强制。
5. **override 只针对已有的 `Configurable` 节点**（P6）。没有节点 ⇒ 上游 ask，不替换父块。
6. **recipe 是增量**（`tests/unit/test_recipes.py::test_recipe_is_delta_not_copy`）。
7. **版本升级走带矩阵结果的 PR**（P5）。绝不随手改 `constraints/torchtitan.sha`。
8. **不加投机性的兜底。** 与上游同一标准：只校验显式契约。
9. **pip 安装永远带 `-c constraints/<track>.txt`**——否则 torch 被升级、torch_npu ABI 损坏。
10. **改 `matrix.py` 的 `TRIAGE` 表或任何被 ruff 重排过的多行元组时，用行定位插入，不要用整行字符串 replace**（本仓已因此丢过三次编辑）。

## 失败归因（经常用）

| traceback 中第一个非框架帧 | 代码 | 动作 |
|---|---|---|
| `torch_npu/...` 或 `op_plugin ... aclnnXxx failed` | NPU / NPU-OP | torch_npu issue，🔴，**不 workaround** |
| `torchtitan/...` 附近有 `cuda`/`nccl` 字样，或 nightly-only API | TT | 上游 issue + 包装型/polyfill shim 候选 |
| CANN 错误码（`EZ9999`、`EI0002`……；`ERR99999` 只是通用包装，不算） | CANN | 记录，停 |
| `attn_gym`/`helion`/`deep_ep`/`cutlass`/`torchao`/`fla` | DEP | 记录；昇腾替代是 L1 任务 |
| `torch/` 自身（设备白名单、缺失的公开 API） | TORCH | pytorch issue；纯改名/别名才做 polyfill shim |
| `EI0020` 端口已绑定 | HARNESS | 同卡有其它 HCCL 作业，重跑 |

## 数值验证
与上游同一标准：非计算类改动在 `--debug.seed=42 --debug.deterministic` 下 loss 必须**完全一致**；计算类改动（算子）需要对上游 eager 路径的对齐测试加 `torch.library.opcheck`。绝不使用 `--debug.deterministic_warn_only`。

## 开发容器
NPU 主机上的 `ascend-titan-dev`（镜像 cann:9.1.0-910b-ubuntu22.04-py3.12-devel）：系统 python = STABLE track（torch 2.12.0），`/opt/venv213` = NEXT track（torch 2.13.0 / torch_npu 2.13.0rc1）。`/data` 是共享 NFS；仓库在容器内路径相同。8 张卡通常与其它作业共享——用 `ASCEND_RT_VISIBLE_DEVICES` 选卡；同一张卡上不能并发两个 HCCL 作业（EI0020）。AscendC 算子库（ops-nn 等）**不能在 NFS 上构建**（flock 拿不到锁 → ES-GEN 轮询超时），先 rsync 到 `/opt/build/`。

## Skill
`.claude/skills/`：`compat-probe`（M0）、`shim-authoring`、`override-authoring`、`capability-matrix`（记录格子 + 归因、跑扫描）、`upstream-sync`（升级 SHA）。`.claude/rules/` 里的规则按路径生效。
