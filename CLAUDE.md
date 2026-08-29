# ascend-torchtitan —— 开发指南

[torchtitan](https://github.com/pytorch/torchtitan) 的昇腾 NPU 树外扩展。我们**扩展，不 fork**：torchtitan 是按 commit SHA 固定的已安装依赖。（`AGENTS.md` 是本文件的符号链接。）

先读：`docs/PRINCIPLES.md`（P0–P13，评审按编号引用）、`docs/adr/ADR-006-nightly-first.md`（基线定义）、`docs/glossary.md`、`docs/roadmap.md`、`docs/baseline.md`（已验证的版本元组）、`docs/design/2026-08-30-architecture-review.md`（当前架构评审与行动清单）。

## 基线（P8）

**NIGHTLY** = 三个 main 对齐：torch nightly（日期 = torch_npu master `requirements_<line>.txt` 的 pin）+ torch_npu master **源码构建**（`constraints/torch_npu.sha`）+ torchtitan main（`constraints/torchtitan.sha`）。这是唯一门禁 track；`constraints/nightly.txt` 是默认约束。
只在正式版 torch 上出现、nightly 上不存在的问题**不是问题**：不写 shim / 补丁 / issue。`constraints/npu.txt`（RELEASE：torch_npu 最新发布版）只做信息性报告；`npu-stable.txt` 已废弃。

## 布局

| 路径 | 层 | 内容 |
|---|---|---|
| `ascend_titan/_bootstrap.py`、`train.py` | 入口 | `setup()` = 唯一的副作用点；必须先于任何 `import torchtitan` |
| `ascend_titan/compat/` | L0 | shim 注册表 + `shims/`（受治理的 monkeypatch）。NIGHTLY 上两条现存 shim 都自动 no-op；目标数量 0 |
| `ascend_titan/kernels/` | L1 | 封装昇腾算子的 `@override` 工厂（attention = custom_op、rope、rms_norm、swiglu、situ_glu） |
| `ascend_titan/parallel/`、`graph/` | L2 | 并行策略、torchair（M4/M5，目前空） |
| `ascend_titan/recipes/` | L3 | `Trainer.Config` = 上游 registry 函数 + 增量；`transforms.npu_baseline`（每条增量挂 issue + 特性探测消失条件，P12）；`matrix.py` 动态 recipe |
| `ascend_titan/tools/` | L4 | `doctor`、`matrix`（扫描 + 归因）、`provenance` |
| `constraints/` | — | `nightly.txt` + `torchtitan.sha` + `torch_npu.sha` = 版本三元组（**唯一事实来源**，P11） |
| `scripts/build_torch_npu.sh` | — | 源码构建 torch_npu（本地盘 `/opt/build/torch_npu`，wheel → `/opt/wheels/` + 元数据 json）；`WITH_PATCHES=1` 叠加在途补丁 |
| `patches/torch_npu/`、`patches/op-plugin/` | — | **在途的 torch_npu / op-plugin 修复**（P9）：每个 = `fix/<ID>` 分支的 format-patch，提 PR 后头部加链接，合入即删 |
| `patches/evidence/` | — | torchtitan / pytorch 的修复方案，只读证据，永不应用（P10） |
| `docs/issues/STATUS.md` | — | 问题状态的唯一事实来源；`docs/issues/<owner>.md` 是可直接贴去提 issue 的正文 |
| `docs/capability-matrix.md` | — | 每个特性 🟢/🔴/⚪，带归因 |
| `tests/assets/losses/npu/` | — | 冻结的 golden loss 曲线（`scripts/check_golden.sh`） |

同级检出：`../torchtitan`（固定上游，**绝不改**）、`../ascend-pytorch`（torch_npu master，只在 `fix/<ID>` 分支上改）、`../ascend-op-plugin`（同上）。

## 命令

```bash
# NIGHTLY 环境（开发容器内；一次性）
python3.12 -m venv /opt/venv-nightly && . /opt/venv-nightly/bin/activate
pip install --pre -c constraints/nightly.txt torch --index-url https://download.pytorch.org/whl/nightly/cpu
source /usr/local/Ascend/cann-9.1.0/set_env.sh && ./scripts/build_torch_npu.sh      # 约 9 分钟；产物 /opt/wheels/torch_npu-2.15.0+git<sha>-*.whl
TORCH_NPU_WHEEL=/opt/wheels/torch_npu-*.whl WITH_TORCH=1 ./scripts/install.sh       # torch + torch_npu + 固定 SHA 的 torchtitan + 本包
ascend-titan-doctor                       # 环境探测（CPU 上也能跑）
pytest tests/unit -x                      # CPU 单测
ruff check . && ruff format --check .
ASCEND_TITAN_SKIP_SHIMS=1 ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 ./scripts/check_golden.sh qwen3_debugmodel_npu   # NIGHTLY 上 shim 必须全关也能过
MODULE=ascend_titan.recipes.qwen3 CONFIG=qwen3_debugmodel_npu NPU=8 ./scripts/run_train.sh
python -m ascend_titan.tools.matrix --cards 0-7 --jobs 4   # 上游用例搬到 NPU 扫描（docs/matrix/）
ascend-titan-provenance --module ascend_titan.recipes.qwen3 --config qwen3_debugmodel_npu
# tyro：`activation-checkpoint:none` 之类的子命令必须放在所有 --flag 之后
```

## 硬规则

1. **`import ascend_titan` 无副作用**（`tests/unit/test_import_purity.py`）。
2. **torch_npu 的问题只能修，不能绕**（P1 / P9）。归因 `NPU` ⇒ 最小复现 → `../ascend-pytorch`（或 `../ascend-op-plugin`）`fix/<ID>` 分支修复 + UT → `WITH_PATCHES=1 REQUIRE_PR_LINK=0 ./scripts/build_torch_npu.sh` → NPU 验证 → `patches/` 存 format-patch → `gitcode-pr-rfc-pipeline` 提 issue + PR → `STATUS.md` 记 URL。**在本仓任何位置（recipe、baseline、shim、换 loss）绕过 torch_npu 缺陷都是违规。**
3. **上游边界**（P10）：只能操作 `gitcode.com/ascend/*`；`github.com/pytorch/*` 只读（不提 issue / PR / 评论）。
4. **先在 NIGHTLY 复现再处理**（P8）：TT/TORCH 类失败先在 NIGHTLY 上确认存在；不存在 ⇒ 关闭；存在 ⇒ `docs/issues/` + `patches/evidence/`。
5. **先配置再 shim**（P0）；**shim 用包装并挂上游链接**（P3/P4）；**override 只针对已有的 `Configurable` 节点**（P6）；**recipe 是增量**。
6. **baseline 最小化**（P12）：`npu_baseline` 只允许"不加就跑不起来"的增量，每条挂 issue ID，用特性探测（不是版本号）决定何时消失。
7. **版本三元组一起升级，走带矩阵结果的 PR**（P5）。绝不随手改 `constraints/*.sha`。
8. **单一事实来源**（P11）：版本只在 `constraints/`，问题状态只在 `STATUS.md`，其它文档只引用 ID。
9. **不加投机性的兜底。** 与上游同一标准：只校验显式契约。
10. **pip 安装永远带 `-c constraints/<track>.txt`**——否则 torch 被升级、torch_npu ABI 损坏。
11. **验证先于断言**（P13）：任何 🟢 / "已修复" 附命令与输出，且在 NIGHTLY 上跑过。
12. **改 `matrix.py` 的 `TRIAGE` 表或任何被 ruff 重排过的多行元组时，用行定位插入，不要用整行字符串 replace**。

## 失败处理（经常用）

```
失败 → 归因（traceback 首个非框架帧）
  torch_npu/… 或 op_plugin … aclnnXxx failed   → NPU / NPU-OP → 硬规则 2 全流程，禁止绕过
  torch/ 自身 或 torchtitan/…（nightly-only API、cuda 字样） → TORCH / TT → 硬规则 4
  attn_gym / helion / deep_ep / cutlass / torchao / fla        → DEP → 记录；昇腾替代是 L1 任务
  CANN 错误码（EZ9999、EI0002…；ERR99999 只是包装）           → CANN → 记录错误码，停
  EI0020 端口已绑定                                            → HARNESS → 同卡有其它 HCCL 作业，重跑
```

## 数值验证
与上游同一标准：非计算类改动在 `--debug.seed=42 --debug.deterministic` 下 loss 必须**完全一致**（NIGHTLY 与 2.13 golden 逐位一致，已验证）；计算类改动（算子）需要对上游 eager 路径的对齐测试加 `torch.library.opcheck`。绝不使用 `--debug.deterministic_warn_only`。

## 开发容器
NPU 主机上的 `ascend-titan-dev`（`ssh root@localhost` → `docker exec`；镜像 cann:9.1.0-910b-ubuntu22.04-py3.12-devel）：`/opt/venv-nightly` = **NIGHTLY**（torch 2.15.0.dev20260812 + torch_npu master 源码构建）；`/opt/venv213` = RELEASE（2.13.0 / 2.13.0rc1）；系统 python = 2.12（废弃）。`/data` 是共享 NFS，仓库在容器内路径相同；**构建一律在本地盘 `/opt/build/`**（NFS 上 flock/构建不可靠）。8 张卡通常与其它作业共享——用 `ASCEND_RT_VISIBLE_DEVICES` 选卡；同一张卡上不能并发两个 HCCL 作业（EI0020）。

## Skill
`.claude/skills/`：`compat-probe`、`shim-authoring`、`override-authoring`、`capability-matrix`、`upstream-sync`。`.claude/rules/` 里的规则按路径生效。提 torch_npu issue / PR 用全局 skill `gitcode-pr-rfc-pipeline`（需要 `GITCODE_TOKEN`，账号 `jimmyisme1`，fork `jimmyisme1/pytorch`）。
