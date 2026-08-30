# ascend-torchtitan —— 架构评审（2026-08-30）

状态：评审稿 v1（含 NPU 实测，见 §6；补丁提交状态见 `docs/issues/STATUS.md`）。评审对象：仓库 `main` @ `8029863`，torchtitan 固定点 `13da2d77c`，torch_npu master @ `15514cc70`（2026-08-29）。
读者：维护者与 Claude Code。配套：`docs/PRINCIPLES.md`（本评审新增 P8–P13）、`docs/adr/ADR-006-nightly-first.md`。

## 0. 结论（TL;DR）

1. **骨架是对的，保留。** 树外扩展不 fork（ADR-001）、override 优先于 shim（P0/P6）、shim 受治理（ADR-002）、按 SHA 固定（ADR-003）、响亮降级（ADR-004）、单 vendor（ADR-005）——这六条决定在实测后仍然成立，本评审不推翻任何一条。
2. **版本基线选错了，这是仓库现在最大的结构性问题。** 我们用 torch **正式版**（2.12 / 2.13）去跑面向 torch **nightly** 的 torchtitan main，于是造出了一整层"接口补丁"：2 条 shim + 5 个 torchtitan 补丁 + 2 个 pytorch 补丁里，**7 项（TT-2 / TT-8 / TT-9 / TORCH-3 / 4 / 5 / 6）纯粹是 torch 版本差**，14 个 CP 红格（TT-5）也是。设计文档 §4 的前提"torch_npu 面向 torch 正式版"**是错的**：torch_npu master 的 `requirements.txt` 钉 `torch==2.14.0.dev20260719+cpu`、`requirements_2.15.txt` 钉 `torch==2.15.0.dev20260812+cpu`、`version.txt` 列出 `2.15.0`，`ci/build.sh --torch=2.15.0` 就是官方构建路径。**正确的基线是三个 main 对齐：torch nightly + torch_npu master（源码构建）+ torchtitan main SHA。**
3. **实测（CPU 侧，torch 2.15.0.dev20260812）**：`set_timeout`、`PipelineSchedule.step(arg_mbs=)`、`torch.Tag.inplace`、FSDP2×`spmd_types` 全部原生存在；torchtitan `13da2d77c` 可 import；本仓 45 个单测全过。也就是说切到 nightly 之后 **两条 shim 自动 no-op、三个 torchtitan 补丁可以直接删除**。NPU 侧结果见 §6。
4. **P1（绝不绕过 torch_npu）在本仓自己的 `npu_baseline` 里被违反了**：TT-4（ChunkedLossWrapper backward "data is not allocated yet"）在 `STATUS.md` 里已"倾向归因 NPU"，却被 `recipes/transforms.py:108` 用"展开成内层 loss"绕过；性能型 override（`npu_rms_norm`）也混进了 baseline，使矩阵红格可能由我们自己的内核引起。
5. **同一事实在 ≥4 处重复维护，已经漂移**（"torch_npu 面向正式版"、`gitee` vs `gitcode`）。需要一个单一事实来源。

## 1. 评审范围与方法

- 通读全部源码（`ascend_titan/` 2223 行、`tests/` 915 行）、`patches/`、`constraints/`、`docs/`、`.claude/`、CI 工作流；对照 torchtitan `13da2d77c` 与 torch_npu master（`../ascend-pytorch`）。
- 在开发容器 `ascend-titan-dev` 上新建 `/opt/venv-nightly`：torch `2.15.0.dev20260812+cpu`（torch_npu master 钉的日期）+ 从源码构建 torch_npu master；用它跑本仓（§6）。
- 评审按"文件结构 → 仓库设计 → 软件设计 → 原则"四层给意见；每条意见附证据（file:line）与建议动作，动作汇总在 §8。

## 2. 仓库设计：版本策略

### 2.1 现状

| 项 | 现状 | 证据 |
|---|---|---|
| 默认 track（NEXT） | torch 2.13.0 + torch_npu 2.13.0rc1 | `constraints/npu.txt` |
| 第二 track（STABLE） | torch 2.12.0 + torch_npu 2.12.0 | `constraints/npu-stable.txt` |
| torchtitan | main `13da2d77c`（2026-08-29） | `constraints/torchtitan.sha` |
| 不用 nightly 的理由 | "torch_npu 跟随 torch 正式版" | `docs/baseline.md:15`、`docs/design/…:78`、`docs/upstream-tracking.md:9`、`ADR-003:7` |

### 2.2 事实（2026-08-30 实测）

- torchtitan main 面向 nightly：README "we recommend using the most recent PyTorch nightly"；`unit_test_cpu.yaml:34` 安装 `--pre torch … /whl/nightly/cpu`。
- **torch_npu master 也面向 nightly**：`requirements.txt:7 torch==2.14.0.dev20260719+cpu`（index = nightly/cpu）、`requirements_2.15.txt torch==2.15.0.dev20260812+cpu`、`version.txt` = `2.15.0 / 2.14.0 / 2.13.0`；`setup.py::_get_torch_requires` 生成 `torch==2.15.*` 前缀匹配，注释明说"letting pip honor an already-installed nightly"。PyPI 上只发到 2.13.0rc1 是**发布节奏**问题，不是兼容性问题。
- 构建条件在容器里全部满足：256 核 / 2 TB 内存、gcc 11.4（C++20）、cmake 3.22、ninja、python 3.12；gitcode 与 nightly index 均可达；`ci/build.sh --python=3.12 --torch=2.15.0 --disable_torchair` 直接可用（§6 给出耗时）。
- 在 torch 2.15.0.dev20260812 上（CPU 探测脚本 `outputs/nightly/probe_torch_apis.py`）：

| 缺口 | 2.13 | nightly | 含义 |
|---|---|---|---|
| `torch.distributed.set_timeout`（TT-2 / TORCH-3） | 缺 | **有** | shim `dist_set_timeout` 自动 no-op；补丁 TT-2 删除 |
| `PipelineSchedule.step(arg_mbs=)`（TT-8 / TORCH-4） | 缺 | **有** | shim `pp_step_presplit_*` 自动 no-op；补丁 TT-8 删除 |
| `torch.Tag.inplace`（TT-9） | 缺 | **有** | 补丁 TT-9 删除 |
| FSDP2 读取 `spmd_types` 注解（TT-5 / TORCH-6） | 缺 | **有** | 14 个 CP 红格有机会翻绿；`npu_baseline` 第 3 步应删 |
| `fork_rng` 默认 cuda（TORCH-5） | 2.12 缺 | 有 | STABLE 独有的 6 个 PP 红格消失 |
| Flex 设备白名单（TORCH-1） | 有 | 仍有 | torch_npu master 自带 `utils/patch_flexattention.py`，需 NPU 实测 |
| `opcheck` autograd 只认 CPU/CUDA/XPU（TORCH-7） | 有 | 仍有 | 仍需本地补丁（测试工具，不进产品路径） |
| `varlen.py` uint64 rng_state（TORCH-8） | 有 | 仍有（`varlen.py:158`、`:200`） | 正解在 op-plugin 支持 uint64（NPU-6），不是改 torch |
| fake 进程组设备表（TORCH-2 / NPU-2） | 无 npu | 仍无 | torch_npu 侧注册（补丁 0001） |

### 2.3 问题

- **shim 层的存在理由几乎全部来自版本差。** 设计把 shim 定义为"上游写死 CUDA 且没有开关"的最后手段（ADR-002），但现有两条 shim 和 `patches/torchtitan/` 里 3/5 的补丁，都是"torchtitan 用了 nightly API 而我们不在 nightly 上"。这不是昇腾问题，是自找的问题；它们还制造了维护面（`pp_step_presplit.py` 130 行，复刻了 torch `step()` 的前导逻辑，随 torch 版本漂移）。
- **两条 track 的价值极低。** NEXT 与 STABLE 的全部差异是 TORCH-5（6 个 PP 用例）；golden 逐位一致。维持两条 track 的代价是矩阵扫描翻倍、golden 文件翻倍、constraints 文件翻倍。
- **CI 与基线不一致。** `.github/workflows/ci.yml:13` 装的是 torch **正式版** CPU wheel，而 NPU 用的是 2.13——CPU 门禁、NPU 门禁、上游 nightly 三者互不相同。
- **`constraints/npu-triton.txt` 是 `npu.txt` 的整份拷贝**再加两行，注定漂移。

### 2.4 建议：nightly-first（ADR-006）

1. **track 重定义**：
   - `NIGHTLY`（默认、门禁）：torch nightly **日期取自 torch_npu master 的 `requirements_2.15.txt`**（唯一来源，脚本读取，不手抄）+ torch_npu master 源码构建（SHA 固定于 `constraints/torch_npu.sha`，含 op-plugin 子模块 SHA）+ torchtitan main SHA。
   - `RELEASE`（次级、只报告不门禁）：torch_npu 最新发布版 + 其配套 torch。用于回答"用户拿发布版能跑什么"。
   - **删除 STABLE**。
2. **`constraints/` 重构**：`nightly.txt`、`release.txt`、`torchtitan.sha`、`torch_npu.sha`、`titan-deps.txt`；Triton-Ascend 等可选层写成 `extras/triton.txt`（只含增量，安装时叠加 `-c`），不再整份复制。
3. **三者一起升级、一 PR 附矩阵**：P5 的对象从"torchtitan SHA"扩为"三元组"。升级脚本 `scripts/bump_baseline.sh` 输出候选三元组 + `probe_compat.sh` 结果。
4. **构建脚本 `scripts/build_torch_npu.sh`**：rsync `../ascend-pytorch` → `/opt/build/torch_npu`（NFS 上不能构建，同 ops-nn），`git submodule update --init --recursive --depth 1`，`ci/build.sh --python=3.12 --torch=2.15.0 --disable_torchair`，wheel 落到 `/opt/wheels/`，记录 `sha256 + 源码 SHA + op-plugin SHA` 到 `outputs/nightly/torch_npu.build.json`；可选 `WITH_PATCHES=1` 叠加 `patches/torch_npu/*.patch`（每个补丁必须带 gitcode PR 链接，否则拒绝，见 P9）。
5. **CI**：`ci.yml` 改装同日期 nightly（`pip install --pre torch==<date> --index-url …/nightly/cpu`）；`npu-nightly.yml` 只跑 NIGHTLY，`probe` 腿保留。

## 3. 文件结构与仓库布局

| 路径 | 现状 | 问题 | 建议 |
|---|---|---|---|
| `ascend_titan/compat/` | 注册表 172 行 + 2 条 shim（163 行） | 两条 shim 在 nightly 上都 no-op；`pp_step_presplit.py` 复刻 torch 内部逻辑，属"替换型"却标 `wrap` | 注册表保留（便宜且是治理机制）；两条 shim 文件待 RELEASE 退役后删除；**已加**门禁测试 `tests/unit/test_nightly_gate.py`：NIGHTLY 上 shim 必须是 no-op |
| `ascend_titan/kernels/` | 5 个 override 模块 | 5 份相同的 `try: import torch_npu … _AVAILABLE` 样板；`attention.py:255-257` 无条件转 bf16（fp32 输入静默降精度）；`rope.py` 未做可用性探测（与 `.claude/rules/kernels.md` 第 1 条不一致） | **已落实（ADR-007/P14）**：抽 `kernels/_probe.py::require_op(name)`，缺模块 / 缺算子一律抛错，不再有 `_AVAILABLE` 降级开关；bf16 转换改为"仅当输入是 fp16/bf16 以外时 raise 或显式 WARNING"；`rope.py` 用同一探测器 |
| `ascend_titan/parallel/`、`graph/` | 空包 + README | 空包会进 wheel；按 YAGNI 本不该建 | 可接受（README 说明了用途）；但 M4/M5 前不要再加空目录 |
| `ascend_titan/recipes/transforms.py` | `npu_baseline` 6 步 | 见 §4.4：混入了 P1 违规项与性能项 | **已落实（2026-08-30）**：拆成 `npu_minimal`（矩阵默认，只含"不加就跑不起来"）与 `npu_fused`（性能，opt-in，`--mode fused`）；每条增量挂 issue ID 与**消失条件** |
| `ascend_titan/tools/matrix.py` | 440 行，`TRIAGE` 正则表 107 行 + 运行器 + 报告 + CLI | 数据（归因规则）写在代码里，ruff 重排导致 `CLAUDE.md` 硬规则 10 那种"用行定位插入"的怪规矩；一个文件五种职责 | `TRIAGE` 迁到 `docs/issues/issues.toml`（见下），`tools/matrix/` 拆成 `triage.py`、`runner.py`、`report.py`、`cli.py` |
| `patches/` | torchtitan 5 / pytorch 2 / torch_npu 3 | 三种归属混在一个目录但**政策完全不同**：torch_npu 的要提 PR，torchtitan/pytorch 的按用户决定不提；torchtitan/pytorch 补丁没有任何产品路径会应用它，只是"证据" | `patches/torch_npu/`：**在途补丁**，每个必须带 gitcode issue + PR URL，合入即删；`patches/evidence/{torchtitan,pytorch}/`：只读证据，README 明说"永不应用于安装路径"；切 nightly 后 TT-2/8/9 三个补丁删除 |
| `constraints/` | 4 个 txt + sha | §2.3 | §2.4 |
| `docs/issues/` + `STATUS.md` + `upstream-tracking.md` + `capability-matrix.md` + `baseline.md` | 同一 issue 的状态在 ≥4 处 | 已漂移：`docs/issues/torch_npu.md:1` 写 `gitee.com`，`STATUS.md` 写 `gitcode.com`；`upstream-tracking.md:9` 与 `baseline.md:15` 的"torch_npu 面向正式版"是错误前提 | 单一事实来源 `docs/issues/issues.toml`（id / owner / title / status / links / triage_regex / cells），`STATUS.md` 表与 `matrix.py` 的 `TRIAGE` 由它生成（`python -m ascend_titan.tools.issues render`）；其它文档只引用 ID，不写状态 |
| `outputs/` | gitignore，但含 10 个 `.py`（`apply_tt1.py`、`tt4_isolate*.py`、`npu3_getitem_prototype.py`…） | 工具与复现脚本放在忽略目录里 = 知识丢失；这些正是提 issue 时要附的最小复现 | 复现脚本移到 `tests/repro/<ID>_*.py`（不进 pytest 收集，或标 `@pytest.mark.repro`）；一次性 apply 脚本删除 |
| `CLAUDE.md` / `AGENTS.md` | `AGENTS.md` 是 `CLAUDE.md` 的符号链接 | 无（单一来源已做对） | 保持；评审初稿误判为两份拷贝，已更正 |
| `.claude/settings.json` | deny 编辑 `../torchtitan/**` | 正确；但没有约束 `../ascend-pytorch`（那里**应该**改，只是必须在分支上） | 加规则文本：`../ascend-pytorch` 只能在 `fix/<ID>` 分支上改，master 保持干净 |
| 工作区（`../` 下 7 个仓库） | 只有 torchtitan 有 SHA 锁 | `ascend-pytorch`、`ops-nn`、`ops-transformer`、`flash-linear-attention-npu`、`triton-ascend-kernels` 的期望 SHA 无处记录，环境不可复现 | `constraints/workspace.lock`（repo / url / sha / 用途）+ `scripts/workspace.sh {sync,status}` |
| `tests/` | unit 45 / npu 5 文件 | 缺"nightly 上 shim 数为 0"的门禁；golden 只有 2.12/2.13 元组 | 加门禁测试；golden 增加 NIGHTLY 元组；删除 STABLE golden |
| `.github/workflows/npu-nightly.yml` | 引用 `self-hosted, npu` runner | 目前不存在该 runner，工作流从未跑过 | 要么接 runner，要么改成 `scripts/nightly.sh` 由 cron 在开发机执行并把报告提交到 `docs/matrix/` |

## 4. 软件设计

### 4.1 引导（`_bootstrap.py`、`train.py`）
- 设计正确：唯一副作用点、幂等、`SetupReport` 可审计、`ASCEND_TITAN_SKIP_SHIMS` 便于验证上游修复。
- 问题：`train.py` 调 `setup()` 时 `require_npu=False`（默认），torch_npu 导入失败只 WARNING，训练会在别处以更难懂的错误死掉。**已落实（ADR-007/P14）**：`setup()` 直接硬导入 torch_npu 并去掉 `require_npu` 参数——ADR-004 的"响亮降级"只针对**可选加速包**，不针对**基础依赖**。

### 4.2 shim 注册表（`compat/registry.py`）
- 设计正确：`wrap / replace / polyfill` 三型、import 时强制 `reason` 与 `upstream`、polyfill 自动 no-op。
- 问题 1：`upstream=` 允许 `draft:` 永久存在。在"不向 torchtitan/pytorch 提 issue"的政策下，TT/TORCH 类 shim 的 `draft:` 永远不会变成 URL，P4 的"到期日"机制失效。
- 问题 2：没有"复现基线"字段。建议 `Shim` 增加 `reproduces_on: str`（必须是 NIGHTLY 元组），并加门禁：nightly 上任何 shim 若 `apply_all()` 实际生效，测试失败并要求给出 nightly 上的复现。
- 问题 3：`pp_step_presplit` 标 `kind="wrap"` 但实际复刻了 `step()` 的前导逻辑（`pp_step_presplit.py:64-77`），是替换型。切 nightly 后此问题随 shim 一起消失。

### 4.3 kernels（L1）
- `attention.py`：custom_op + `register_fake` + `register_autograd` + LSE 尾部——这是仓库里质量最高的模块。保留的疑虑：每步一次 D2H（OURS-1）；`sparse_mode=4` 滑窗未测（OURS-3）；`_lse_from_stats` 的 head-major 布局是"实测"出来的（注释写明），没有 torch_npu 文档背书，应在 torch_npu 侧补一个 `npu_fusion_attention` 返回 LSE 的诉求。
- `rope.py`：两个 override（实数缓存 ComplexRoPE、CosSinRoPE 走 `npu_rotary_mul`）。前者存在的唯一理由是 NPU-3（复数索引）；**NPU-3 在 torch_npu 侧修好后这个 override 应降级为"可选性能项"而不是 baseline 必需项**。
- `swiglu.py`：复用上游 `FusedSwiGLU` 的 w13 布局与 checkpoint hook，只换激活——这是 P6 的模范用法。
- `situ_glu.py`：ops-nn 单算子构建 + 可微 custom_op；受 TT-11 阻塞无法接模型。
- 共性问题：可用性探测样板重复 5 次；`_AVAILABLE` 为 False 时模块体大段缩进在 `if` 里，难读。**已落实（ADR-007/P14）**：改为 `_probe.require_op()` 硬探测 + 模块体顶层定义，`if _AVAILABLE:` 整块消失。

### 4.4 recipes / transforms（L3）—— **P1 违规点**
`npu_baseline` 当前 6 步：flex→varlen、attention override、RoPE override、**RMSNorm override**、spmd_backend→partial_dtensor、**ChunkedLossWrapper 展开**。
- 第 4 步（`transforms.py:108`）：TT-4 在 `STATUS.md` 里已归因"倾向 NPU：`libtorch_npu.so: add_param_to_buf` 在 backward 的 `mul` 上读到已释放存储"。按 P1 这必须去 torch_npu 修，不能在 baseline 里 unwrap。**它现在是全矩阵 61 个用例都在用的绕过。** 处置：在 NIGHTLY + torch_npu master 上复现（§6），归因确定后走 P9 流程。
- 第 2c 步：`npu_rms_norm` 是性能内核，数值与 `torch.rms_norm` 在 bf16 舍入级别不同。放进 baseline 意味着任何红格都要先排除"是不是我们的内核"，而且 baseline golden 与 fused golden 的边界被打破。处置：移到 `npu_fused`。
- 第 3 步：nightly 的 FSDP2 已读取 spmd_types 注解，这一步应删（§6 验证）。
- 目标状态：**`npu_baseline` = identity。** 每留一条增量都必须写明 issue ID 与消失条件（如 `until="torch_npu 含 PR #N"`），并由测试在条件满足时报错提醒删除。

### 4.5 tools（L4）
- `doctor`：好。建议增加：torch_npu 的源码 SHA（wheel 元数据 `git_version`）、是否含本仓补丁（在 `WITH_PATCHES=1` 构建时写入 wheel 的 `torch_npu/_ascend_titan_patches.txt`）。
- `provenance`：好；应接入 `matrix.py` 报告（路线图已列）。
- `matrix.py`：见 §3；另外 `TRIAGE` 里 `NPU-OP` 规则只匹配 `aclnn\w+ failed`，而 §6 会出现的新失败特征需要一并进入 issues.toml。

### 4.6 测试与数值
- 单测覆盖 shim 注册表、recipe 增量、import 纯净性——这些是"廉价漂移探测器"，设计正确。
- 缺：NIGHTLY golden；"shim 数为 0"门禁；`tests/repro/` 最小复现（提 issue 的附件）。
- 数值标准（非计算改动逐位一致、计算改动对齐 + opcheck）正确；`opcheck` 的 autograd 检查因 TORCH-7 在 NPU 上跳过，应在 `tests/npu` 里用数值梯度补齐（目前已这么做，记录在案即可）。

## 5. 问题清单在 nightly 基线下的重新归因

| ID | 原归因 / 处置 | nightly 上是否存在 | 新处置 |
|---|---|---|---|
| TT-2、TT-8、TT-9 | 本地补丁 + shim | **不存在** | 删除补丁与 shim；关闭 |
| TT-5 / TORCH-6 | 阻塞，等 torch ≥ 2.14 | **不存在**（`_fsdp_param.py` 已读 spmd_types） | 复测 14 个 CP 用例（§6） |
| TORCH-5 | STABLE 独有 | 不存在 | 随 STABLE 一起删除 |
| TORCH-3 / TORCH-4 | 无需处理 | 不存在 | 关闭 |
| TT-1（`import triton`） | 装纯 Python triton | 存在（依赖问题，与 torch 版本无关） | 维持 `titan-deps.txt` 的 `triton`；补丁归入 evidence |
| TT-11（kimi_k3 需 `cutlass`） | 本地补丁 | 存在（attn_gym 的 import 期依赖） | 归入 evidence；kimi_k3 在无 CUDA 环境不可导入是上游（attn_gym）问题，本仓不提上游 ⇒ 矩阵保持 🔴 DEP，诚实记录 |
| TT-4（ChunkedLossWrapper） | 倾向 NPU，被 baseline 绕过 | **待 §6 复现** | 若复现：走 P9（torch_npu 修复 + PR）；baseline 去掉 unwrap |
| TORCH-1（Flex 白名单） | pytorch draft | 存在于 torch；**torch_npu master 已带 `patch_flexattention.py`** | §6 实测 flex；若通过则 `attention/flex` 翻绿、`npu_baseline` 第 1 步删除 |
| TORCH-7（opcheck） | 本地补丁 | 存在 | 归入 evidence（测试工具） |
| TORCH-8（varlen uint64） | 本地补丁 | 存在 | **改为 op-plugin 侧修 NPU-6**（`zero_`/`zeros` 支持 uint64，gitcode.com/ascend/op-plugin 在授权范围内）；torch 补丁归入 evidence |
| NPU-1（`_flash_attention_forward`） | torch_npu 补丁 0003 | 存在（master 未实现） | P9：在 master 上重验 → gitcode issue + PR |
| NPU-2（fake 进程组） | torch_npu 补丁 0001 | 存在 | P9 |
| NPU-3（复数索引） | torch_npu 补丁 0002（Python 层 `__getitem__` 路由） | 存在 | P9；注意 0002 是 torch_npu **内部**的 Python 层绕行，正解在 op-plugin `aclnnIndex` 支持 complex——PR 描述里必须写明是过渡实现并同时开 op-plugin issue |
| NPU-6（uint64 `zero_`） | 已确认 | 存在 | 同 TORCH-8 行 |
| OURS-11（Triton-Ascend） | 进行中 | 与 torch 版本无关 | `extras/triton.txt`；torch_npu master 已改为先试 `tl.extra.cann`（`_inductor/runtime/triton_helpers.py:13`），NIGHTLY 上 inductor 用例可复测 |

## 6. NPU 实测：torch nightly + torch_npu master

> 环境：`ascend-titan-dev`，`/opt/venv-nightly`，torch `2.15.0.dev20260812+cpu`（git `3eb0e5d0`），torch_npu master `15514cc70`（op-plugin `2b02a5aa0`），`ci/build.sh --python=3.12 --torch=2.15.0 --disable_torchair`：**8 分 28 秒，零报错**（gcc 11.4，256 核）。CANN 9.1.0，910B2。全部运行 `ASCEND_TITAN_SKIP_SHIMS=1`。原始日志：`outputs/nightly/`（脚本已固化到 `tests/repro/`、`scripts/build_torch_npu.sh`、`ascend_titan/models/llama3/recipes.py`）。

### 6.1 原版 torch_npu master（未打任何补丁）

| 项 | 结果 | 归因 |
|---|---|---|
| `import torch` 自动加载 torch_npu，`doctor`：8 卡可见 | 🟢 | |
| `qwen3_debugmodel_npu` 单卡 / FSDP2×2 golden（shim 全关） | 🟢 **与 2.13 golden 逐位一致**（5.10304 / 3.3061；5.07792 / 3.3201） | 两条 shim、TT-2/8/9 补丁确认为版本差 |
| ChunkedLossWrapper（TT-4）单卡 / FSDP2×2 | 🟢 5.10291 / 5.07796（bf16 级差异） | TT-4 在 NIGHTLY 不存在；baseline 的 unwrap 是 P1 违规，已删 |
| `flex_attention` eager op 级前反向 | 🟢 | torch_npu master `patch_flexattention.py` 绕开 TORCH-1 |
| stock varlen（`aten::_flash_attention_forward`） | 🔴 无 PrivateUse1 内核 | NPU-1 |
| `--comm.mode=fake_backend` | 🔴 `No backend type associated with device type npu` | NPU-2 |
| 复数张量高级索引 | 🔴 aclnnIndex 161002 | NPU-3 |
| `torch.zeros(dtype=uint64)` | 🔴 aclnnInplaceZero 161002 | NPU-6 |
| stock flex（模型级，`torch.compile(flex_attention)`） | 🔴 `make_reduction() got an unexpected keyword argument 'strict_reduction'` | **NPU-7**（新）：torch_npu inductor 覆盖层对 torch 2.15 的签名漂移 |
| 先 `import spmd_types` 再 `import torch`（矩阵工具、torchtitan `trainer.py` 顺序） | 🔴 `Failed to load the backend extension: torch_npu` ← spmd_types 循环导入 | **NPU-8**（新）：torch_npu 自动加载时经 `torch.distributed._tensor` / `sharded_grad_scaler` 拖入 fsdp |

### 6.2 torch_npu / op-plugin 侧修复后（`patches/`，六项，均带 UT）

| 项 | 结果 |
|---|---|
| `tests/repro/probe_npu_gaps.py` | NPU-1 / 2 / 3 / 6 全部 `[OK ]`；`_flash_attention_forward` PrivateUse1 = True |
| **stock varlen**（qwen3，零 override） | 🟢 10 步 loss 5.10302 / grad_norm 3.3060（与融合 override golden 5.10304 / 3.3061 仅 bf16 级差异） |
| **stock llama3**（`ascend_titan.models.llama3`：stock VarlenAttention + stock ComplexRoPE 复数索引 + ChunkedLossWrapper + spmd_types，零 override） | 🟢 单卡 4.01820 / 1.7382；FSDP2×2 3.97774 / 1.7523 |
| `import torch` 足迹 | fsdp / checkpoint 不再被自动加载拖入；`import spmd_types` 先行 OK；`ShardedGradScaler` 仍被替换 |
| 矩阵工具（先导入 torchtitan） | 可运行：`pp_1f1b` 🟢（无 shim）；`cp` / `fsdp+cp` / `fused_mla` 🔴 **DEP-INDUCTOR**（需要 Triton-Ascend，与 NPU 无关） |
| stock flex 模型级 | lowering 通过（NPU-7 生效），停在 `0 active drivers`——Triton-Ascend 依赖（DEP-INDUCTOR） |
| UT（最终 wheel） | fake PG 1 OK；flash-attention 4 OK；autoload 4 OK；op-plugin index-complex 6 OK；zero-unsigned 3 OK；inductor 签名 1 passed |
| `fake_backend` | 🟢（最终 wheel：单卡模拟 8 卡 `step: 1  loss: 7.66238`）。过程：NPU-2 第一版（改 `_init/registry/distributed.py`）在 master 上是**死代码**；第二版（init 时追加）被 nightly 的**懒注册**（entry point 首次 `BackendConfig("fake")` 覆盖 `backend_capability`）冲掉；第三版包装 `Backend.register_backend`（torch_npu.testing 本来就这么做，只是没放到运行时） |

### 6.3 结论
- nightly-first 成立：切换后本仓 **0 条生效 shim、0 个 torchtitan 版本差补丁**，golden 逐位不变。
- 昇腾侧真正的缺口是 6 项，全部可在 torch_npu / op-plugin 修复且已有补丁与 UT；其中 NPU-7 / NPU-8 是 **torch_npu 未跟上 nightly** 的直接证据——正是用户担心的那类问题，只能靠 nightly 基线才能暴露。
- 剩余红格归 Triton-Ascend（DEP-INDUCTOR）与 cutlass（TT-11），都不是 NPU/CANN 问题。

## 7. 给 Claude Code 的原则（新增 P8–P13，写入 `docs/PRINCIPLES.md`）

| # | 原则 | 理由 |
|---|---|---|
| **P8** | **nightly-first。** 开发、验证、门禁的基线 = torch nightly（日期取自 torch_npu master 的 requirements）+ torch_npu master 源码构建 + torchtitan main SHA。任何只在正式版 torch 上出现、nightly 上不存在的问题，**不是问题**：不写 shim、不写补丁、不记 issue，只在矩阵里标注"RELEASE 不支持"。 | torchtitan 与 torch_npu 的 main 都面向 nightly；追正式版等于自己制造一层接口补丁。 |
| **P9** | **torch_npu 的问题只能修，不能绕；修好才算数。** 归因 NPU 的失败必须走完整流程：最小复现（`tests/repro/`）→ 在 `../ascend-pytorch`（或 op-plugin）的 `fix/<ID>` 分支修 → `scripts/build_torch_npu.sh` 重建 → NPU 验证（对齐测试 / opcheck / golden）→ `patches/torch_npu/` 存 `format-patch` → 用 `gitcode-pr-rfc-pipeline` 在 gitcode.com/Ascend 提 issue + PR → `STATUS.md` 记 URL → 合入后删补丁、升 `torch_npu.sha`。**在 torch_npu 之外的任何位置（本仓、recipe、baseline、shim）绕过 torch_npu 缺陷都是违规**，包括"展开 loss wrapper"这类看似无害的配置改动。 | P1 的执行细则；没有流程的红线会被"临时绕一下"侵蚀（TT-4 就是例子）。 |
| **P10** | **上游边界。** 允许操作的远端只有 `gitcode.com/ascend/*`（pytorch、op-plugin、torchair…）。`github.com/pytorch/*` 只读：不提 issue、不提 PR、不评论。TT/TORCH 归因的问题先按 P8 确认在 nightly 上存在；存在则记录到 `docs/issues/` 并把修复方案存为 `patches/evidence/`，**永不应用于安装路径**。 | 用户授权范围。 |
| **P11** | **单一事实来源。** 版本三元组只在 `constraints/`；问题状态只在 `docs/issues/issues.toml`（生成 `STATUS.md` 与 `TRIAGE`）；其它文档只引用 ID。发现两处不一致时，先修事实来源再改引用。 | 本评审在 5 个文档里发现同一前提的 3 种说法。 |
| **P12** | **baseline 最小化。** `npu_baseline` 只允许"不加就跑不起来"的增量，每条挂 issue ID 与消失条件；性能 override 放 `npu_fused`。目标：baseline = identity。 | 否则矩阵红格无法区分"上游问题"与"我们的内核问题"。 |
| **P13** | **构建可复现，验证先于断言。** 源码构建的组件（torch_npu、ops-nn、fla-npu…）都有 `scripts/build_*.sh` + SHA 锁 + 产物记录；不在 NFS 上构建。任何 🟢 / "已修复" 必须附命令与输出（golden、opcheck、测试），在 NIGHTLY 上跑过才算数。 | 先前"已验证"的 STABLE/NEXT 数据在 nightly 上有一半不再适用。 |

配套的 Claude Code 工作流（写入 `CLAUDE.md` "失败处理"节）：

```
失败 → 归因（traceback 首个非框架帧）
  NPU / NPU-OP / CANN-in-torch_npu → P9 全流程（禁止任何绕过）
  TORCH / TT → 在 NIGHTLY 上复现？否 → 关闭（版本差）。是 → docs/issues + patches/evidence（不提上游，不进安装路径）
  DEP → 记录；昇腾替代 = L1 任务
  CANN（驱动/固件/算子库错误码）→ 记录错误码，停
  HARNESS → 重跑
```

## 8. 行动清单（按优先级）

| 优先级 | 动作 | 交付物 |
|---|---|---|
| ~~P0~~ 已完成 | NIGHTLY track 落地：`constraints/nightly.txt`、`torch_npu.sha`、`scripts/build_torch_npu.sh`、`install.sh` 支持本地 wheel、NIGHTLY golden、`tests/unit/test_nightly_gate.py` | 本仓 |
| P0（部分完成） | 版本差补丁 TT-2/8/9 已删；`npu_baseline` 第 3 步改特性探测、第 4 步删除；**待做**：删除 `npu-stable.txt`、STABLE golden 与两条 shim 文件（RELEASE 退役时） | 单测 + golden |
| ~~P0~~ 已完成 | TT-4 在 NIGHTLY 不复现（单卡 + FSDP2×2 通过）；baseline 的展开已删除 | `docs/issues/STATUS.md` |
| ~~P1~~ 已完成 | NPU-1/2/3/6/7/8 在 master 上重验并修复 → gitcode issue + PR（Ascend/pytorch !45526–!45529、Ascend/op-plugin !5800–!5801；CLA ✅，CI 运行中） | `STATUS.md` |
| P1 | `docs/issues/issues.toml` + 生成器；`TRIAGE` 迁出 `matrix.py` | `STATUS.md` 由生成 |
| ~~P1~~ 已完成 | `PRINCIPLES.md` P8–P13、`ADR-006`、`CLAUDE.md` 工作流、skill `torch-npu-fix` | 文档 |
| ~~P2~~ 已完成 | `kernels/_probe.py`、`train.py` 硬依赖（P14 / ADR-007：硬导入 + `require_op`，而不是原方案的"只 WARNING 一次"）；`npu_baseline` → `npu_minimal` + `npu_fused`（矩阵 `--mode minimal\|stock\|fused`） | 代码 + 测试 |
| ~~P1~~ 已完成（2026-08-30 追加） | L3 按模型重组：`ascend_titan/models/<model>/`（recipes + probes + 必需 README）、`models/registry.py`、`models/_template/`；`recipes/` 只留跨模型机制。`tests/unit/test_models_registry.py` 强制登记与文档 | 代码 + 测试 |
| ~~P2~~ 已完成（2026-08-30 追加） | README 重写：banner / 架构图 / golden 曲线（`docs/assets/`）、模型与特性支持表、上游修复表 | 文档 |
| P2 | `constraints/workspace.lock` + `scripts/workspace.sh`；`outputs/*.py` → `tests/repro/` | 环境可复现 |
| P3 | `tools/matrix/` 拆分；provenance 接入报告；`npu-nightly.yml` 落到真实 runner 或 cron | 工具 |
