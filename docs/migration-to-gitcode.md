# 迁移到 gitcode.com/renyujin/torchtitan_npu

## 0 迁的是哪个仓

你说的是 `ascend-pytorch`，但那个目录是 `gitcode.com/Ascend/pytorch`（torch_npu）的克隆，
而目标仓 `renyujin/torchtitan_npu` 的描述写的是"将 torchtitan 适配至昇腾 NPU"。
按目标仓的定位，这份计划迁的是 **`ascend-torchtitan`**（本仓）。

`ascend-pytorch` 上那 9 个 `fix/*` 分支**不迁移**：按 P9，它们的归宿是 `Ascend/pytorch` 的 PR。
本仓只保留 `patches/torch_npu/` 里的 format-patch 和 `docs/issues/STATUS.md` 里的 PR 链接。
如果你要迁的确实是 torch_npu 那一套，这份计划作废，重写。

## 1 现状（2026-09-04 核实）

| 项 | 值 |
|---|---|
| 源 | `ascend-torchtitan`，远端 `github.com/JavaZeroo/ascend-torchtitan`（公开） |
| 源体积 | 141 个提交，199 个受控文件，`.git` 9.6M，最大历史 blob 64K（`docs/design.html`）；工作区 2.0G 全部被 `.gitignore` 挡住，`git status` 干净 |
| 源作者 | `JavaZeroo <2487163254@qq.com>` 107 个，`ascend-torchtitan <jimmy.eyjc@gmail.com>` 34 个 |
| 目标 | `gitcode.com/renyujin/torchtitan_npu`，**私有**，2026-09-02 建，`main` 上只有一个 `Initial commit` `f7b15ed`（一个 README.md） |
| 目标成员 | `renyujin`（owner，renyujin1@huawei.com）、`jimmyisme1`（**Maintainer**，access_level 40） |
| 推送权限 | 已验证：`git ls-remote` 用 `GITCODE_TOKEN` 能读，角色是维护者，可推 |

两个后果值得先记住：目标仓**私有**，所以现在推上去不等于开源；目标仓在 `renyujin` 个人名下，
不是 `Ascend` 组织，所以**没有 Ascend 的门禁机器人**——CI 得自己带。

## 2 推第一行代码之前必须定的四件事

这四件事定了才动手。每件我给了推荐，但 D1 和 D4 是 renyujin 的决定。

### D1 包名要不要跟着仓名改

仓叫 `torchtitan_npu`，包叫 `ascend_titan`，发行名叫 `ascend-torchtitan`，
命令行叫 `ascend-titan-doctor`，环境变量叫 `ASCEND_TITAN_SKIP_SHIMS`。四个名字，三种写法。

改名的范围：134 个文件、622 处（`ascend_titan` 540、`ascend-titan` 31、`ascend-torchtitan` 29、`ASCEND_TITAN` 22）。
机械替换 + 目录重命名 + 一次 `pytest`，半天以内。

**推荐现在改**，理由是目标仓零下游用户，这是改名最便宜的时刻；
`import torchtitan_npu` 和 `import torch_npu` 对齐，日后往 Ascend 组织搬也不用再改一次。
反对意见也成立：本地 `../ascend-torchtitan`、`.claude/skills/`、golden 资产路径都要跟着动。
**如果 renyujin 不表态，就改**——拖到有人依赖之后再改代价是现在的十倍。

### D2 历史保留还是压平

141 个提交是一周的工作，里面有反复和回退，但也正是 P13（验证先于断言）的证据链：
每个 🟢 背后的命令和输出都能顺着提交找回去。

**推荐：两者都要。**
完整历史推成一个只读归档分支 `archive/github-history`，永不作为基线；
`main` 用第 3 节的五个批次重建，每批一个可评审的 PR。
这样 renyujin 拿到的是五个能读完的 PR，而证据链一行没丢。

顺带把作者身份统一掉：107 个提交挂的是 QQ 邮箱，34 个挂的是 Gmail，
推到华为内部仓之前用 `git filter-repo --mailmap` 归一到 `lijunbiao6@huawei.com`。

### D3 门禁怎么接

`.github/workflows/ci.yml`（CPU，ubuntu-latest）和 `npu-nightly.yml`（自建 NPU runner）
**在 GitCode 上不会运行**——GitHub Actions 不跨平台。`Ascend/pytorch` 的做法是 `.gitcode/PULL_REQUEST_TEMPLATE.md`
加服务端配置的门禁机器人，那是组织仓的待遇，个人仓没有。

**推荐分两步**：迁移期先把两条流水线降级成仓内脚本 `scripts/ci_cpu.sh` 和 `scripts/ci_npu.sh`
（内容就是 workflow 里那几行），在 PR 描述里贴执行输出，人肉当门禁；
落地之后再问 renyujin 有没有可挂的 runner，有就接上，没有就在 NPU 开发机上挂 cron 跑 `ci_npu.sh` 并把结果写回仓里。
**不要**为了省事把 GitHub 仓留着跑 CI——两个事实来源会立刻打架（P11）。

### D4 私有仓 vs 开源合规

现在是私有仓，可以先不管合规；**但公开那天之前必须补齐**，而补齐的成本随时间涨，所以放进计划：

- 本仓不 vendor 任何上游代码（已核实：没有 Meta/BSD 版权头的文件），但
  `patches/evidence/pytorch/*.patch` 里含 PyTorch 源码片段（BSD-3），
  `patches/torch_npu/*.patch` 含 torch_npu 代码 —— 需要 `Third_Party_Open_Source_Software_Notice` 声明。
- 对齐 Ascend 仓的文件集：现有 `LICENSE`（Apache-2.0）、`README.md`、`CONTRIBUTING.md`、`SECURITY.md`；
  缺 `OWNERS`、`SECURITYNOTE.md`（Ascend 用这个名字，不是 `SECURITY.md`）、`Third_Party_Open_Source_Software_Notice`，
  以及英文版 `README.en.md` / `CONTRIBUTING.en.md`。
- `CLAUDE.md` 里有开发容器的细节（`ssh root@localhost`、镜像名、`/data` NFS 布局）。
  内部仓无所谓，公开前删掉。仓里没有 IP、口令、token（已扫过工作区与历史）。

## 3 分期

### 第 0 期 · 本地准备，不推任何东西

在本仓开 `migrate/gitcode` 分支做完这些，全程可回退：

1. D1 定了就执行改名，一个提交；`pytest tests/unit` + `ruff check .` 必须绿。
2. 重写指向 GitHub 的链接：12 个文件里有 `JavaZeroo` 字样，包括 `README.md` 的克隆命令和徽章、
   以及 9 个 patch 头部。改成 `gitcode.com/renyujin/torchtitan_npu`。
3. 补 D4 列的合规文件；`SECURITY.md` → `SECURITYNOTE.md`。
4. `.github/` → `.gitcode/`：PR 模板按 `Ascend/pytorch` 的格式重写，issue 模板保留内容换目录，
   两条 workflow 转成 `scripts/ci_cpu.sh` / `scripts/ci_npu.sh`。
5. `AGENTS.md` 是指向 `CLAUDE.md` 的符号链接，GitCode 支持，不用动。

验收：本地 `pip install -e .` → `ascend-titan-doctor` → `pytest tests/unit -x` → `ruff` 全绿。

### 第 1 期 · 归档分支先上去

```bash
cd /data/ljb/projects/create-ascend-titian/ascend-torchtitan
git remote add gitcode https://gitcode.com/renyujin/torchtitan_npu.git
git push gitcode migrate/gitcode:archive/github-history
```

一条命令，把全部历史落到目标仓的只读分支。之后 GitHub 那个仓就可以停止接收新提交了。
**先做这一步的理由**：万一后面几期出岔子，代码已经在目标仓里了。

### 第 2–6 期 · 五个批次重建 main

每批从 `f7b15ed`（目标仓自己的 Initial commit）往上长，一个 PR，一个验收标准。
顺序不能换：后一批的验收依赖前一批。

| 批次 | 内容 | 验收 |
|---|---|---|
| **B1 骨架与约束** | `LICENSE`、`README.md`、合规文件集、`.gitignore`、`pyproject.toml`、`constraints/*`、`scripts/install.sh`、`scripts/build_torch_npu.sh`、`docs/PRINCIPLES.md`、`docs/adr/*`、`docs/glossary.md`、`ascend_titan/__init__.py`、`tools/doctor.py` | `pip install -e .` 通过，`ascend-titan-doctor` 在 CPU 上出报告 |
| **B2 机制层** | `_bootstrap.py`、`compat/`、`kernels/`、`graph/`、`recipes/`、`tools/` 全量、`tests/unit/` 全量 | `pytest tests/unit -x` 绿，`test_import_purity` 保证 import 无副作用 |
| **B3 模型层** | `models/*`（llama3 / qwen3 / qwen3_5 / kimi_k3 / _template）、`train.py`、`scripts/` 其余、`tests/npu/`、`tests/assets/losses/` | NPU 上 `ASCEND_TITAN_SKIP_SHIMS=1 ./scripts/check_golden.sh qwen3_debugmodel_npu` → `GOLDEN MATCH`；`pytest tests/npu -q` |
| **B4 证据与文档** | `docs/issues/`、`patches/`、`docs/matrix/`、`docs/bench/`、`docs/capability-matrix.md`、`docs/baseline.md`、`docs/roadmap.md`、`docs/design/`、三个 html | `pytest tests/unit/test_patches_and_status.py`（每个补丁必须有 Upstream-PR 链接） |
| **B5 协作面** | `.gitcode/` 模板、`OWNERS`、`scripts/ci_*.sh` 接线、分支保护、`.claude/` skills 与 rules | main 设为保护分支，PR 必须过 `ci_cpu.sh` |

B1–B2 可以同一天推完（纯 CPU 验收）。B3 卡在 NPU 机器和 8 卡排队上，是唯一有等待成本的一批。

### 第 7 期 · 切事实来源

1. 本仓 `origin` 改指 GitCode，GitHub 仓设为只读归档，README 顶部一行指向新地址。
2. `CLAUDE.md` 里的"同级检出"段落、`.claude/skills/*` 里的远端名更新。
3. `docs/upstream-tracking.md` 和 `patches/README.md` 里凡是写 GitHub 路径的地方跟着改。
4. 全局 memory 里的 `ascend-torchtitan project` 条目改成新仓地址。

验收：`grep -rn "JavaZeroo" .` 为空。

## 4 风险

| 风险 | 说明 | 处置 |
|---|---|---|
| **门禁真空期** | 第 0 期起 GitHub Actions 就不再是门禁，GitCode 侧的门禁要到 B5 才有。中间三四批全靠人肉贴输出。 | 每个 PR 描述里贴 `ci_cpu.sh` 的完整输出，不贴不合入。B3 的 golden 结果同样。 |
| **和 TorchTitanTurbo 撞车** | `gitcode.com/Ascend/TorchTitanTurbo` 在做同一件事（torchtitan 的 NPU converters + patches 插件库）。 | 迁移前问 renyujin 定位：合并、并行、还是替代。这个答案会改变 B1 的 README 怎么写。 |
| **私有仓的开源路径** | 现在私有，将来若要进 Ascend 组织或公开，D4 那批合规文件是硬门槛。 | 在 B1 就补齐，别等公开前突击。 |
| **改名与不改名都有成本** | 见 D1。 | 定了就一次做完，不要半改。 |
| **历史里的个人邮箱** | 107 个提交挂 QQ 邮箱。 | `git filter-repo --mailmap` 在第 0 期做掉，归档分支推上去之前。 |

## 5 明确不做的事

- 不迁 `ascend-pytorch` / `ascend-op-plugin` 的 `fix/*` 分支（P9：归宿是 Ascend 上游 PR）。
- 不把 `../torchtitan` fork 进来（P1 / ADR-001）。
- 不为迁移写任何兼容层、别名包或 `try: import`（P14）。改名就是改名，不留旧名转发。
- 不在 GitHub 和 GitCode 之间做双向同步（P11：单一事实来源）。
