# ascend-torchtitan

[torchtitan](https://github.com/pytorch/torchtitan) 的**昇腾 NPU** 树外扩展。把它装在一个按 commit 固定的 torchtitan 旁边，torchtitan 就能在 NPU 上跑；昇腾融合算子、并行策略和图模式按 recipe 选择性开启。torchtitan 本身**从不 fork、从不在磁盘上打补丁**。

> 状态：**NIGHTLY 基线（2026-08-30）** —— torch nightly `2.15.0.dev20260812` + torch_npu master 源码构建 + torchtitan `13da2d77c`：**不加任何 shim**，qwen3 单卡 / FSDP2×2 golden 与 2.13 逐位一致；此前 2 条 shim、3 个 torchtitan 补丁、14 个 CP 红格被证实只是 torch 版本差，已删除。剩余的 6 个昇腾侧问题（stock varlen 内核、fake 进程组、复数索引、uint64 `zero_`、inductor 签名漂移、spmd_types 循环导入）**全部在 torch_npu / op-plugin 侧修复并带 UT**（`patches/`，走 gitcode PR）。M2 扫描（正式版 torch）：NEXT 25 🟢 / STABLE 18 🟢 of 61。详见 `docs/design/2026-08-30-architecture-review.md`、`docs/baseline.md`、`docs/issues/STATUS.md`。

## 安装

```bash
git clone https://github.com/JavaZeroo/ascend-torchtitan.git && cd ascend-torchtitan
pip install --pre -c constraints/nightly.txt torch --index-url https://download.pytorch.org/whl/nightly/cpu
source /usr/local/Ascend/cann-9.1.0/set_env.sh && ./scripts/build_torch_npu.sh     # torch_npu master 源码构建（约 9 分钟，本地盘）
TORCH_NPU_WHEEL=$(ls -t /opt/wheels/torch_npu-*.whl | head -1) WITH_TORCH=1 TITAN_DIR=../torchtitan ./scripts/install.sh
ascend-titan-doctor                                          # 打印版本元组和缺失项
```
RELEASE track（PyPI 的 torch_npu 2.13.0rc1，需要 2 条 shim）：`CONSTRAINTS=constraints/npu.txt WITH_TORCH=1 ./scripts/install.sh`。

## 运行

```bash
MODULE=ascend_titan.recipes.qwen3 CONFIG=qwen3_debugmodel_npu NPU=8 ./scripts/run_train.sh
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 ./scripts/check_golden.sh qwen3_debugmodel_npu   # 确定性运行并与 golden 对比
python -m ascend_titan.tools.matrix --cards 0-7 --jobs 4     # 把上游集成用例整体搬到 NPU 上扫描
```

`python -m ascend_titan.train` 等价于 `python -m torchtitan.train`，只是先执行了 `ascend_titan.setup()`（导入 torch_npu + 应用 compat shim）。其余全部是上游代码。

## 结构

| 层 | 包 | 机制 |
|---|---|---|
| L0 compat | `ascend_titan.compat` | 受治理的 monkeypatch，每条都挂上游 issue——NIGHTLY 上生效数为 0 |
| L1 kernels | `ascend_titan.kernels` | torchtitan `@override` 工厂，封装 AscendC / Triton-Ascend 算子 |
| L2 parallel / graph | `ascend_titan.parallel`、`.graph` | `ModelSpec.parallelize_fn` 替换、torchair 后端 |
| L3 recipes | `ascend_titan.recipes` | 上游配置 + 增量（delta）+ `override.imports` |
| L4 tools | `ascend_titan.tools` | `doctor`、矩阵扫描、provenance |

设计：`docs/design/2026-08-29-ascend-torchtitan-design.md`；评审与行动清单：`docs/design/2026-08-30-architecture-review.md`。原则：`docs/PRINCIPLES.md`（P0–P13）。决策记录：`docs/adr/`。问题清单：`docs/issues/`。与 Claude Code 协作：`CLAUDE.md` 与 `.claude/skills/`。

## 开发

```bash
pytest tests/unit -x          # CPU
ruff check .
./scripts/probe_compat.sh     # torchtitan 固定点最远能前进到哪个 commit
```

许可证：Apache-2.0。
