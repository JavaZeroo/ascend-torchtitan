# ascend-torchtitan

[torchtitan](https://github.com/pytorch/torchtitan) 的**昇腾 NPU** 树外扩展。把它装在一个按 commit 固定的 torchtitan 旁边，torchtitan 就能在 NPU 上跑；昇腾融合算子、并行策略和图模式按 recipe 选择性开启。torchtitan 本身**从不 fork、从不在磁盘上打补丁**。

> 状态：**M2 进行中（2026-08-29）** —— `qwen3_debugmodel` 已在 Ascend 910B2 上完成训练（单卡与 FSDP2×2），只用了一个融合注意力 override + 一个 polyfill shim；torch 2.12.0/2.13.0 + torch_npu 2.12.0/2.13.0rc1 两套组合 loss 曲线逐位一致。
> 上游 57 个集成用例的全量矩阵扫描正在进行，每个红格都带归因。详见 `docs/baseline.md`、`docs/capability-matrix.md`、`docs/issues/`。

## 安装

```bash
git clone https://github.com/JavaZeroo/ascend-torchtitan.git && cd ascend-torchtitan
WITH_TORCH=1 TITAN_DIR=../torchtitan ./scripts/install.sh   # torch + torch_npu + 固定 SHA 的 torchtitan（不带 CUDA-only extras）
ascend-titan-doctor                                          # 打印版本元组和缺失项
```

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
| L0 compat | `ascend_titan.compat` | 受治理的 monkeypatch，每条都挂上游 issue——数量目标是归零 |
| L1 kernels | `ascend_titan.kernels` | torchtitan `@override` 工厂，封装 AscendC / Triton-Ascend 算子 |
| L2 parallel / graph | `ascend_titan.parallel`、`.graph` | `ModelSpec.parallelize_fn` 替换、torchair 后端 |
| L3 recipes | `ascend_titan.recipes` | 上游配置 + 增量（delta）+ `override.imports` |
| L4 tools | `ascend_titan.tools` | `doctor`、矩阵扫描、provenance |

设计：`docs/design/2026-08-29-ascend-torchtitan-design.md`。原则：`docs/PRINCIPLES.md`。决策记录：`docs/adr/`。问题清单：`docs/issues/`。与 Claude Code 协作：`CLAUDE.md` 与 `.claude/skills/`。

## 开发

```bash
pytest tests/unit -x          # CPU
ruff check .
./scripts/probe_compat.sh     # torchtitan 固定点最远能前进到哪个 commit
```

许可证：Apache-2.0。
