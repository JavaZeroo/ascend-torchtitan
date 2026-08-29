# ascend-torchtitan

[torchtitan](https://github.com/pytorch/torchtitan) 的**昇腾 NPU** 树外扩展。把它装在一个按 commit 固定的 torchtitan 旁边，torchtitan 就能在 NPU 上跑；昇腾融合算子、并行策略和图模式按 recipe 选择性开启。torchtitan 本身**从不 fork、从不在磁盘上打补丁**。

> 状态：**M2 完成（2026-08-29）** —— 上游 61 个集成用例整体搬到 Ascend 910B2 上扫描：NEXT track（torch 2.13.0 / torch_npu 2.13.0rc1）**24 🟢**，STABLE（2.12.0 / 2.12.0）18 🟢；FSDP2 / HSDP / TP / PP / EP（deepseek_v3 MoE）/ DCP 与 HF checkpoint / SFT 全部可跑，靠两个 L1 override（融合注意力、实数缓存 RoPE）和两条 shim。剩余红格全部归因到 6 个根因，**没有一个落在 NPU/CANN 上**。详见 `docs/capability-matrix.md`、`docs/baseline.md`、`docs/issues/`。

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
