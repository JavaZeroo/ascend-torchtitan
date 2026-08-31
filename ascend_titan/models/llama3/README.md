# Llama 3 on Ascend

**状态 🟢** · **零 override**：上游 llama3 在昇腾上原样训练。这是本项目最强的一条结论。

| | |
|---|---|
| 上游模型包 | `torchtitan.models.llama3` |
| 我们的 recipe | `ascend_titan/models/llama3/recipes.py` |
| 最近验证 | torch 2.15.0.dev20260812 + torch_npu 2.15.0（master + `patches/` 六个修复），2026-08-30 |
| 实测（`--debug.seed 42 --debug.deterministic`，两次一致） | 单卡 `step: 10  loss:  4.01820  grad_norm:  1.7382`；FSDP2×2 `loss 3.97774  grad_norm: 1.7523` |

## 1. 跑

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
MODULE=ascend_titan.models.llama3 CONFIG=llama3_debugmodel_stock_npu ./scripts/run_train.sh \
    --debug.seed 42 --debug.deterministic          # 去掉这两个开关 loss 会有正常抖动

ASCEND_RT_VISIBLE_DEVICES=0,1 NPU=2 \
MODULE=ascend_titan.models.llama3 CONFIG=llama3_debugmodel_stock_npu_fsdp2 ./scripts/run_train.sh
```

## 2. recipe

| 函数 | 卡数 | 说明 |
|---|:--:|---|
| `llama3_debugmodel_stock_npu` | 1 | 上游 `llama3_debugmodel`，`override.imports = []` |
| `llama3_debugmodel_stock_npu_fsdp2` | 2 | 同上 + 2 路 FSDP2 |

## 3. 唯一的增量

`FlexAttention` → `VarlenAttention`。原因：模型级 flex 要经 inductor 编译，昇腾上需要
Triton-Ascend（DEP-INDUCTOR，未装）。其余**全部**是上游默认：

| 上游默认 | 在昇腾上曾经/仍然依赖什么 |
|---|---|
| `ComplexRoPE`（复数 cache 索引） | **NPU-3**：`aclnnIndex` 不支持复数 → 已在 op-plugin 修复并提 PR |
| `ChunkedLossWrapper` | **TT-4**：正式版 torch 上 backward 报 "data is not allocated yet"；NIGHTLY 🟢 |
| `spmd_types` 后端 | 上游默认，NIGHTLY 直接可用。（历史上正式版 torch 的 FSDP2 不读 `spmd_types`；NIGHTLY 🟢 |
| stock `VarlenAttention` → `aten::_flash_attention_forward` | **NPU-1**：torch_npu 没有该算子的 NPU 内核 → 已在 torch_npu 修复并提 PR |

**这条路径变红 = 昇腾侧回归。** 它是 `patches/` 里那几个修复的验收用例：任何一个失效，这里先红。

## 4. 并行覆盖

上游 `tests/integration_tests/features.py` 的并行用例几乎都跑在 llama3 debugmodel 上，我们通过矩阵 runner
整体搬到昇腾（不复制配置）：

```bash
python -m ascend_titan.tools.matrix --cards 0-7 --jobs 4 --suite features
```

结果见 `docs/capability-matrix.md`（FSDP2 / DDP / HSDP / TP+SP / PP 全绿；CP 停在 DEP-INDUCTOR）。

## 5. 真实尺寸（⚪）

`llama3_8b` 等尚未评估。跑法同 qwen3 README 第 5 节（走矩阵 runner，别新建配置）。
