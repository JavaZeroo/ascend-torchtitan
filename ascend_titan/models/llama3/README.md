# Llama 3 on Ascend

> 910B2 / CANN 9.1.0 / torch 2.15.0.dev20260812 + torch_npu master + `patches/`。
> ✅ 实测通过 ｜ ❌ 实测失败 ｜ ❓ 没测过（不代表坏）
>
> **零 override：上游 llama3 在昇腾上原样训练。** 上游 `tests/integration_tests/features`
> 那一整套并行 / checkpoint / compile 用例（38 格）也都跑在它上面。

## 场景总表

| 模型 | 场景 | 状况 | 限制 |
|---|---|:--:|---|
| **`llama3_debugmodel`**<br>原生上游 | 单卡 / FSDP2 / HSDP / DDP | ✅ | 无 |
| | TP + SP（`2d_eager`、`hsdp+tp`） | ✅ | 无 |
| | PP（1F1B / GPipe / looped / PP+DP+TP） | ✅ | 无 |
| | CP（`cp`、`fsdp+tp+cp`、`validation_tp_cp_pp`） | ✅ | CP 下注意力只能走 eager flex（2026-09-02 转绿） |
| | `fsdp+cp` / `hsdp+cp_*`（低 CP 度） | ❌ | **显存**：CP 度低 → 每卡序列长 → eager flex 的 O(T²) 装不下 |
| | torch.compile（1d / 2d / 3d / sac_op） | ✅ | 需要 Triton-Ascend 在基线里 |
| | checkpoint（full / optional / seed / HF / bf16-only） | ✅ | 无 |
| | 梯度累积、varlen + SAC、SFT | ✅ | 无 |
| | `float8_emulate_lora` | ❌ | **硬件**：910B2 能存 float8 但没有 cast 内核，要 Ascend950 |
| | `override_fused_swiglu` / `override_fused_grouped_experts` | ❌ | **上游按设计 CUDA-only**（树内 Triton 内核）。昇腾替代见 `kernels/swiglu.py` |
| | `2d_asynctp_compile`、`pp_zbv`、`pp_custom_csv`、`pp_looped_zero_bubble` | ❓ | 上游自己禁用了这些用例 |
| | 真实尺寸 / 长稳 / 性能基线 | ❓ | 只有 debugmodel，R1–R8 一条都没取 |
| **`llama3_debugmodel_stock_npu`**<br>唯一增量：flex → varlen | 单卡 / FSDP2×2 | ✅ | 这条增量不是 llama3 特有的，是 910B2 上所有用 flex 的模型共有的 |

## 能替换哪些模块

**一个都不用换**，这是本模型最重要的结论：复数 RoPE、`ChunkedLoss`、`spmd_types`
全部走上游默认实现。

唯一的例外是 flex → varlen → 昇腾融合注意力，而它是**通用限制**（见
`ascend_titan/models/README.md` 的「三个通用限制」），不是 llama3 的问题。
CP 场景下这条转换会自动跳过。

---

## 以下是使用与实现细节

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
910B2 编不了 flex 的 document mask（硬件门）。其余**全部**是上游默认：

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

结果见 `docs/capability-matrix.md`（FSDP2 / DDP / HSDP / TP+SP / PP 全绿；CP 停在硬件门）。

## 5. 真实尺寸（⚪）

`llama3_8b` 等尚未评估。跑法同 qwen3 README 第 5 节（走矩阵 runner，别新建配置）。
