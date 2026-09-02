<!-- 手工网格，不是 tools.matrix 的用例集：上游 tests/integration_tests 里没有 qwen3_5 的 CP/PP 用例，所以这些格子在能力矩阵里一直是 ⚪。-->
# qwen3.5 的并行空白格（2026-09-02）

`models/registry.py` 里 qwen3.5 的 R2 判据长期是 🟡「单卡 / FSDP2×8 🟢；TP/PP/EP 未测」。
这里把它测掉，并且**两条路径各测一遍**：原味上游（零 override）与换上我们的融合算子。

命令形如（`<cfg>` = `qwen35_debugmodel` 或 `qwen35_debugmodel_npu`）：

```bash
torchrun --nproc_per_node=8 -m ascend_titan.train \
  --module ascend_titan.models.qwen3_5 --config <cfg> \
  --training.steps 2 --checkpoint.no-enable --parallelism.<...>
```

| 布局 | stock | 换融合算子 |
|---|:--:|:--:|
| 单卡 | 🟢 `loss 12.72494 → 12.56159` | 🟢（golden 已冻结） |
| FSDP2×8 | 🟢 `loss 12.22373` | 🟢 `loss 12.00507` |
| TP2 × FSDP2-4 | 🟢 `loss 12.01387` | 🟢 `loss 12.02401` |
| PP2 × FSDP2-4 | 🟢 2 步 | 🟢 2 步 |
| CP4 × FSDP2-2 | 🔴 上游不支持 | 🔴 同一条 |

**两条路径逐格一致。** 差别只在性能，不在能力边界。

## CP：上游明确不支持 qwen3.5，与昇腾无关

stock 与融合算子两条路径报的是**同一条**错：

```
NotImplementedError: Context Parallel is not yet supported for Qwen3.5.
GatedDeltaNet (75% of layers) requires full-sequence allgather...
```

qwen3.5 有 75% 的层是 gated delta net，它要整条序列，序列分片本身就不成立。
这**不是** `docs/capability-matrix.md` 里那条 CP 护栏（`decoder.py:186`，varlen/SDPA 不支持 CP），
也不是芯片，也不是我们的 override——**stock 一样跑不了**。

所以 qwen3.5 这一族的 CP 是 ⚫「上游按设计不支持」，不该记成红格，也不该出现在
「让 CP 走融合算子」那个待办的收益里（那个待办的受益者是 llama3 / qwen3 / deepseek_v3 / gpt_oss）。

## 两个不下的结论

**不拿这些 loss 论证 flex 与融合算子的数值等价。** `stock_fsdp8`(12.22373) 与
`npu_fsdp8`(12.00507) 差 0.22，但 `stock_tp2`(12.01387) 与 `npu_tp2`(12.02401) 几乎一样——
更像数据顺序的 run-to-run 波动。数值等价该由 `tests/npu/test_kernel_attention.py`
的 fp32 逐项对拍来说（已绿）；真要在模型级比，得固定种子跑同一条数据。

**PP 那两格的绿不带 loss 数字**：loss 只在最后一段 PP stage 打印，
被 `--local-ranks-filter 0` 挡了。用 `step:` 行数确认过各有 2 步，rc=0。

## 过程中的自查

`stock_moe_tp2ep2` / `npu_moe_tp2ep2` 第一次红在
`Invalid parallel dims: dp_shard(4) * tp(2) * pp(2) != WORLD_SIZE(8)`——**是脚本参数写错**，
不是缺陷：`qwen35_debugmodel_moe` 自带 dp2×tp2×pp2×ep4=8，已经是合法的 8 卡布局，
不该再覆盖并行度。用自带布局重跑，结果见下（待补）。
