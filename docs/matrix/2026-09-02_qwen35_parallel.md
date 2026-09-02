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
| MoE dp2×tp2×pp2×ep4（`qwen35_debugmodel_moe`） | 🟢 `loss 12.55579` | 🟢 `loss 12.31024` |

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
不该再覆盖并行度。用自带布局重跑，两格都绿（`loss 12.55579` / `12.31024`，各 4 个 step 行——
`--local-ranks-filter 0,7` 同时打了首尾两段 PP stage）。

**这一格顺带否掉一个担心**：9-01 那轮 `qwen3_5_moe_fsdp+tp+ep` 与 `qwen3_5_moe_fsdp+tp+ep+pp`
红在 TA-1 上，我一度以为 qwen3.5 的 MoE + TP/EP/PP 组合在昇腾上不通。实际是通的——
上游那两个用例走的是 `npu_minimal` 的通用变换（含 flex→varlen 与 ComplexRoPE 覆盖），
而 debugmodel_moe 自带布局跑我们的 recipe 是绿的。**TA-1 挡住的是那两个用例的具体组合，
不是「MoE + 并行」这个能力。** 这两件事之前被我混在一起说了。

## 追加：TA-1 的三个红格全部转绿（2026-09-02 重测）

9-01 那轮 qwen3.5 有三格红、归因 TA-1，我一度据此说「qwen3.5 的 MoE + TP/EP/PP
在昇腾上不通」。用矩阵工具重测，**三格全绿**：

| 用例 | 9-01 | 现在 |
|---|:--:|:--:|
| `qwen3_5_fsdp+tp+varlen_attn+per_op_sac` | 🔴 TA-1 | 🟢 53s |
| `qwen3_5_moe_fsdp+tp+ep` | 🔴 TA-1 | 🟢 72s |
| `qwen3_5_moe_fsdp+tp+ep+pp` | 🔴 TA-1 | 🟢 59s |

原因说得通：TA-1 是**编译**掩码构建时 autotune 走进 SIMT 路径才触发的。今天把
`flex_block_mask_eager` 的门从「确定性模式」放宽到硬件探测之后，掩码构建不再进
inductor，那条路径就不存在了。

同一轮还跑了 stock 对照（`__stock`，零 override），`qwen35_debugmodel_varlen_attn_fsdp2_tp2_sac`
与 `qwen35_debugmodel_moe_fsdp4_tp2_ep4` 也都绿——**两条路径依然逐格一致**。

**TA-1 因此不是阻塞**：它挡住的矩阵格子现在是 0 个。重新界定见
`docs/issues/triton-ascend.md`。
