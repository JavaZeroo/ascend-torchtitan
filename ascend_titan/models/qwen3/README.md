# Qwen3 on Ascend

**状态 🟢** · 参考模型：NIGHTLY 门禁跑的就是它，golden loss 曲线逐位冻结。

| | |
|---|---|
| 上游模型包 | `torchtitan.models.qwen3`（`../torchtitan/torchtitan/models/qwen3/`） |
| 我们的 recipe | `ascend_titan/models/qwen3/recipes.py` |
| 矩阵探针 | `ascend_titan/models/qwen3/probes.py`（只用于测量，别拿来训练） |
| golden | `tests/assets/losses/npu/qwen3_debugmodel_npu*__torch<v>_npu<v>.txt`；四条曲线（参考 / FSDP2 / fused / fused_fsdp2）在 NIGHTLY 上均已录制，且与 torch 2.13 的曲线逐位一致 |
| 最近验证 | torch 2.15.0.dev20260812 + torch_npu 2.15.0（master 源码构建 + `patches/`），CANN 9.1.0，2026-08-30 |

## 1. 五分钟跑起来

```bash
# 前置：已按 README 装好 NIGHTLY（torch nightly + 源码构建的 torch_npu + 固定 SHA 的 torchtitan）
source /usr/local/Ascend/cann-9.1.0/set_env.sh
ascend-titan-doctor                       # 先确认版本元组齐全

# 单卡 10 步冒烟
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_debugmodel_npu ./scripts/run_train.sh

# FSDP2 ×2
ASCEND_RT_VISIBLE_DEVICES=0,1 NPU=2 \
MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_debugmodel_npu_fsdp2 ./scripts/run_train.sh

# 与冻结的 golden 逐位对比（确定性开关由脚本加）
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 ./scripts/check_golden.sh qwen3_debugmodel_npu
```

预期输出（NIGHTLY，2026-08-30）：`step: 10  loss:  5.10291  grad_norm:  3.3062`，
`check_golden.sh` 打印 `GOLDEN MATCH: qwen3_debugmodel_npu @ torch2.15.0.dev20260812_npu2.15.0`。

## 2. 有哪些 recipe

| 函数 | 卡数 | 说明 |
|---|:--:|---|
| `qwen3_debugmodel_npu` | 1 | **参考路径**。上游 `qwen3_debugmodel` + 4 条增量，golden 冻结 |
| `qwen3_debugmodel_npu_fsdp2` | 2 | 上面 + `data_parallel_shard_degree=2`，golden 冻结 |
| `qwen3_debugmodel_npu_fused` | 1 | 再叠加三个零构建的 torch_npu 融合算子（RMSNorm / SwiGLU / rotary）。**性能** recipe，数值与 golden 差在 bf16 舍入级别，自己有一条 golden |
| `qwen3_debugmodel_npu_fused_fsdp2` | 2 | 同上 ×2 卡 |

命名规则：`qwen3_<flavor>_npu[_<variant>]`。`<flavor>` 是上游 config registry 的名字，`<variant>` 是并行或算子增量。

## 3. 参考 recipe 的四条增量（每条都是矩阵里的一格）

| # | 增量 | 为什么 | 什么时候能删 |
|---|---|---|---|
| 1 | `attn_backend="varlen"` + `kernels.attention` override | 上游只剩 `flex` / `varlen`；模型级 flex 走 inductor（DEP-INDUCTOR），stock varlen 需要 `aten::_flash_attention_forward`（NPU-1） | 装上 Triton-Ascend 后 flex 可评估；NPU-1 合入后 stock varlen 也能跑（见 `probes.py`），但融合算子仍是性能路径 |
| 2 | `spmd_backend="partial_dtensor"` | 正式版 torch 上 `spmd_types` + FSDP2 拿到的是普通张量（TT-5）。NIGHTLY 上上游默认已可用（llama3 stock 就在用），这里保留只为让 golden 跨 track 可比 | 弃用 RELEASE track 时 |
| 3 | `checkpoint.enable = False` | 冒烟不做 DCP I/O，DCP 是独立的矩阵格 | 有专门的 checkpoint recipe 后 |

> **loss 用上游默认的 `ChunkedLossWrapper`**（2026-08-30 起）。此前有一条 DELTA 4 把它换成
> 普通 `CrossEntropyLoss`，原因是正式版 torch + NPU 上 chunked loss 的 backward 撞
> "data is not allocated yet"（TT-4）——那是 torch 版本差，NIGHTLY 上不存在。按 P8/P12
> 这条增量已删除：参考路径回到上游默认，chunked loss 是被支持并有 golden 门禁的。
> 需要对比非 chunked 路径时用探针 `qwen3_debugmodel_npu_ce_loss`。

## 4. 探针（`probes.py`，别用来训练）

| 函数 | 测的是 | 现在的预期 |
|---|---|---|
| `qwen3_debugmodel_stock_flex` | 上游默认 flex | 🔴 DEP-INDUCTOR（未装 Triton-Ascend） |
| `qwen3_debugmodel_stock_varlen` | 零 override 的上游 varlen | 🟢（带 NPU-1 补丁）/ 🔴（stock torch_npu）——这是判断 NPU-1 有没有合入的格子 |
| `qwen3_debugmodel_npu_ce_loss` | 非 chunked 的 `CrossEntropyLoss` | 🟢 `5.10304 / 3.3061`——与删除 DELTA 4 之前的旧 golden 逐位相同，即这条探针精确保留了原路径 |
| `qwen3_debugmodel_npu_fused_norm` | 只叠加 RMSNorm 一个算子 | 🟢 |

## 5. 真实尺寸（⚪ 未评估）

上游 registry 里还有 `qwen3_0_6b`、`qwen3_1_7b`、`qwen3_14b`、`qwen3_32b`、`qwen3_30b_a3b`、
`qwen3_moe_debug` 等。它们**尚未在昇腾上跑过**，按 P2 记为 ⚪ 而不是 🟢。要跑的话不要新建配置，
走矩阵 runner 直接跑上游配置 + `npu_baseline`：

```bash
MODULE=ascend_titan.recipes.matrix \
CONFIG=torchtitan.models.qwen3.config_registry__qwen3_0_6b \
NPU=8 ./scripts/run_train.sh
```

跑绿之后再决定要不要在 `recipes.py` 里固化成一条 recipe（附 golden + `validated:` 头行）。

## 6. 待办

- **DELTA 2** 随 RELEASE track 退役一并删除（NIGHTLY 上上游默认的 `spmd_types` 已可用）。
- 0.6B 起的真实尺寸扫描（上表第 5 节）。
