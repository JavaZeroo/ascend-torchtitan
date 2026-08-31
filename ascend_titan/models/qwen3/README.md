# Qwen3 on Ascend

**状态 🟢 release 级** · 参考模型：NIGHTLY 门禁跑的就是它，golden loss 曲线逐位冻结。
`docs/model-release-criteria.md` 的 R1–R8 每条都有记录下来的命令与输出，见下面第 6 节。

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
| 3 | `checkpoint.enable = False` | 冒烟不做 DCP I/O，DCP 是独立的矩阵格 | 有专门的 checkpoint recipe 后 |

## 4. 探针（`probes.py`，别用来训练）

| 函数 | 测的是 | 现在的预期 |
|---|---|---|
| `qwen3_debugmodel_stock_flex` | 上游默认 flex | 🔴 DEP-INDUCTOR（未装 Triton-Ascend） |
| `qwen3_debugmodel_stock_varlen` | 零 override 的上游 varlen | 🟢（带 NPU-1 补丁）/ 🔴（stock torch_npu）——这是判断 NPU-1 有没有合入的格子 |
| `qwen3_debugmodel_npu_ce_loss` | 非 chunked 的 `CrossEntropyLoss` | 🟢 `5.10304 / 3.3061`——与删除 DELTA 4 之前的旧 golden 逐位相同，即这条探针精确保留了原路径 |
| `qwen3_debugmodel_npu_fused_norm` | 只叠加 RMSNorm 一个算子 | 🟢 |

## 5. 真实尺寸（release 级）

判据见 `docs/model-release-criteria.md`。资产先备好（这台机器访问不到 huggingface.co，脚本走镜像）：

```bash
./scripts/fetch_assets.sh tokenizer Qwen/Qwen3-0.6B
./scripts/fetch_assets.sh tokenizer Qwen/Qwen3-8B           # PP 的证据用 8B，原因见下
./scripts/fetch_assets.sh c4 1                              # 真实 C4 分片 + 50k 篇子集
export ASCEND_TITAN_ASSETS=/opt/assets
```

| recipe | 卡 | 说明 |
|---|:--:|---|
| `qwen3_0_6b_npu` | 1 | Qwen3-0.6B，真实 tokenizer + 真实 C4 + 4096 上下文 |
| `qwen3_0_6b_npu_fsdp2` | 8 | FSDP2 8 路 |
| `qwen3_0_6b_npu_tp2` | 8 | FSDP2 4 × TP 2 |
| `qwen3_8b_npu_pp2` | 8 | Qwen3-8B × (PP 2 × FSDP2 4)，全量重计算 |

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu ./scripts/run_train.sh --training.steps 20
```

一次跑完全部 release 检查并留档：

```bash
python -m ascend_titan.tools.release_check --model qwen3 --cards 0-7 --out docs/release
```

### 一个会让人白查一天的坑：LR 曲线绑在 `--training.steps` 上

`lr_scheduler.total_steps` 缺省回落到 `training.steps`，`warmup_steps` 又会被 clamp 到它。
所以 `--training.steps 5` 和 `--training.steps 10` 的**前五步学习率不一样**——用短跑做
checkpoint 续训对比、或者拿 20 步的探针当性能基线时，都要显式钉住
`--lr_scheduler.total_steps`。`release_check` 与 `bench` 都已经这么做了。

### 为什么 PP 的证据用 8B 而不是 0.6B、也不是 14B

0.6B / 1.7B / 4B（连 debugmodel）都 tie 了 embedding 与 lm_head，而上游明确
`Weight tying is not supported with Pipeline Parallel`——这是上游限制，与昇腾无关
（实测：`qwen3_0_6b_npu` + `pipeline_parallel_degree=2` 直接抛 `NotImplementedError`）。
8B 是第一个不 tie 的尺寸。

14B 试过，装不下：`FullAC` + 1×4096 微批仍然 OOM（54.81 GiB 已分配时再要 1.60 GiB
失败）。参数、梯度与 AdamW 状态本身就占掉绝大部分，不是激活能省出来的。

PP 本身在昇腾上是通的：能力矩阵里 llama3 的 `pp_1f1b`、`pp_dp_1f1b`、
`pp_dp_tp`、`pp_looped_1f1b`、`pp_tp_gpipe`、`llama3_fsdp+tp+pp` 全绿。

## 6. release 判据逐条（`docs/model-release-criteria.md`）

| 判据 | 状态 | 证据 |
|---|:--:|---|
| R1 真实形态 | 🟢 | `qwen3_0_6b_npu`：Qwen3-0.6B + 真实 HF tokenizer + 真实 C4 + 4096 上下文，20 步 loss 12.11337 → 7.75390 |
| R2 并行覆盖 | 🟢 | 单卡 🟢（12.11337 → 7.75390）；FSDP2×8 🟢（12.12893 → 7.72891）；FSDP2×4+TP2 🟢（12.12286 → 7.71191）；PP2×FSDP2-4 🟢 —— `qwen3_8b_npu_pp2` 20 步 12.77976 → 9.45129（8B 是第一个不共享 embedding 的尺寸，见下） |
| R3 数值可信 | 🟢 | 四条 debugmodel golden 逐位冻结；500 步 loss 12.11569 → 6.28435 稳定下降（中间有正常抖动，不是单调）；`tests/npu/` 对 attention / rope / rms_norm / swiglu 逐个对上游 eager 对拍 |
| R4 checkpoint | 🟢 | DCP 存取 + 续训：第 5 步存档 → 续训到第 10 步 loss `9.42568`，与一口气跑到底**逐位相同**。HF 互操作：导出成 safetensors（`model-00001-of-00001.safetensors` + index）后用 `--checkpoint.initial_load_in_hf` 读回，第一步 loss `9.95988`（导出时 `10.23167`，随机初始 `12.14750`）——权重完整往返 |
| R5 性能基线 | 🟢 | `docs/bench/`，30 步取后半程中位数，每行都带 provenance：0.6B 单卡 10,186 tps / 65.14 TFLOPs / 19.08 GiB；0.6B FSDP2×8 9,440 tps / 60.37 TFLOPs / 10.29 GiB；8B PP2×FSDP2-4 1,409 tps / 74.22 TFLOPs / 51.43 GiB。MFU 那一列的分母是 torchtitan 回落的 A100 峰值（312 TFLOPS），不是 910B2 的，跨机器不可比 |
| R6 长稳 | 🟢 | 500 步 rc=0，无 NaN，显存自第 51 步起恒定 19.08 GiB |
| R7 文档 | 🟢 | 本文每条结论都有可照抄的命令 |
| R8 无隐藏降级 | 🟢 | `ascend-titan-provenance` 里注意力节点是 `ascend`，其余走上游默认 |

HF 互操作是 R4 的第三项，也是"能不能交付出去"的分界。`release_check` 会跑它：导出成
HF safetensors，再用 `--checkpoint.initial_load_in_hf` 读回来跑一步，要求 loss 贴着导出
时的值而不是弹回 `ln(vocab)`。注意 `--checkpoint.last_save_in_hf` 必须配
`--checkpoint.last_save_model_only`——HF 导出是模型快照，不是可续训的 checkpoint。

## 7. 待办

- **14B**：在 8×910B2 上的显存配平（`FullAC` + 1×4096 微批仍 OOM）。PP 的证据已经取在
  8B 上，这条只是想把更大的尺寸也跑起来。
- 1.7B / 32B / 30B-A3B 等其它真实尺寸（⚪）。
