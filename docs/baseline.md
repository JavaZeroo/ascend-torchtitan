# 基线

2026-08-30 起基线 = **NIGHTLY**（ADR-006）。硬件：Ascend 910B2 ×8，驱动 25.5.1，CANN 9.1.0（镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12-devel`，容器 `ascend-titan-dev`，venv `/opt/venv-nightly`）。

## 版本三元组

| track | torch | torch_npu | torchtitan | 文件 | 状态 |
|---|---|---|---|---|---|
| **NIGHTLY**（默认、唯一门禁） | 2.15.0.dev20260812+cpu（torch_npu master `requirements_2.15.txt` 的 pin） | master `15514cc70` 源码构建（op-plugin `2b02a5aa0`），`ci/build.sh --python=3.12 --torch=2.15.0 --disable_torchair`，256 核 gcc 11.4 **8 分 28 秒** | `13da2d77c` | `constraints/nightly.txt` + `torch_npu.sha` + `torchtitan.sha` | 🟢 |
| RELEASE（信息性） | 2.13.0（+cpu wheel） | 2.13.0rc1（PyPI） | `13da2d77c` | `constraints/npu.txt` | 🟢（需 2 条 shim） |
| ~~STABLE~~ | 2.12.0 | 2.12.0 | `13da2d77c` | `constraints/npu-stable.txt` | 废弃 |

为什么是 nightly：torchtitan main 与 torch_npu master 都面向 torch nightly；正式版基线造出的 2 条 shim、5 个 torchtitan 补丁中的 3 个、14 个 CP 红格全是版本差（`docs/design/2026-08-30-architecture-review.md` §2）。

## NIGHTLY 实测（2026-08-30，全部 `ASCEND_TITAN_SKIP_SHIMS=1`，即**不加任何 shim**）

| 路径 | 结果 |
|---|---|
| `qwen3_debugmodel_npu` 单卡 10 步（seed 42，deterministic） | 🟢 **与 2.13 golden 逐位一致**（step 10 loss 5.10304，grad_norm 3.3061）；golden `tests/assets/losses/npu/qwen3_debugmodel_npu__torch2.15.0.dev20260812_npu2.15.0.txt` |
| `qwen3_debugmodel_npu_fsdp2` ×2 | 🟢 逐位一致（step 10 loss 5.07792，grad_norm 3.3201） |
| `qwen3_debugmodel_npu_chunked_loss`（上游默认 ChunkedLossWrapper，TT-4） | 🟢 单卡 step 10 loss 5.10291；FSDP2×2 loss 5.07796（bf16 级差异）——TT-4 在 NIGHTLY 不复现，`npu_baseline` 不再展开 loss |
| `flex_attention` eager（torch_npu master 自带 `patch_flexattention`） | 🟢 op 级前反向（`tests/repro/probe_npu_gaps.py`） |
| 原版 torch_npu master 上的 stock 路径 | 🔴 NPU-1（stock varlen）、NPU-2（fake_backend）、NPU-3（复数索引）、NPU-6（uint64）、NPU-7（stock flex → inductor lowering）、NPU-8（先 `import spmd_types`）——**六项全部在 torch_npu / op-plugin 侧修复**（`patches/`），修复后的结果见下节 |

### 含六个补丁的 torch_npu（第二轮）
| 路径 | 结果 |
|---|---|
| `probe_npu_gaps.py`：NPU-1 / 2 / 3 / 6 | 全部 `[OK ]`；`_flash_attention_forward` PrivateUse1 内核已注册 |
| **stock varlen**（qwen3 `qwen3_debugmodel_stock_varlen`，零 override） | 🟢 step 10 loss 5.10302 / grad_norm 3.3060 |
| **stock llama3**（`ascend_titan.recipes.stock.llama3_debugmodel_stock_npu`：stock VarlenAttention + 复数缓存 ComplexRoPE + ChunkedLossWrapper + spmd_types，零 override） | 🟢 单卡 4.01820 / 1.7382；FSDP2×2 3.97774 / 1.7523 |
| `pp_1f1b`（矩阵，无 shim） | 🟢 |
| `cp`、`fsdp+cp`、`deepseek_v3_fused_mla_swiglu`、stock flex 模型级 | 🔴 DEP-INDUCTOR：Triton-Ascend 未装（NPU-7 修复后 lowering 已通过） |
| `--comm.mode=fake_backend`（NPU-2，最终 wheel） | 🟢 单卡模拟 8 卡干跑 `step: 1  loss: 7.66238` |

## 生效中的 shim
NIGHTLY 上 **0**：`dist_set_timeout`（polyfill）与 `pp_step_presplit_*`（wrap）在 torch 2.15 上探测到原生实现后自动 no-op；文件保留到 RELEASE track 退役。

## 复现
```bash
# 容器内，一次性
python3.12 -m venv /opt/venv-nightly && . /opt/venv-nightly/bin/activate
pip install --pre -c constraints/nightly.txt torch --index-url https://download.pytorch.org/whl/nightly/cpu
source /usr/local/Ascend/cann-9.1.0/set_env.sh && WITH_PATCHES=1 REQUIRE_PR_LINK=0 ./scripts/build_torch_npu.sh
TORCH_NPU_WHEEL=$(ls -t /opt/wheels/torch_npu-*.whl | head -1) WITH_TORCH=1 ./scripts/install.sh
# 验证
ASCEND_TITAN_SKIP_SHIMS=1 ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 ./scripts/check_golden.sh qwen3_debugmodel_npu
ASCEND_TITAN_SKIP_SHIMS=1 ASCEND_RT_VISIBLE_DEVICES=0,1 NPU=2 ./scripts/check_golden.sh qwen3_debugmodel_npu_fsdp2
python tests/repro/probe_npu_gaps.py
```
