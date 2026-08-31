# 基线

2026-08-30 起基线 = **NIGHTLY**（ADR-006）。硬件：Ascend 910B2 ×8，驱动 25.5.1，CANN 9.1.0（镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12-devel`，容器 `ascend-titan-dev`，venv `/opt/venv-nightly`）。

## 版本三元组

| track | torch | torch_npu | torchtitan | 文件 | 状态 |
|---|---|---|---|---|---|
| **NIGHTLY**（默认、唯一门禁） | 2.15.0.dev20260812+cpu（torch_npu master `requirements_2.15.txt` 的 pin） | master `15514cc70` 源码构建（op-plugin `2b02a5aa0`），`ci/build.sh --python=3.12 --torch=2.15.0 --disable_torchair`，256 核 gcc 11.4 **8 分 28 秒** | `13da2d77c` | `constraints/nightly.txt` + `torch_npu.sha` + `torchtitan.sha` | 🟢 |

为什么是 nightly：torchtitan main 与 torch_npu master 都面向 torch nightly。正式版 torch 是唯一被支持之外的东西——不为它写 shim、不为它保留兼容代码。

## NIGHTLY 实测（2026-08-30，全部 `ASCEND_TITAN_SKIP_SHIMS=1`，即**不加任何 shim**）

| 路径 | 结果 |
|---|---|
| `qwen3_debugmodel_npu` 单卡 10 步（seed 42，deterministic） | 🟢 当时与 2.13 golden 逐位一致（step 10 loss 5.10304，grad_norm 3.3061）。**2026-08-30 晚：recipe 删掉 DELTA 4，loss 回到上游 `ChunkedLossWrapper`，四条 golden 全部重录**（见下节） |
| `qwen3_debugmodel_npu_fsdp2` ×2 | 🟢 逐位一致（step 10 loss 5.07792，grad_norm 3.3201） |
| `qwen3_debugmodel_npu_fused` / `_fused_fsdp2`（RMSNorm + SwiGLU + rotary 三个融合算子） | 🟢 2026-08-30 补录 NIGHTLY golden，当时与 2.13 golden 逐位一致；随后随 DELTA 4 的删除一并重录 |

### golden 重录（2026-08-30 晚，删除 DELTA 4 之后）

参考 recipe 的 loss 回到上游默认 `ChunkedLossWrapper`（TT-4 只是正式版 torch 的版本差，P8/P12）。
recipe 定义变了，旧 golden 随之失效，四条曲线在 NIGHTLY 上重录并复跑验证：

| golden | step 10 loss / grad_norm |
|---|---|
| `qwen3_debugmodel_npu` | 5.10291 / 3.3062 |
| `qwen3_debugmodel_npu_fsdp2` | 5.07796 / 3.3200 |
| `qwen3_debugmodel_npu_fused` | 5.09586 / 3.2994 |
| `qwen3_debugmodel_npu_fused_fsdp2` | 5.10656 / 3.3339 |

前两条与上一轮 `qwen3_debugmodel_npu_chunked_loss` 探针的实测值（5.10291 / 5.07796）一致，互为交叉验证。
正式版 torch（RELEASE track）上这条 recipe 会因 TT-4 失败——这正是 P8 说的“只在正式版上出现的问题不算问题”，
因此 2.12 / 2.13 的旧 golden 一并删除，不再维护。
| `qwen3_debugmodel_npu_chunked_loss`（上游默认 ChunkedLossWrapper，TT-4） | 🟢 单卡 step 10 loss 5.10291；FSDP2×2 loss 5.07796——TT-4 在 NIGHTLY 不复现。**该配置已升为参考 recipe 的默认**；探针反转为 `qwen3_debugmodel_npu_ce_loss`（测非 chunked 路径） |
| `flex_attention` eager（torch_npu master 自带 `patch_flexattention`） | 🟢 op 级前反向（`tests/repro/probe_npu_gaps.py`） |
| 原版 torch_npu master 上的 stock 路径 | 🔴 NPU-1（stock varlen）、NPU-2（fake_backend）、NPU-3（复数索引）、NPU-6（uint64）、NPU-7（stock flex → inductor lowering）、NPU-8（先 `import spmd_types`）——**六项全部在 torch_npu / op-plugin 侧修复**（`patches/`），修复后的结果见下节 |

### 含六个补丁的 torch_npu（第二轮）
| 路径 | 结果 |
|---|---|
| `probe_npu_gaps.py`：NPU-1 / 2 / 3 / 6 | 全部 `[OK ]`；`_flash_attention_forward` PrivateUse1 内核已注册 |
| **stock varlen**（qwen3 `qwen3_debugmodel_stock_varlen`，零 override） | 🟢 step 10 loss 5.10302 / grad_norm 3.3060 |
| **stock llama3**（`ascend_titan.models.llama3.llama3_debugmodel_stock_npu`：stock VarlenAttention + 复数缓存 ComplexRoPE + ChunkedLossWrapper + spmd_types，零 override） | 🟢 单卡 4.01820 / 1.7382；FSDP2×2 3.97774 / 1.7523 |
| `pp_1f1b`（矩阵，无 shim） | 🟢 |
| `cp`、`fsdp+cp`、`deepseek_v3_fused_mla_swiglu`、stock flex 模型级 | 🔴 DEP-INDUCTOR：Triton-Ascend 未装（NPU-7 修复后 lowering 已通过） |
| `--comm.mode=fake_backend`（NPU-2，最终 wheel） | 🟢 单卡模拟 8 卡干跑 `step: 1  loss: 7.66238` |

## 生效中的 shim
3 条，都实测承重（关掉就炸）：

- `flex_block_mask_eager_lm` / `_vision` —— torchtitan 无条件编译 `create_block_mask`，
  昇腾没有 inductor 后端时抛 `0 active drivers`；换成上游自己的未编译函数。
  装上 Triton-Ascend 后自动让位。
- `flex_eager_when_deterministic` —— `set_determinism` 在非 ROCm 分支上重新编译
  `_compiled_flex_attn`，这条路在昇腾上不通；上游对 ROCm 的处理就是改用 eager（TT-12）。

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
