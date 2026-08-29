# 基线

建立于 2026-08-29（M0 + M1）。硬件：Ascend 910B2 ×8，驱动 25.5.1，CANN 9.1.0
（镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12-devel`）。

## 版本元组

| track | torch | torch_npu | torchtitan | 文件 | 状态 |
|---|---|---|---|---|---|
| **NEXT**（默认） | 2.13.0（+cpu wheel） | 2.13.0rc1 | `13da2d77c`（main，2026-08-29） | `constraints/npu.txt` | 🟢 M1 |
| STABLE | 2.12.0（+cpu wheel） | 2.12.0 | `13da2d77c` | `constraints/npu-stable.txt` | 🟢 M1 |

两条 track 在 `--debug.seed 42 --debug.deterministic` 下 M1 recipe 的 loss/grad_norm 曲线**逐位一致**（`tests/assets/losses/npu/`）。torch 从 CPU index 安装；torch_npu 提供设备后端，并由 `import torch` 自动加载（`ascend-titan-doctor` 中 `torch_npu autoload True`）。

为什么不用 torch nightly：torch_npu 跟随 torch 正式版（最新预发布 2.13.0rc1）；torch nightly 是 2.15.0.dev。差距靠**两条 shim** 弥合（`torch.distributed.set_timeout` polyfill、PP `step(arg_mbs=)` 转发），torchtitan main 需要的其它东西 2.13.0 都有——见 `docs/issues/torchtitan.md`。

## M1 跑的是什么
`qwen3_debugmodel_npu` = 上游 `qwen3_debugmodel` + 4 个增量：
1. inner attention 用 `varlen` 节点 + override `ascend_titan.kernels.attention.npu_fusion_attention`（stock flex/varlen 在 NPU 上跑不了：TORCH-1、NPU-1）；
2. `parallelism.spmd_backend = partial_dtensor`（TT-5）；
3. 关闭 checkpoint（DCP 是单独的矩阵格）；
4. `CrossEntropyLoss` 替代 `ChunkedLossWrapper`（TT-4）。

| 路径 | 卡数 | 结果（seed 42，deterministic） |
|---|---|---|
| 单卡 eager | 1 | step 1 loss 7.6546 → step 10 loss 5.10304 grad_norm 3.3061，约 55k tps，2.4 GiB |
| FSDP2 ×2 | 2 | step 10 loss 5.07792 grad_norm 3.3201，约 51k tps，2.2 GiB |
| fake_backend | 1 | 🔴 NPU-2（fake 进程组没有 npu） |

## 生效中的 shim
| shim | 类型 | 原因 | 何时删除 |
|---|---|---|---|
| `dist_set_timeout` | polyfill | torchtitan 在第 1 步后调用 nightly-only 的 `torch.distributed.set_timeout` | torch 自带后自动 no-op；或 torchtitan 加 fallback（TT-2） |
| `pp_step_presplit_*` | wrap | torchtitan 用 nightly-only 的 `step(arg_mbs=...)` 关键字传预切分的 microbatch | torch 的 `step` 接受 `arg_mbs` 后自动 no-op；或 torchtitan 加 fallback（TT-8） |

注意：torch ≤ 2.13 上 `_set_pg_timeout` 只处理 nccl/gloo，并警告 `Set timeout is now only supported for either nccl or gloo`；HCCL 组的超时在第 1 步后**没有**被缩短（与 CUDA 的行为差异，记为 TORCH-3）。

## 复现
```bash
WITH_TORCH=1 ./scripts/install.sh                 # NEXT track
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 ./scripts/check_golden.sh qwen3_debugmodel_npu
ASCEND_RT_VISIBLE_DEVICES=0,1 NPU=2 ./scripts/check_golden.sh qwen3_debugmodel_npu_fsdp2
```
