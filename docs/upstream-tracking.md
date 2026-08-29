# 上游跟踪

## 固定的 torchtitan
见 `constraints/torchtitan.sha`。升级流程：`.claude/skills/upstream-sync`。

塑造这一策略的事实（2026-08-29 测得）：
- 最新 release v0.2.2（2026-02-20）。override 机制 2026-06-10 落地（#3396）；kimi_k3 2026-08-24 落地（#4025），是*带 FSDP2 的 eager 参考实现*（还没有 TP）。两者都不在任何 release 里 ⇒ 按 SHA 固定。
- main 近 30 天 170 个 commit，其中 42 个动了我们的扩展面（`config/`、`protocols/`、`models/common/`）。
- torchtitan main 面向 PyTorch **nightly**（README），release 也 pin nightly 日期（docs/release.md）。**torch_npu master 同样面向 nightly**（`requirements_2.15.txt` 钉 `2.15.0.dev20260812+cpu`；2026-08-30 实测，见 ADR-006）——此前『torch_npu 面向正式版』的说法是错的。`scripts/probe_compat.sh`（最远可 import 的 SHA）用于 torchtitan 升级前探测。

## shim ↔ 上游 issue
| shim | 类型 | 目标 | 上游 | 删除条件 |
|---|---|---|---|---|
| `dist_set_timeout` | polyfill | `torch.distributed:set_timeout` | 版本差（TT-2 / TORCH-3），NIGHTLY 上不存在 | **NIGHTLY 上已自动 no-op**（2026-08-30 验证）；RELEASE track 退役后删除文件 |
| `pp_step_presplit_single` / `_multi` | wrap | `torch.distributed.pipelining.schedules:PipelineScheduleSingle/Multi` | 版本差（TT-8 / TORCH-4），NIGHTLY 上不存在 | 同上 |

shim 数量：**2 个文件，NIGHTLY 上生效 0**。健康目标：0（P8 之后新的 shim 只能来自 NIGHTLY 上可复现的 TT/TORCH 问题）。

## 上游 ask（我们无法 override 的东西）
| 计算 | 为什么不能 override | 建议的上游改动 | 状态 |
|---|---|---|---|
| kimi_k3 `attn_res` | `_apply_attention_residual` 是 `models/kimi_k3/model.py` 里的自由函数（没有 `Configurable` 节点）；override 意味着替换整个 `KimiK3TransformerBlock.Config` | 抽成带 `sharding_config` 的 `Module`——这也解决上游自己在该函数上的 `TODO: Add TP Support` | 等 kimi_k3 在上游稳定（刚落地几天） |
| kimi_k3 `causal_conv1d` | 在 `InnerKDA.forward` 内被调用；只有 `InnerKDA.Config` 是节点 | 暂时 override `InnerKDA` 可接受 | — |

## 上游 issue / PR（P10：只提 gitcode.com/Ascend/*）
| 仓库 | 编号 | 内容 | 状态 |
|---|---|---|---|
| Ascend/pytorch（torch_npu） | NPU-1 | `_flash_attention_forward/_backward` 的 PrivateUse1 内核 | 已提交：issue [4439](https://gitcode.com/Ascend/pytorch/issues/4439) · PR [!45527](https://gitcode.com/Ascend/pytorch/merge_requests/45527) |
| Ascend/pytorch（torch_npu） | NPU-2 | 为 `fake` 进程组后端注册 `npu` | 已提交：issue [4438](https://gitcode.com/Ascend/pytorch/issues/4438) · PR [!45526](https://gitcode.com/Ascend/pytorch/merge_requests/45526) |
| Ascend/pytorch（torch_npu） | NPU-7 | inductor `make_reduction` 覆盖缺 `strict_reduction`（torch 2.15） | 已提交：issue [4440](https://gitcode.com/Ascend/pytorch/issues/4440) · PR [!45528](https://gitcode.com/Ascend/pytorch/merge_requests/45528) |
| Ascend/pytorch（torch_npu） | NPU-8 | 自动加载经 `torch.distributed._tensor` 拖入 checkpoint/fsdp（spmd_types 循环导入） | 已提交：issue [4441](https://gitcode.com/Ascend/pytorch/issues/4441) · PR [!45529](https://gitcode.com/Ascend/pytorch/merge_requests/45529) |
| Ascend/op-plugin | NPU-3 | `index.Tensor` 支持复数（经实数视图） | 已提交：issue [466](https://gitcode.com/Ascend/op-plugin/issues/466) · PR [!5800](https://gitcode.com/Ascend/op-plugin/merge_requests/5800) |
| Ascend/op-plugin | NPU-6 | `zero_` 支持 uint16/32/64 | 已提交：issue [467](https://gitcode.com/Ascend/op-plugin/issues/467) · PR [!5801](https://gitcode.com/Ascend/op-plugin/merge_requests/5801) |
| pytorch/pytorch、pytorch/torchtitan | TORCH-1/7/8、TT-1/4/5/11 | **不提**（P10）；NIGHTLY 上仍存在的记录在 `docs/issues/`，修复方案在 `patches/evidence/` | — |
