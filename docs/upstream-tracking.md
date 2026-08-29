# 上游跟踪

## 固定的 torchtitan
见 `constraints/torchtitan.sha`。升级流程：`.claude/skills/upstream-sync`。

塑造这一策略的事实（2026-08-29 测得）：
- 最新 release v0.2.2（2026-02-20）。override 机制 2026-06-10 落地（#3396）；kimi_k3 2026-08-24 落地（#4025），是*带 FSDP2 的 eager 参考实现*（还没有 TP）。两者都不在任何 release 里 ⇒ 按 SHA 固定。
- main 近 30 天 170 个 commit，其中 42 个动了我们的扩展面（`config/`、`protocols/`、`models/common/`）。
- torchtitan main 面向 PyTorch **nightly**（README），release 也 pin nightly 日期（docs/release.md）。torch_npu 面向 torch 正式版。`scripts/probe_compat.sh`（最远可 import 的 SHA）就是为这个缺口而存在。

## shim ↔ 上游 issue
| shim | 类型 | 目标 | 上游 | 删除条件 |
|---|---|---|---|---|
| `dist_set_timeout` | polyfill | `torch.distributed:set_timeout` | draft：`docs/issues/torchtitan.md#set-timeout`（TT-2） | torch ≥ 2.14 自带 `set_timeout`（自动 no-op）**且** STABLE track 越过 2.13 |
| `pp_step_presplit_single` / `_multi` | wrap | `torch.distributed.pipelining.schedules:PipelineScheduleSingle/Multi` | draft：`docs/issues/torchtitan.md#pp-step`（TT-8） | torch 的 `step` 接受 `arg_mbs`（自动 no-op） |

shim 数量：**2（3 个注册项）**。健康目标：0。

## 上游 ask（我们无法 override 的东西）
| 计算 | 为什么不能 override | 建议的上游改动 | 状态 |
|---|---|---|---|
| kimi_k3 `attn_res` | `_apply_attention_residual` 是 `models/kimi_k3/model.py` 里的自由函数（没有 `Configurable` 节点）；override 意味着替换整个 `KimiK3TransformerBlock.Config` | 抽成带 `sharding_config` 的 `Module`——这也解决上游自己在该函数上的 `TODO: Add TP Support` | 等 kimi_k3 在上游稳定（刚落地几天） |
| kimi_k3 `causal_conv1d` | 在 `InnerKDA.forward` 内被调用；只有 `InnerKDA.Config` 是节点 | 暂时 override `InnerKDA` 可接受 | — |

## 待提交的 issue（草稿在 `docs/issues/`）
| 仓库 | 编号 | 内容 | 状态 |
|---|---|---|---|
| Ascend/pytorch（torch_npu） | NPU-1 | `_flash_attention_forward` 的 NPU 内核 | draft |
| Ascend/pytorch（torch_npu） | NPU-2 | 为 `fake` 进程组后端注册 `npu` | draft |
| Ascend/pytorch（torch_npu） | NPU-3 | 复数张量的高级索引（aclnnIndex 161002） | draft |
| pytorch/pytorch | TORCH-1 | FlexAttention 设备白名单 | draft |
| pytorch/torchtitan | TT-1 | minimal_async_ep 中懒加载 `import triton` | draft |
| pytorch/torchtitan | TT-2 | `set_timeout` 的特性检查 / fallback | draft |
| pytorch/torchtitan | TT-4 | ChunkedLossWrapper 在 NPU 上的 backward | investigating |
| pytorch/torchtitan | TT-5 | spmd_types 下 FSDP 收到普通张量 | investigating |
| pytorch/torchtitan | TT-8 | PP `step(arg_mbs=)` 的特性检查 / fallback | draft |
| pytorch/torchtitan | TT-9 | fused_mla 使用 nightly-only 的 `torch.Tag.inplace` | draft |
