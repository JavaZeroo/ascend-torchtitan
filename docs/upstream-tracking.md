# Upstream tracking

## Pinned torchtitan
See `constraints/torchtitan.sha`. Bump procedure: `.claude/skills/upstream-sync`.

Facts that shape the policy (measured 2026-08-29):
- Last release v0.2.2 (2026-02-20). Override mechanism landed 2026-06-10 (#3396); kimi_k3 landed 2026-08-24 (#4025) as an *eager reference with FSDP2* (no TP yet). Neither is in any release ⇒ pin SHA.
- 170 commits / 30 days on main; 42 touch our extension surfaces (`config/`, `protocols/`, `models/common/`).
- torchtitan main targets PyTorch **nightly** (README) and releases pin nightly dates (docs/release.md). torch_npu targets torch releases. The "furthest importable SHA" probe (`scripts/probe_compat.sh`) exists because of this gap.

## Shims ↔ upstream issues
| shim | kind | target | upstream | delete when |
|---|---|---|---|---|
| `dist_set_timeout` | polyfill | `torch.distributed:set_timeout` | draft: `docs/issues/torchtitan.md#set-timeout` (TT-2) | torch ≥ 2.14 ships `set_timeout` (auto no-op) **and** STABLE track moves past 2.13 |

Shim count: **1**. Health target: 0.

## Upstream asks (things we cannot override)
| computation | why not overridable | proposed upstream change | status |
|---|---|---|---|
| kimi_k3 `attn_res` | `_apply_attention_residual` is a free function in `models/kimi_k3/model.py` (no `Configurable` node); overriding would mean replacing the whole `KimiK3TransformerBlock.Config` | extract into a `Module` with a `sharding_config` — this also resolves upstream's own `TODO: Add TP Support` on that function | wait until kimi_k3 stabilises upstream (it is days old) |
| kimi_k3 `causal_conv1d` | called inside `InnerKDA.forward`; only `InnerKDA.Config` is a node | acceptable to override `InnerKDA` for now | — |

## Issues to file (drafts in `docs/issues/`)
| repo | id | about | status |
|---|---|---|---|
| Ascend/pytorch (torch_npu) | NPU-1 | `_flash_attention_forward` NPU kernel | draft |
| Ascend/pytorch (torch_npu) | NPU-2 | register `npu` for the `fake` PG backend | draft |
| pytorch/pytorch | TORCH-1 | FlexAttention device whitelist | draft |
| pytorch/torchtitan | TT-1 | lazy `import triton` in minimal_async_ep | draft |
| pytorch/torchtitan | TT-2 | `set_timeout` feature check / fallback | draft |
| pytorch/torchtitan | TT-4 | ChunkedLossWrapper backward on NPU | investigating |
| pytorch/torchtitan | TT-5 | spmd_types + FSDP plain params on NPU | investigating |
