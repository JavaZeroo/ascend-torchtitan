"""TT-12: flex attention + 读张量的 mask_mod + 确定性算法 = SubgraphLoweringException

Run on the NPU box: python tests/repro/probe_flex_deterministic.py

结论（2026-08-31 实测，910B2 / torch 2.15.0.dev20260812 / torch_npu master 15514cc70）:
读张量的 mask_mod（document mask）在昇腾上**能跑**——前向、LSE、反向都通过。
只有开了 ``torch.use_deterministic_algorithms(True)`` 才会炸。

机制在 torch 自己的 inductor 里，与设备无关。``torch/_inductor/lowering.py`` 的 ``expand``::

    # In deterministic modes, preserve the old materialization boundary
    # since fusing through expanded inputs can change reduction numerics.
    x.mark_reuse(..., graph_reuse=config.deterministic
                      or torch.are_deterministic_algorithms_enabled())

确定性模式强制物化这个中间结果，而物化就是建 buffer；flex 的 mask_mod 必须整体降解成
pointwise 子图，于是 ``subgraph_lowering.py`` 抛::

    SubgraphLoweringException: Buffers cannot be created while lowering a pointwise subgraph.

影响面：训练不受影响；受影响的是**确定性 golden**——``scripts/check_golden.sh`` 加
``--debug.deterministic``，所以掩码读张量的模型录不了逐位 golden。

归因 TORCH（不是 NPU、不是硬件）。按 P10，github.com/pytorch 只读：只记录，不提上游。
"""

import torch
import torch_npu  # noqa: F401
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

DEV = "npu:0"
B, H, T, D = 1, 4, 512, 64
SEGMENTS = 4


def _mask_mods(device):
    segment_ids = torch.repeat_interleave(
        torch.arange(SEGMENTS, device=device, dtype=torch.int32), T // SEGMENTS
    )

    def causal(b, h, q_idx, kv_idx):
        """纯索引算术，不读张量。"""
        return q_idx >= kv_idx

    def document(b, h, q_idx, kv_idx):
        """读张量——这是 torchtitan 视觉塔 create_block_diagonal_mask 的形状。"""
        return segment_ids[q_idx] == segment_ids[kv_idx]

    return {"causal": causal, "document": document}


def run_once(mask_mod, *, backward):
    q, k, v = (
        torch.randn(B, H, T, D, device=DEV, dtype=torch.bfloat16, requires_grad=backward)
        for _ in range(3)
    )
    block_mask = create_block_mask(mask_mod, B, H, T, T, device=DEV)
    out = flex_attention(q, k, v, block_mask=block_mask)
    if backward:
        out.float().sum().backward()
    return float(out.abs().max())


def run_via_torchtitan_deterministic_path(mask_mod):
    """复刻 torchtitan set_determinism 在非 ROCm 上做的事。

    它把 ``FlexAttention._compiled_flex_attn`` **重新赋值**成
    ``torch.compile(flex_attention, options=FlexAttention.inductor_configs)``——
    覆盖掉我们在 setup() 里装的 eager shim，并且编译的是整个 flex_attention 函数，
    而不是它内部那个 HOP 包装器。图更大，mask_mod 子图就在这个上下文里撞上
    确定性模式的物化边界。ROCm 有一条分支走 eager，昇腾没有。
    """
    from torchtitan.models.common.attention import FlexAttention

    configs = dict(FlexAttention.inductor_configs)
    configs["max_autotune"] = False
    configs["coordinate_descent_tuning"] = False
    compiled = torch.compile(flex_attention, options=configs)

    q, k, v = (torch.randn(B, H, T, D, device=DEV, dtype=torch.bfloat16) for _ in range(3))
    block_mask = create_block_mask(mask_mod, B, H, T, T, device=DEV)
    out = compiled(q, k, v, block_mask=block_mask)
    return float(out.abs().max())


def main() -> int:
    failures = 0
    for deterministic in (False, True):
        torch.use_deterministic_algorithms(deterministic, warn_only=False)
        label = "deterministic=ON " if deterministic else "deterministic=OFF"
        for name, mask_mod in _mask_mods(DEV).items():
            for backward in (False, True):
                what = f"{label}  {name:8} {'fwd+bwd' if backward else 'fwd    '}"
                try:
                    peak = run_once(mask_mod, backward=backward)
                    print(f"[OK ] {what}  |out|max={peak:.4e}", flush=True)
                except Exception as exc:  # noqa: BLE001 - the failure is the result
                    kind = (
                        "SubgraphLoweringException"
                        if "SubgraphLowering" in str(exc)
                        else type(exc).__name__
                    )
                    print(f"[ERR] {what}  {kind}", flush=True)
                    failures += 1
    # 关键的一格：走 torchtitan 在确定性模式下真正会走的那条路
    torch.use_deterministic_algorithms(True)
    for name, mask_mod in _mask_mods(DEV).items():
        what = f"torchtitan 确定性路径  {name:8} fwd    "
        try:
            peak = run_via_torchtitan_deterministic_path(mask_mod)
            print(f"[OK ] {what}  |out|max={peak:.4e}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the failure is the result
            kind = (
                "SubgraphLoweringException"
                if "SubgraphLowering" in str(exc)
                else type(exc).__name__
            )
            print(f"[ERR] {what}  {kind}", flush=True)
            failures += 1
    torch.use_deterministic_algorithms(False)
    print(
        "\n预期：直接调用 flex_attention 的六格全 OK（含 deterministic=ON）；"
        "只有最后 torchtitan 确定性路径的 document 那格报 SubgraphLoweringException。"
    )
    return 0 if failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
