"""Gated DeltaNet (Qwen3.5) on Ascend.

Targets ``torchtitan/models/qwen3_5/gdn.py::GatedDeltaKernel.Config``. Upstream
dispatches to flash-linear-attention's Triton kernels
(``fla.ops.gated_delta_rule``); those compile for CUDA and fail on Ascend even
with Triton-Ascend installed -- ``bishengir-compile`` rejects the generated MLIR.

attn_gym ships a device-agnostic implementation of the same recurrence,
``attn_gym.linear.gdn.chunk_gdn(impl="reference")``, documented as "eager
PyTorch" with FP32 recurrence math. So this override keeps upstream's *math* and
only selects the implementation -- the same shape as the KDA and RoPE overrides.

Two things need replacing, and they sit on top of each other in the config tree,
so a single override claims both (torchtitan rejects an override whose ancestor
another override already claims):

* ``GatedDeltaKernel`` -- the recurrence, above;
* ``InnerGatedDeltaNet`` -- its short convolution. The dense branch is already
  pure torch (``F.conv1d`` + ``silu``), but the packed (``cu_seqlens``) branch
  calls fla's ``causal_conv1d_varlen``, which enters ``torch.cuda`` and dies with
  "PyTorch was compiled without CUDA support". We reuse the depthwise causal
  convolution written for kimi_k3's KDA, which handles both layouts exactly.

Layout: upstream passes ``[B, T, N, K]``; ``chunk_gdn`` takes the SDPA layout
``[B, N, T, K]``. ``use_qk_l2norm_in_kernel=True`` on the fla side means the
kernel L2-normalises Q/K itself, so the override does it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from attn_gym.linear.gdn import chunk_gdn
from torchtitan.config import derive, override
from torchtitan.models.qwen3_5.gdn import GatedDeltaKernel, InnerGatedDeltaNet

from ascend_titan.kernels._probe import require_op  # noqa: F401  (P14: hard dependency)
from ascend_titan.kernels.kda import ascend_causal_conv1d, l2norm

__all__ = ["AscendGatedDeltaKernel", "AscendInnerGatedDeltaNet", "npu_gated_delta_net"]


def _chunk_gdn_btnk(
    q_BTNK: torch.Tensor,
    k_BTNK: torch.Tensor,
    v_BTNV: torch.Tensor,
    g_BTN: torch.Tensor,
    beta_BTN: torch.Tensor,
) -> torch.Tensor:
    """``chunk_gdn`` on upstream's ``[B, T, N, *]`` layout."""
    out = chunk_gdn(
        l2norm(q_BTNK).transpose(1, 2),
        l2norm(k_BTNK).transpose(1, 2),
        v_BTNV.transpose(1, 2),
        g_BTN.transpose(1, 2),
        beta_BTN.transpose(1, 2),
        impl="reference",
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.transpose(1, 2)


class AscendGatedDeltaKernel(GatedDeltaKernel):
    """``GatedDeltaKernel`` on attn_gym's device-agnostic reference recurrence."""

    @dataclass(kw_only=True, slots=True)
    class Config(GatedDeltaKernel.Config):
        pass

    def forward(
        self,
        xq_TNK: torch.Tensor,
        xk_TNK: torch.Tensor,
        xv_TNV: torch.Tensor,
        g_TN: torch.Tensor,
        beta_TN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Grouped linear attention: expand Q/K heads to match V, exactly as upstream
        # does, and for the same reason (repeat_interleave must run on local tensors
        # under TP).
        if xq_TNK.shape[1] != xv_TNV.shape[1]:
            if xv_TNV.shape[1] % xq_TNK.shape[1] != 0:
                raise ValueError(
                    f"value heads {xv_TNV.shape[1]} is not a multiple of "
                    f"query heads {xq_TNK.shape[1]}"
                )
            repeat = xv_TNV.shape[1] // xq_TNK.shape[1]
            xq_TNK = xq_TNK.repeat_interleave(repeat, dim=1)
            xk_TNK = xk_TNK.repeat_interleave(repeat, dim=1)

        if cu_seqlens is None:
            return _chunk_gdn_btnk(
                xq_TNK.unsqueeze(0),
                xk_TNK.unsqueeze(0),
                xv_TNV.unsqueeze(0),
                g_TN.unsqueeze(0),
                beta_TN.unsqueeze(0),
            ).squeeze(0)

        # Packed input: the recurrence must restart at every sequence boundary, so
        # run it per segment. attn_gym's reference has no cu_seqlens argument, and
        # a loop is the definition of what packing means here.
        bounds = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens).tolist()
        pieces = []
        for begin, end in zip(bounds[:-1], bounds[1:], strict=True):
            if end <= begin:
                continue
            pieces.append(
                _chunk_gdn_btnk(
                    xq_TNK[begin:end].unsqueeze(0),
                    xk_TNK[begin:end].unsqueeze(0),
                    xv_TNV[begin:end].unsqueeze(0),
                    g_TN[begin:end].unsqueeze(0),
                    beta_TN[begin:end].unsqueeze(0),
                ).squeeze(0)
            )
        return torch.cat(pieces, dim=0)


class AscendInnerGatedDeltaNet(InnerGatedDeltaNet):
    """``InnerGatedDeltaNet`` with the Ascend depthwise causal convolution."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerGatedDeltaNet.Config):
        pass

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TN: torch.Tensor,
        b_TN: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_N: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        num_tokens = query_TC.shape[0]
        packed = None
        if cu_seqlens_host is not None:
            packed = torch.tensor(cu_seqlens_host, dtype=torch.long, device="cpu")

        def conv(x_TC: torch.Tensor, weight_C1W: torch.Tensor) -> torch.Tensor:
            return ascend_causal_conv1d(
                x_TC.unsqueeze(0),
                weight_C1W[:, 0],
                activation="silu",
                cu_seqlens=packed,
            ).squeeze(0)

        xq_TNK = conv(query_TC, conv_q_weight_C1W).reshape(num_tokens, -1, key_head_dim)
        xk_TNK = conv(key_TC, conv_k_weight_C1W).reshape(num_tokens, -1, key_head_dim)
        xv_TNV = conv(value_TC, conv_v_weight_C1W).reshape(num_tokens, -1, value_head_dim)

        # Same gate math as upstream, kept verbatim so numerics stay comparable.
        g_TN = -torch.exp(A_log_N.float()) * torch.nn.functional.softplus(a_TN.float() + dt_bias_N)
        beta_TN = torch.sigmoid(b_TN)
        return self.kernel(
            xq_TNK,
            xk_TNK,
            xv_TNV,
            g_TN,
            beta_TN,
            cu_seqlens=cu_seqlens if cu_seqlens_host is not None else None,
            cu_seqlens_cpu=packed,
        )


@override(
    target=InnerGatedDeltaNet.Config,
    exact=True,
    description="Gated DeltaNet on Ascend: torch depthwise causal conv1d + attn_gym's reference "
    "recurrence (fla's kernels are CUDA-only)",
)
def npu_gated_delta_net(cfg: InnerGatedDeltaNet.Config) -> AscendInnerGatedDeltaNet.Config:
    """One override for the whole subtree; see kernels/kda.py for why it must be one."""
    new = derive(cfg, AscendInnerGatedDeltaNet.Config)
    new.kernel = derive(cfg.kernel, AscendGatedDeltaKernel.Config)
    return new
