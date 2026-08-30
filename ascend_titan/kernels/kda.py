"""Kimi Delta Attention (KDA) on Ascend.

Targets ``torchtitan/models/kimi_k3/kda.py``: ``KDAKernel.Config`` (the linear-
attention kernel) and ``InnerKDA.Config`` (short convolution + kernel). Upstream
is CUDA-only by construction on three separate counts:

* ``KDAKernel.forward`` raises unless the tensors are CUDA *and* the device is
  Blackwell SM100/SM103;
* ``l2norm`` comes from attn_gym's Triton implementation;
* ``causal_conv1d`` is attn_gym's CuTeDSL (cutlass) kernel.

attn_gym itself offers a device-agnostic path for the first two: ``bound_gate``
and ``chunk_kda`` take ``impl="reference"``, documented as "differentiable eager
PyTorch in FP32". So this override keeps upstream's *math* and only selects the
implementation -- the same shape as the RoPE override, not a workaround.

``causal_conv1d`` has no reference implementation upstream, so the depthwise
causal convolution is written here: ``W`` shifted reads plus a boundary mask,
which is exact for the dense ``[B, T, C]`` layout and for packed sequences
delimited by ``cu_seqlens`` (a tap that would reach across a sequence start
reads zero, matching "each sequence is convolved independently").

Numerics: FP32 accumulation, cast back to the input dtype, matching the
documented contract of the fused kernel. Alignment tests live in
tests/unit/test_kernel_kda.py (CPU) and tests/npu/test_kernel_kda.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from attn_gym.linear.kda import bound_gate, chunk_kda
from torchtitan.config import derive, override
from torchtitan.models.kimi_k3.kda import InnerKDA, KDAKernel

from ascend_titan.kernels._probe import torch_npu  # noqa: F401  (P14: hard dependency)

__all__ = [
    "AscendInnerKDA",
    "AscendKDAKernel",
    "ascend_causal_conv1d",
    "npu_kda",
]


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise the last dim in FP32 (attn_gym's Triton ``l2norm``, in torch)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


def _segment_starts(cu_seqlens: torch.Tensor | None, tokens: int, device) -> torch.Tensor:
    """Per-token index of the first token of its sequence."""
    if cu_seqlens is None:
        return torch.zeros(tokens, dtype=torch.long, device=device)
    bounds = cu_seqlens.to(device=device, dtype=torch.long)
    starts = torch.zeros(tokens, dtype=torch.long, device=device)
    lengths = bounds[1:] - bounds[:-1]
    for begin, length in zip(bounds[:-1].tolist(), lengths.tolist(), strict=True):
        if length > 0:
            starts[begin : begin + length] = begin
    return starts


def ascend_causal_conv1d(
    x_BTC: torch.Tensor,
    weight_CW: torch.Tensor,
    *,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise causal conv1d over ``[B, T, C]`` with weights ``[C, W]``.

    Replaces ``attn_gym.linear.kda.short_conv.causal_conv1d`` (CuTeDSL). Taps that
    would reach before the start of the current sequence read zero, so packed
    inputs behave exactly like independently convolved sequences.
    """
    if x_BTC.dim() != 3:
        raise ValueError(f"expected [B, T, C], got {tuple(x_BTC.shape)}")
    width = weight_CW.shape[1]
    tokens = x_BTC.shape[1]
    device = x_BTC.device

    positions = torch.arange(tokens, device=device)
    starts = _segment_starts(cu_seqlens, tokens, device)

    out = torch.zeros(x_BTC.shape, dtype=torch.float32, device=device)
    xf = x_BTC.float()
    wf = weight_CW.float()
    for back in range(width):
        # tap `back` steps into the past; weight column W-1-back is "now - back"
        src = positions - back
        valid = src >= starts
        gathered = xf.index_select(1, src.clamp_min(0))
        out = out + torch.where(valid[None, :, None], gathered, 0.0) * wf[:, width - 1 - back]

    if activation == "silu":
        out = F.silu(out)
    elif activation is not None:
        raise NotImplementedError(f"activation {activation!r} is not implemented on NPU")
    return out.to(x_BTC.dtype)


class AscendKDAKernel(KDAKernel):
    """``KDAKernel`` on attn_gym's device-agnostic reference path."""

    @dataclass(kw_only=True, slots=True)
    class Config(KDAKernel.Config):
        pass

    def forward(
        self,
        q_BTNK: torch.Tensor,
        k_BTNK: torch.Tensor,
        v_BTNV: torch.Tensor,
        raw_gate_BTNK: torch.Tensor,
        raw_beta_BTN: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_NK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_BTNK = bound_gate(
            raw_gate_BTNK,
            A_log_N.float(),
            dt_bias_NK.float(),
            lower_bound=self.lower_bound,
            impl="reference",
        )
        output_BTNV, _ = chunk_kda(
            l2norm(q_BTNK),
            l2norm(k_BTNK),
            v_BTNV,
            gate_BTNK,
            raw_beta_BTN.float().sigmoid(),
            cu_seqlens=cu_seqlens,
            impl="reference",
        )
        return output_BTNV


class AscendInnerKDA(InnerKDA):
    """``InnerKDA`` with the Ascend short convolution."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerKDA.Config):
        pass

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_TNK: torch.Tensor,
        raw_beta_TN: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_NK: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        mixed_qkv_BTC = torch.cat((query_TC, key_TC, value_TC), dim=-1).unsqueeze(0)
        conv_weight_C1W = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W), dim=0
        )
        conv_output_BTC = ascend_causal_conv1d(
            mixed_qkv_BTC,
            conv_weight_C1W[:, 0],
            activation="silu",
            cu_seqlens=cu_seqlens,
        )
        q_BTC, k_BTC, v_BTC = conv_output_BTC.chunk(3, dim=-1)
        q_BTNK, k_BTNK, v_BTNV = (
            tensor.unflatten(-1, (-1, self.head_dim)) for tensor in (q_BTC, k_BTC, v_BTC)
        )
        output_BTNV = self.kernel(
            q_BTNK,
            k_BTNK,
            v_BTNV,
            raw_gate_TNK.unsqueeze(0),
            raw_beta_TN.unsqueeze(0),
            A_log_N,
            dt_bias_NK,
            cu_seqlens=cu_seqlens,
        )
        return output_BTNV.squeeze(0)


@override(
    target=InnerKDA.Config,
    exact=True,
    description="KDA on Ascend: torch depthwise causal conv1d + attn_gym's reference kernel",
)
def npu_kda(cfg: InnerKDA.Config) -> AscendInnerKDA.Config:
    """Swap the whole KDA subtree in one override.

    ``InnerKDA.Config`` owns ``kernel: KDAKernel.Config``, and torchtitan rejects
    an override that claims a node whose ancestor another override already claims
    (nested overrides are order-dependent). So the nested kernel config is derived
    here rather than by a second ``@override``.
    """
    new = derive(cfg, AscendInnerKDA.Config)
    new.kernel = derive(cfg.kernel, AscendKDAKernel.Config)
    return new
