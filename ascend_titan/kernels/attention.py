"""Ascend fused attention as a torchtitan inner-attention override.

Targets ``torchtitan/models/common/attention.py::VarlenAttention.Config``. The
stock ``VarlenAttention`` calls ``torch.nn.attention.varlen.varlen_attn`` which
dispatches to ``aten::_flash_attention_forward``; torch_npu provides no NPU
kernel for that op (capability matrix: attention/varlen = NPU-1). ``FlexAttention``
is rejected by torch itself on non-CUDA/CPU/HPU devices (TORCH-1). So this
override is the only way to run an upstream language model on Ascend at all.

Kernel: ``torch_npu.npu_fusion_attention`` in ``TND`` layout with document
boundaries passed as ``actual_seq_qlen``/``actual_seq_kvlen`` (cumulative end
offsets, host ints) and ``sparse_mode=3`` (per-document causal via the standard
2048x2048 upper-triangular mask). GQA (fewer K/V heads) is handled natively.

``torch.compile``: the kernel wants *host* offsets, and a ``.tolist()`` inside a
compiled region becomes unbacked SymInts that break tracing (OURS-8). Forward
and backward are therefore wrapped as ``torch.library.custom_op`` taking the
``cu_seq`` **tensors**; the D2H happens inside the op (opaque to dynamo) and the
``register_fake`` implementations only describe shapes.

Contract kept identical to the stock module: packed ``(T, N, H)`` inputs, a
``VarlenMetadata`` mask, ``window_size`` semantics, output in the query dtype.
Import is safe without torch_npu: the override is simply not registered and a
WARNING is logged (ADR-004).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import torch_npu

    _AVAILABLE = hasattr(torch_npu, "npu_fusion_attention")
    if not _AVAILABLE:
        logger.warning(
            "[ascend_titan] torch_npu has no npu_fusion_attention; "
            "VarlenAttention stays on the upstream (CUDA-only) kernel"
        )
except ImportError as e:
    _AVAILABLE = False
    logger.warning(
        "[ascend_titan] torch_npu unavailable (%s); VarlenAttention stays on the "
        "upstream (CUDA-only) kernel",
        e,
    )

if _AVAILABLE:
    import torch
    from torchtitan.config import derive, override
    from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata
    from torchtitan.protocols.module import Module

    _INT_MAX = 2147483647
    # sparse_mode=3 expects the fixed 2048x2048 upper-triangular "masked" pattern;
    # the kernel tiles it over the real sequence. One per device, built lazily.
    _CAUSAL_MASK_SIZE = 2048
    _causal_masks: dict[torch.device, torch.Tensor] = {}
    # softmax statistics from npu_fusion_attention in TND layout: (T, N, 8) fp32
    _SOFTMAX_STAT_WIDTH = 8

    def _causal_mask(device: torch.device) -> torch.Tensor:
        m = _causal_masks.get(device)
        if m is None:
            m = torch.triu(
                torch.ones(_CAUSAL_MASK_SIZE, _CAUSAL_MASK_SIZE, dtype=torch.bool, device=device),
                diagonal=1,
            )
            _causal_masks[device] = m
        return m

    def _offsets(cu: torch.Tensor) -> list[int]:
        """Cumulative end offsets as host ints (what the kernel wants)."""
        return [int(x) for x in cu[1:].tolist()]

    @torch.library.custom_op("ascend_titan::fusion_attention_varlen", mutates_args=())
    def _fa_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        scale: float,
        sparse_mode: int,
        pre_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, smax, ssum, *_ = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            q.shape[1],
            "TND",
            atten_mask=_causal_mask(q.device),
            scale=scale,
            pre_tockens=pre_tokens,
            next_tockens=0,
            actual_seq_qlen=_offsets(cu_q),
            actual_seq_kvlen=_offsets(cu_k),
            sparse_mode=sparse_mode,
        )
        return out, smax, ssum

    @_fa_fwd.register_fake
    def _(q, k, v, cu_q, cu_k, scale, sparse_mode, pre_tokens):
        t, n, _ = q.shape
        stat = q.new_empty((t, n, _SOFTMAX_STAT_WIDTH), dtype=torch.float32)
        return torch.empty_like(q), stat, stat.clone()

    @torch.library.custom_op("ascend_titan::fusion_attention_varlen_bwd", mutates_args=())
    def _fa_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        smax: torch.Tensor,
        ssum: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        scale: float,
        sparse_mode: int,
        pre_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad(
            q,
            k,
            v,
            dout,
            q.shape[1],
            "TND",
            atten_mask=_causal_mask(q.device),
            softmax_max=smax,
            softmax_sum=ssum,
            attention_in=out,
            scale_value=scale,
            pre_tockens=pre_tokens,
            next_tockens=0,
            actual_seq_qlen=_offsets(cu_q),
            actual_seq_kvlen=_offsets(cu_k),
            sparse_mode=sparse_mode,
        )
        return dq, dk, dv

    @_fa_bwd.register_fake
    def _(q, k, v, out, dout, smax, ssum, cu_q, cu_k, scale, sparse_mode, pre_tokens):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _setup_context(ctx, inputs, output):
        q, k, v, cu_q, cu_k, scale, sparse_mode, pre_tokens = inputs
        out, smax, ssum = output
        ctx.save_for_backward(q, k, v, out, smax, ssum, cu_q, cu_k)
        ctx.scale, ctx.sparse_mode, ctx.pre_tokens = scale, sparse_mode, pre_tokens

    def _backward(ctx, dout, _dsmax, _dssum):
        q, k, v, out, smax, ssum, cu_q, cu_k = ctx.saved_tensors
        dq, dk, dv = _fa_bwd(
            q, k, v, out, dout.contiguous(), smax, ssum, cu_q, cu_k,
            ctx.scale, ctx.sparse_mode, ctx.pre_tokens,
        )  # fmt: skip
        return dq, dk, dv, None, None, None, None, None

    _fa_fwd.register_autograd(_backward, setup_context=_setup_context)

    def fusion_attention_varlen(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        *,
        scale: float,
        sparse_mode: int = 3,
        pre_tokens: int = _INT_MAX,
    ) -> torch.Tensor:
        """Packed varlen causal attention on NPU (public functional entry)."""
        return _fa_fwd(q, k, v, cu_q, cu_k, scale, sparse_mode, pre_tokens)[0]

    class AscendFusionAttention(VarlenAttention):
        """``VarlenAttention`` backed by ``torch_npu.npu_fusion_attention``."""

        @dataclass(kw_only=True, slots=True)
        class Config(VarlenAttention.Config):
            pass

        def __init__(self, config: Config) -> None:
            # Skip VarlenAttention.__init__: it selects a CUDA flash-attention impl.
            Module.__init__(self)
            self.window_size = tuple(config.window_size)
            left, right = self.window_size
            if right != 0 or left < -1 or left == 0:
                raise NotImplementedError(
                    f"AscendFusionAttention supports window_size (-1, 0) [causal] and "
                    f"(W, 0) [sliding causal]; got {self.window_size}"
                )

        def forward(
            self,
            q_TNH: torch.Tensor,
            k_TNH: torch.Tensor,
            v_TNH: torch.Tensor,
            *,
            attention_masks: VarlenMetadata,
            scale: float | None = None,
            out_transform=None,
            **kwargs,
        ) -> torch.Tensor:
            assert isinstance(attention_masks, VarlenMetadata), (
                f"attention_masks must be VarlenMetadata, got {type(attention_masks)}"
            )
            if out_transform is not None:
                # LSE epilogue (context parallel / attention sinks). npu_fusion_attention
                # returns softmax_max/softmax_sum instead of LSE; wiring that up is M4.
                raise NotImplementedError("AscendFusionAttention: out_transform (LSE) unsupported")
            head_dim = q_TNH.shape[-1]
            if scale is None:
                scale = head_dim**-0.5
            left, _ = self.window_size
            # sparse_mode 3: causal, right-aligned; 4: band (pre/next tokens) with mask.
            sparse_mode = 3 if left == -1 else 4
            pre_tokens = _INT_MAX if left == -1 else left
            out = fusion_attention_varlen(
                q_TNH.to(torch.bfloat16),
                k_TNH.to(torch.bfloat16),
                v_TNH.to(torch.bfloat16),
                attention_masks.cu_seq_q,
                attention_masks.cu_seq_k,
                scale=scale,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
            )
            return out.to(q_TNH.dtype)

    @override(
        target=VarlenAttention.Config,
        description="Ascend npu_fusion_attention (TND, varlen, GQA) for VarlenAttention",
    )
    def npu_fusion_attention(cfg: VarlenAttention.Config) -> AscendFusionAttention.Config:
        return derive(cfg, AscendFusionAttention.Config)
