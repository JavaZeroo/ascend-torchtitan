"""Ascend fused attention as a torchtitan inner-attention override.

Targets ``torchtitan/models/common/attention.py::VarlenAttention.Config``. The
stock ``VarlenAttention`` calls ``torch.nn.attention.varlen.varlen_attn`` which
dispatches to ``aten::_flash_attention_forward``; torch_npu provides no NPU
kernel for that op (capability matrix: attention/varlen = NPU). ``FlexAttention``
is rejected by torch itself on non-CUDA/CPU/HPU devices. So this override is the
only way to run an upstream language model on Ascend at all, not a speed-up.

Kernel: ``torch_npu.npu_fusion_attention`` in ``TND`` layout with document
boundaries passed as ``actual_seq_qlen``/``actual_seq_kvlen`` (cumulative end
offsets, host ints) and ``sparse_mode=3`` (per-document causal via the standard
2048x2048 upper-triangular mask). GQA (fewer K/V heads) is handled natively.

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

    def _causal_mask(device: torch.device) -> torch.Tensor:
        m = _causal_masks.get(device)
        if m is None:
            m = torch.triu(
                torch.ones(_CAUSAL_MASK_SIZE, _CAUSAL_MASK_SIZE, dtype=torch.bool, device=device),
                diagonal=1,
            )
            _causal_masks[device] = m
        return m

    def _host_offsets(cu_seq: torch.Tensor, cached: tuple[int, ...] | None) -> list[int]:
        """Cumulative end offsets as host ints (what the kernel wants), skipping
        the leading 0. Uses the metadata's host copy when present; otherwise one
        D2H sync per call. TODO(perf): request include_host_offsets upstream."""
        offs = cached if cached is not None else tuple(int(x) for x in cu_seq.tolist())
        return list(offs[1:])

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

            actual_q = _host_offsets(attention_masks.cu_seq_q, attention_masks.cu_seq_q_host)
            actual_k = _host_offsets(attention_masks.cu_seq_k, None)
            head_dim = q_TNH.shape[-1]
            if scale is None:
                scale = head_dim**-0.5

            left, _ = self.window_size
            # sparse_mode 3: causal, right-aligned; 4: band (pre/next tokens) with mask.
            sparse_mode = 3 if left == -1 else 4
            pre_tokens = _INT_MAX if left == -1 else left

            out = torch_npu.npu_fusion_attention(
                q_TNH.to(torch.bfloat16),
                k_TNH.to(torch.bfloat16),
                v_TNH.to(torch.bfloat16),
                q_TNH.shape[1],
                "TND",
                atten_mask=_causal_mask(q_TNH.device),
                scale=scale,
                pre_tockens=pre_tokens,
                next_tockens=0,
                actual_seq_qlen=actual_q,
                actual_seq_kvlen=actual_k,
                sparse_mode=sparse_mode,
            )[0]
            return out.to(q_TNH.dtype)

    @override(
        target=VarlenAttention.Config,
        description="Ascend npu_fusion_attention (TND, varlen, GQA) for VarlenAttention",
    )
    def npu_fusion_attention(cfg: VarlenAttention.Config) -> AscendFusionAttention.Config:
        return derive(cfg, AscendFusionAttention.Config)
