"""Gated DeltaNet (Qwen3.5) chunk recurrence fused on fla-npu AscendC kernels.

This is the R5 perf path: the same chunk-parallel gated delta rule that
:func:`ascend_titan.kernels.gdn.ascend_chunk_gdn` runs in plain torch, but with
the per-op AscendC kernels from flash-linear-attention-npu fused into ONE
differentiable custom op (forward over 6 kernels, backward over 7). Activation
checkpointing and torch dispatch stop paying for every intermediate transpose
and matmul of the recurrence.

Callback contract (identical math, different implementation):

* Targets the same two config nodes as kernels/gdn.py -- GatedDeltaKernel (the
  recurrence) and InnerGatedDeltaNet (its short convolution) -- claimed by the
  same single subtree override, because torchtitan rejects an override whose
  ancestor another override already claims.
* The short convolution is unchanged: it reuses ascend_causal_conv1d.

fla-npu is an optional add-on (ADR-004), not a base dependency: it needs its
own wheel build and is not part of the NIGHTLY baseline. This module is
therefore import-side-effect-free -- it only probes optional_module("fla_npu")
here and defers every fla_npu.ops.ascendc attribute access to call time. The
canonical import fla_npu belongs in _bootstrap.setup() (after torch_npu, after
the CANN environment is sourced, before the first NPU op). When the wheel is
missing, the override is not registered and the plain-torch recurrence from
kernels/gdn.py keeps running (loud WARNING, no silent eager, ADR-004).

Layout legend (all head-first; C = chunk size, NT = chunk count):

    q, k  [B, H, T, K]   v  [B, H, T, V]   g, beta, g_cumsum  [B, H, T] (fp32)
    o     [B, H, T, V]   A  [B, H, T, C]   w  [B, H, T, K]    u  [B, H, T, V]
    h     [B, H, NT, K, V]

Q/K arrive L2-normalised (the caller runs kda.l2norm exactly as _chunk_gdn_btnk
does before ascend_chunk_gdn), so the fla pipeline runs with
use_qk_l2norm_in_kernel=False and l2norm's gradient flows through torch autograd
rather than the fused op.

Fused-shape gate (known-good on 910B2, all verified end-to-end in tests/repro):
V in {128, 256} (recompute_w_u tiles V to {128, 256}), K <= 128 (K >= 256
crashes the aicore and poisons the stream), chunk_size in {64, 128}, q/k/v in
fp16 or bf16 with equal heads. Anything outside that falls back to
ascend_chunk_gdn in the kernel class; the custom op re-raises so the gate is
authoritative in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torchtitan.config import derive, override
from torchtitan.models.qwen3_5.gdn import GatedDeltaKernel, InnerGatedDeltaNet

from ascend_titan.kernels._probe import optional_module
from ascend_titan.kernels.gdn import (
    CHUNK_SIZE,
    AscendGatedDeltaKernel,
    AscendInnerGatedDeltaNet,
    ascend_chunk_gdn,
)
from ascend_titan.kernels.kda import l2norm

logger = logging.getLogger(__name__)

__all__ = [
    "AscendFusedGatedDeltaKernel",
    "AscendFusedInnerGatedDeltaNet",
    "fused_chunk_gdn",
    "npu_gated_delta_net_fused",
]

# Stable fla_npu.ops.ascendc entries the fused pipeline needs. Checked at call
# time (never at import), because reaching fla_npu.ops.ascendc finishes loading
# the op-api libraries, which must not happen while this module is imported.
_REQUIRED_ASCENDC_OPS = (
    "chunk_local_cumsum",
    "chunk_scaled_dot_kkt",
    "solve_tri",
    "recompute_w_u_fwd",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_fwd_o",
    "chunk_bwd_dv_local",
    "chunk_gated_delta_rule_bwd_dhu",
    "chunk_bwd_dqkwg",
    "prepare_wy_repr_bwd_da",
    "prepare_wy_repr_bwd_full",
)

_FUSED_CHUNK_SIZES = (64, 128)
_FUSED_VALUE_DIMS = (128, 256)
_MAX_FUSED_KEY_DIM = 128

# Import-side-effect-free probe: this is the optional-module WARNING degrade path
# (ADR-004). In the real NIGHTLY path the canonical import already happened in
# _bootstrap.setup(), so this importlib.import_module is a cached no-op; on a CPU
# box without fla_npu/CANN it degrades to (None, err) and only logs a warning.
_fla_npu, _fla_err = optional_module("fla_npu")
_AVAILABLE = _fla_npu is not None
if not _AVAILABLE:
    logger.warning(
        "[ascend_titan] fla-npu unavailable (%s); GDN keeps the plain-torch "
        "recurrence from kernels.gdn (ADR-004)",
        _fla_err,
    )


def _fused_shape_gate(
    q_BHTK: torch.Tensor, k_BHTK: torch.Tensor, v_BHTV: torch.Tensor, chunk_size: int
) -> bool:
    """Whether the AscendC path can fuse this shape on 910B2."""
    if q_BHTK.ndim != 4 or k_BHTK.ndim != 4 or v_BHTV.ndim != 4:
        return False
    if q_BHTK.shape[:3] != k_BHTK.shape[:3] or q_BHTK.shape[:3] != v_BHTV.shape[:3]:
        return False
    if q_BHTK.dtype != k_BHTK.dtype or q_BHTK.dtype != v_BHTV.dtype:
        return False
    if q_BHTK.dtype not in (torch.float16, torch.bfloat16):
        return False
    return (
        q_BHTK.shape[-1] <= _MAX_FUSED_KEY_DIM
        and v_BHTV.shape[-1] in _FUSED_VALUE_DIMS
        and int(chunk_size) in _FUSED_CHUNK_SIZES
    )


def _ascendc():
    """Return fla_npu.ops.ascendc, guarding the 11 entries we need.

    Called inside the custom-op bodies only, so importing this module never
    loads fla-npu's op-api libraries. Raises when a needed entry is absent.
    """
    if _fla_npu is None:
        raise RuntimeError("fla-npu is not installed; the fused GDN path is unavailable")
    # fla_npu's top-level __init__ does not import the .ops subpackage, so reach
    # it explicitly (cached no-op on the second call onward).
    import importlib

    ascendc = importlib.import_module("fla_npu.ops.ascendc")
    missing = [name for name in _REQUIRED_ASCENDC_OPS if not hasattr(ascendc, name)]
    if missing:
        raise RuntimeError(
            "fla_npu.ops.ascendc is missing entries the fused GDN path needs: " + ", ".join(missing)
        )
    return ascendc


if _AVAILABLE:

    @torch.library.custom_op("ascend_titan::chunk_gated_delta_rule", mutates_args=())
    def _chunk_gdn_fwd(
        q_BHTK: torch.Tensor,
        k_BHTK: torch.Tensor,
        v_BHTV: torch.Tensor,
        g_BHT: torch.Tensor,
        beta_BHT: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns (o, g_cumsum, A): g_cumsum and A are saved for the backward
        # pass and are non-differentiable outputs.
        if not _fused_shape_gate(q_BHTK, k_BHTK, v_BHTV, chunk_size):
            raise ValueError(
                f"fused GDN shape gate failed: K={q_BHTK.shape[-1]} "
                f"V={v_BHTV.shape[-1]} chunk_size={chunk_size}"
            )
        ascendc = _ascendc()
        scale = float(k_BHTK.shape[-1] ** -0.5)
        g_BHT = g_BHT.contiguous().float()
        beta_BHT = beta_BHT.contiguous().float()

        # 1. Chunk-local cumulative sum of the log gate -> [B,H,T] fp32.
        g_cumsum_BHT = ascendc.chunk_local_cumsum(
            g_BHT, chunk_size, head_first=True, output_dtype="float32"
        )
        # 2. Lower-triangular decayed K^T K matrix -> [B,H,T,C] fp32.
        A_BHTC = ascendc.chunk_scaled_dot_kkt(k_BHTK, g_cumsum_BHT, beta_BHT, chunk_size=chunk_size)
        # 3. (I + A)^-1: npu_solve_tri computes (I + A)^-1, and fla's A
        #    carries the positive sign, so no negation (our _chnk_forward feeds -A).
        #    solve_tri wants BSND [B,S,H,BT] and fp16/bf16 (no fp32).
        A_solved = ascendc.solve_tri(
            A_BHTC.transpose(1, 2).contiguous().to(k_BHTK.dtype), layout="bsnd"
        )
        A_BHTC = A_solved.transpose(1, 2).contiguous()
        # 4. W and U representations.
        w_BHTK, u_BHTV = ascendc.recompute_w_u_fwd(
            k_BHTK, v_BHTV, beta_BHT, A_BHTC, chunk_size, g=g_cumsum_BHT
        )
        torch.npu.synchronize()
        # 5. Materialise the per-chunk hidden state.
        h_BHTKV, v_new_BHTV, _ = ascendc.chunk_gated_delta_rule_fwd_h(
            k_BHTK,
            w_BHTK,
            u_BHTV,
            g=g_cumsum_BHT,
            gk=None,
            initial_state=None,
            output_final_state=False,
            chunk_size=chunk_size,
        )
        # 6. Attend over the chunk with Q.
        o_BHTV = ascendc.chunk_fwd_o(
            q_BHTK,
            k_BHTK,
            v_new_BHTV,
            h_BHTKV,
            scale,
            g=g_cumsum_BHT,
            g_gamma=None,
            chunk_size=chunk_size,
        )
        return o_BHTV, g_cumsum_BHT, A_BHTC

    @_chunk_gdn_fwd.register_fake
    def _chunk_gdn_fwd_fake(q_BHTK, k_BHTK, v_BHTV, g_BHT, beta_BHT, chunk_size):
        B, H, T, _ = q_BHTK.shape
        o_BHTV = torch.empty_like(v_BHTV)
        g_cumsum_BHT = g_BHT.new_empty((B, H, T))
        A_BHTC = k_BHTK.new_empty((B, H, T, int(chunk_size)))
        return o_BHTV, g_cumsum_BHT, A_BHTC


if _AVAILABLE:

    @torch.library.custom_op("ascend_titan::chunk_gated_delta_rule_backward", mutates_args=())
    def _chunk_gdn_bwd(
        d_o_BHTV: torch.Tensor,
        q_BHTK: torch.Tensor,
        k_BHTK: torch.Tensor,
        v_BHTV: torch.Tensor,
        g_cumsum_BHT: torch.Tensor,
        beta_BHT: torch.Tensor,
        A_BHTC: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward of the fused recurrence, head-first throughout."""
        ascendc = _ascendc()
        scale = float(k_BHTK.shape[-1] ** -0.5)
        g_BHT = g_cumsum_BHT.contiguous().float()
        beta_BHT = beta_BHT.contiguous().float()
        d_o_BHTV = d_o_BHTV.contiguous()

        w_BHTK, u_BHTV = ascendc.recompute_w_u_fwd(
            k_BHTK, v_BHTV, beta_BHT, A_BHTC, chunk_size, g=g_BHT
        )
        torch.npu.synchronize()

        h_BHTKV, v_new_BHTV, _ = ascendc.chunk_gated_delta_rule_fwd_h(
            k_BHTK,
            w_BHTK,
            u_BHTV,
            g=g_BHT,
            gk=None,
            initial_state=None,
            output_final_state=False,
            chunk_size=chunk_size,
        )
        dv_BHTV = ascendc.chunk_bwd_dv_local(
            q_BHTK, k_BHTK, d_o_BHTV, g_BHT, scale, chunk_size, g_gamma=None, A=A_BHTC
        )
        dh_BHTKV, _dh0, dv_BHTV = ascendc.chunk_gated_delta_rule_bwd_dhu(
            q_BHTK,
            k_BHTK,
            w_BHTK,
            d_o_BHTV,
            dv_BHTV,
            scale,
            chunk_size,
            g=g_BHT,
            gK=None,
            h0=None,
            dht=None,
            use_exp2=False,
            transpose_state_layout=False,
        )
        dq_BHTK, dk_BHTK, dw_BHTK, dg_BHT = ascendc.chunk_bwd_dqkwg(
            q_BHTK,
            k_BHTK,
            v_new_BHTV,
            g_BHT,
            h_BHTKV,
            d_o_BHTV,
            dh_BHTKV,
            dv_BHTV,
            chunk_size,
            w=None,
            g_gamma=None,
            scale=scale,
            use_exp2=False,
            transpose_state_layout=False,
        )
        dA_BHTC = ascendc.prepare_wy_repr_bwd_da(
            k_BHTK, v_BHTV, beta_BHT, A_BHTC, dw_BHTK, dv_BHTV, g_BHT, chunk_size=chunk_size
        )
        dk2_BHTK, dv_BHTV, dbeta_BHT, dg2_BHT = ascendc.prepare_wy_repr_bwd_full(
            k_BHTK, v_BHTV, beta_BHT, A_BHTC, dA_BHTC, dw_BHTK, dv_BHTV, g_BHT, chunk_size
        )
        dk_BHTK = dk_BHTK + dk2_BHTK
        dg_BHT = dg_BHT + dg2_BHT
        # Backward of the forward chunk cumsum is its reverse cumsum.
        dg_BHT = ascendc.chunk_local_cumsum(
            dg_BHT, chunk_size, reverse=True, head_first=True, output_dtype="float32"
        )
        return dq_BHTK, dk_BHTK, dv_BHTV, dbeta_BHT, dg_BHT

    @_chunk_gdn_bwd.register_fake
    def _chunk_gdn_bwd_fake(d_o, q, k, v, g_cumsum, beta, A, chunk_size):
        return (
            torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(v),
            torch.empty_like(beta),
            torch.empty_like(g_cumsum),
        )


if _AVAILABLE:

    def _setup_context(ctx, inputs, output):
        q, k, v, g, beta, chunk_size = inputs
        o, g_cumsum, A = output
        ctx.save_for_backward(q, k, v, beta, g_cumsum, A)
        ctx.mark_non_differentiable(g_cumsum, A)
        ctx.chunk_size = chunk_size

    def _backward(ctx, grad_o, _d_g_cumsum, _d_A):
        q, k, v, beta, g_cumsum, A = ctx.saved_tensors
        dq, dk, dv, dbeta, dg = _chunk_gdn_bwd(grad_o, q, k, v, g_cumsum, beta, A, ctx.chunk_size)
        return dq, dk, dv, dg, dbeta.to(beta.dtype), None

    _chunk_gdn_fwd.register_autograd(_backward, setup_context=_setup_context)

    def fused_chunk_gdn(
        q_BHTK: torch.Tensor,
        k_BHTK: torch.Tensor,
        v_BHTV: torch.Tensor,
        g_BHT: torch.Tensor,
        beta_BHT: torch.Tensor,
        *,
        chunk_size: int = CHUNK_SIZE,
    ) -> torch.Tensor:
        """ascend_chunk_gdn's contract, fused on fla-npu AscendC kernels.

        Same SDPA layout in/out and same pre-normalised Q/K expectation as
        kernels.gdn.ascend_chunk_gdn; only the implementation differs (11 AscendC
        kernels behind one custom op instead of the torch recurrence). Unsupported
        shapes raise -- call the kernel class, which falls back to
        ascend_chunk_gdn, for anything outside the gate.
        """
        o_BHTV, _g_cumsum, _A = _chunk_gdn_fwd(
            q_BHTK, k_BHTK, v_BHTV, g_BHT, beta_BHT, int(chunk_size)
        )
        return o_BHTV


def _chunk_gdn_btnk_fused(
    q_BTNK: torch.Tensor,
    k_BTNK: torch.Tensor,
    v_BTNV: torch.Tensor,
    g_BTN: torch.Tensor,
    beta_BTN: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Fused recurrence on upstream's [B, T, N, *] layout, with torch fallback."""
    q_BHTK = l2norm(q_BTNK).transpose(1, 2)
    k_BHTK = l2norm(k_BTNK).transpose(1, 2)
    v_BHTV = v_BTNV.transpose(1, 2)
    g_BHT = g_BTN.transpose(1, 2)
    beta_BHT = beta_BTN.transpose(1, 2)
    if _fla_npu is not None and _fused_shape_gate(q_BHTK, k_BHTK, v_BHTV, chunk_size):
        return (
            fused_chunk_gdn(q_BHTK, k_BHTK, v_BHTV, g_BHT, beta_BHT, chunk_size=chunk_size)
            .transpose(1, 2)
            .contiguous()
        )
    return (
        ascend_chunk_gdn(q_BHTK, k_BHTK, v_BHTV, g_BHT, beta_BHT, chunk_size=chunk_size)
        .transpose(1, 2)
        .contiguous()
    )


class AscendFusedGatedDeltaKernel(AscendGatedDeltaKernel):
    """GatedDeltaKernel on fla-npu's fused chunk recurrence.

    Same head-expansion for grouped linear attention and same per-document loop
    as the parent; only the chunk-level recurrence switches from the torch
    decomposition to the fused custom op (and back, for shapes outside the gate).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AscendGatedDeltaKernel.Config):
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
            return _chunk_gdn_btnk_fused(
                xq_TNK.unsqueeze(0),
                xk_TNK.unsqueeze(0),
                xv_TNV.unsqueeze(0),
                g_TN.unsqueeze(0),
                beta_TN.unsqueeze(0),
                chunk_size=self.chunk_size,
            ).squeeze(0)

        bounds = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens).tolist()
        pieces = []
        for begin, end in zip(bounds[:-1], bounds[1:], strict=True):
            if end <= begin:
                continue
            pieces.append(
                _chunk_gdn_btnk_fused(
                    xq_TNK[begin:end].unsqueeze(0),
                    xk_TNK[begin:end].unsqueeze(0),
                    xv_TNV[begin:end].unsqueeze(0),
                    g_TN[begin:end].unsqueeze(0),
                    beta_TN[begin:end].unsqueeze(0),
                    chunk_size=self.chunk_size,
                ).squeeze(0)
            )
        return torch.cat(pieces, dim=0)


class AscendFusedInnerGatedDeltaNet(AscendInnerGatedDeltaNet):
    """InnerGatedDeltaNet with the fused recurrence.

    The short convolution and gate math are identical to kernels/gdn.py's
    pure-torch version, so this class only swaps the kernel config to the
    fused chunk recurrence; forward is inherited verbatim from
    AscendInnerGatedDeltaNet (single source of truth, not a copy).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AscendInnerGatedDeltaNet.Config):
        pass


if _AVAILABLE and GatedDeltaKernel is not None:
    _MODEL_AVAILABLE = True

    @override(
        target=InnerGatedDeltaNet.Config,
        exact=True,
        description="Gated DeltaNet on Ascend: torch depthwise causal conv1d + the "
        "chunk-parallel delta rule fused on fla-npu AscendC kernels (perf path, R5)",
    )
    def npu_gated_delta_net_fused(
        cfg: InnerGatedDeltaNet.Config,
    ) -> AscendFusedInnerGatedDeltaNet.Config:
        """Sibling of kernels.gdn.npu_gated_delta_net; same subtree, fused kernel."""
        new = derive(cfg, AscendFusedInnerGatedDeltaNet.Config)
        new.kernel = derive(cfg.kernel, AscendFusedGatedDeltaKernel.Config)
        return new
