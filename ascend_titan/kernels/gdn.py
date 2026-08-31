"""Gated DeltaNet (Qwen3.5) on Ascend.

Targets ``torchtitan/models/qwen3_5/gdn.py::GatedDeltaKernel.Config``. Upstream
dispatches to flash-linear-attention's Triton kernels
(``fla.ops.gated_delta_rule``); those compile for CUDA and fail on Ascend even
with Triton-Ascend installed -- ``bishengir-compile`` rejects the generated MLIR.

attn_gym ships a device-agnostic implementation of the same recurrence,
``attn_gym.linear.gdn.chunk_gdn(impl="reference")``, documented as "eager
PyTorch" with FP32 recurrence math. So this override keeps upstream's *math* and
only selects the implementation -- the same shape as the KDA and RoPE overrides.

We do not call attn_gym's reference at runtime, though: it is written to be read,
not to be fast. Its intra-chunk step inverts the unit lower-triangular transition
matrix by forward substitution -- ``for row in range(1, chunk_size)``, each
iteration cloning the whole ``[B, H, chunks, C, C]`` block. At debugmodel size
that is invisible; at 0.8B (24 layers, 4096 context, under SelectiveAC, which
routes every one of those ops through a ``__torch_dispatch__`` mode) a single
step did not finish in 10 minutes. ``ascend_chunk_gdn`` below is the same
decomposition with that one loop replaced by a closed form, and
``tests/unit/test_kernel_gdn.py`` pins it against attn_gym's reference -- which
stays the oracle, exactly what a readable reference implementation is for.

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
import torch.nn.functional as F
from torchtitan.config import derive, override
from torchtitan.models.qwen3_5.gdn import GatedDeltaKernel, InnerGatedDeltaNet

from ascend_titan.kernels._probe import require_op  # noqa: F401  (P14: hard dependency)
from ascend_titan.kernels.kda import ascend_causal_conv1d, l2norm

__all__ = [
    "AscendGatedDeltaKernel",
    "AscendInnerGatedDeltaNet",
    "ascend_chunk_gdn",
    "npu_gated_delta_net",
]

CHUNK_SIZE = 64
"""Tokens per chunk -- the same default as ``attn_gym`` and fla.

Bigger chunks are faster here: the chunk loop is sequential Python and every op
inside it pays torchtitan's activation-checkpoint dispatch mode, so step time
tracks ``tokens / chunk_size``. Measured on qwen3.5-0.8B, 910B2, one card
(step-1 tps): 64 -> 231, 128 -> 314, 256 -> 385.

We stay at 64 anyway, for conditioning. The intra-chunk transition matrix is
``(I - A)^-1`` for a strictly lower-triangular ``A`` whose entries are
``beta * (k_i . k_j) * decay``, so bounded by one; its largest entry grows fast
with the chunk size -- measured 5.7e3 (C=64), 5.7e6 (C=128), 5.7e15 (C=256) --
and everything downstream multiplies by it. That is a property of the
decomposition, not of how we invert, which is why fla and attn_gym also use 64.
It is also the size our alignment tests target.

(An earlier note here blamed qwen3.5-0.8B's step-4 non-finite loss on raising
this value, and a later one blamed the training configuration. Both were wrong:
it was the Neumann series in ``_UnitLowerInverse``, which C=64 also triggered.
See that class.)
"""


BASE_BLOCK = 16
"""Diagonal-block size for the inverse. See :class:`_UnitLowerInverse`."""


def _series_inverse(strictly_lower: torch.Tensor) -> torch.Tensor:
    """``(I - A)^-1`` by the Neumann series, evaluated by doubling.

    Exact for nilpotent ``A``: ``I + A + ... + A^(C-1)``, in ``log2(C)`` matmuls.
    Only used on the small diagonal blocks, where the highest power formed is
    ``A^(BASE_BLOCK/2)`` -- at full chunk size the intermediate powers are far
    larger than the sum they telescope into, and overflow fp32.
    """
    size = strictly_lower.shape[-1]
    eye = torch.eye(size, dtype=strictly_lower.dtype, device=strictly_lower.device)
    inverse = eye + strictly_lower
    power = strictly_lower
    covered = 2
    while covered < size:
        power = power @ power
        inverse = inverse + power @ inverse
        covered *= 2
    return inverse


class _UnitLowerInverse(torch.autograd.Function):
    """``(I - A)^-1`` for a *strictly* lower-triangular ``A``, with an exact backward.

    Forward is block forward substitution -- substitution's numerics at matmul's
    cost. With ``X = I + A X`` and ``A_ii`` strictly lower inside each diagonal
    block::

        X_ii     = (I - A_ii)^-1                        (batched, small blocks)
        X[i, :i] = X_ii @ (A[i, :i] @ X[:i, :i])        (one matmul per block row)

    Every step after the diagonal uses an already-final ``X``, never a power of
    ``A``. That is the property attn_gym's ``for row in range(1, chunk_size)``
    loop has and the Neumann series does not.

    Three implementations were measured on 910B2 (batched ``[1, 16, chunks, 64, 64]``):

    ==================  =========  ==========  ===================================
    method              chunks=1   chunks=64   note
    ==================  =========  ==========  ===================================
    Neumann doubling      0.50 ms     0.52 ms  overflows fp32 on trained gates
    ``solve_triangular``  1.05 ms    54.93 ms  correct; cost is linear in chunks
    block substitution    1.50 ms     2.44 ms  correct and flat -- this one
    ==================  =========  ==========  ===================================

    Backward is the closed form and does not care how the forward was computed:
    ``X = (I - A)^-1`` has ``dX = X dA X``, so ``grad_A = X^T @ grad_X @ X^T``
    masked back to strictly lower -- two matmuls.
    """

    @staticmethod
    def forward(ctx, strictly_lower_NCC: torch.Tensor) -> torch.Tensor:
        size = strictly_lower_NCC.shape[-1]
        block = min(BASE_BLOCK, size)
        if size % block:
            block = size
        blocks = size // block

        diagonal = torch.stack(
            [
                strictly_lower_NCC[..., i * block : (i + 1) * block, i * block : (i + 1) * block]
                for i in range(blocks)
            ],
            dim=-3,
        )
        diagonal_inverse = _series_inverse(diagonal)

        inverse = torch.zeros_like(strictly_lower_NCC)
        for i in range(blocks):
            begin, end = i * block, (i + 1) * block
            inverse[..., begin:end, begin:end] = diagonal_inverse[..., i, :, :]
            if i:
                below = strictly_lower_NCC[..., begin:end, :begin] @ inverse[..., :begin, :begin]
                inverse[..., begin:end, :begin] = diagonal_inverse[..., i, :, :] @ below
        ctx.save_for_backward(inverse)
        return inverse

    @staticmethod
    def backward(ctx, grad_NCC: torch.Tensor) -> torch.Tensor:
        (inverse,) = ctx.saved_tensors
        transposed = inverse.transpose(-1, -2)
        return (transposed @ grad_NCC @ transposed).tril(-1)


def _unit_lower_inverse(strictly_lower_NCC: torch.Tensor) -> torch.Tensor:
    """See :class:`_UnitLowerInverse`."""
    return _UnitLowerInverse.apply(strictly_lower_NCC)


def _chunk_forward(
    query_BHTK: torch.Tensor,
    key_BHTK: torch.Tensor,
    value_BHTV: torch.Tensor,
    log_decay_BHT: torch.Tensor,
    beta_BHT: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """The chunk-parallel gated delta rule, in ``attn_gym``'s decomposition.

    Mirrors ``attn_gym.linear.gdn.impl.reference.chunk_forward`` step for step;
    the only difference is ``_unit_lower_inverse`` in place of its row loop.
    Kept in the same variable names so the two read side by side.
    """
    batch, heads, sequence, key_dimension = query_BHTK.shape
    value_dimension = value_BHTV.shape[-1]
    scale = key_dimension**-0.5

    padding = (-sequence) % chunk_size
    if padding:
        query_BHTK, key_BHTK, value_BHTV = (
            F.pad(t, (0, 0, 0, padding)) for t in (query_BHTK, key_BHTK, value_BHTV)
        )
        beta_BHT, log_decay_BHT = (F.pad(t, (0, padding)) for t in (beta_BHT, log_decay_BHT))

    padded_length = query_BHTK.shape[-2]
    chunk_count = padded_length // chunk_size
    query = query_BHTK * scale
    value = value_BHTV * beta_BHT[..., None]
    beta_key = key_BHTK * beta_BHT[..., None]

    query, key, value, beta_key = (
        t.reshape(batch, heads, chunk_count, chunk_size, t.shape[-1])
        for t in (query, key_BHTK, value, beta_key)
    )
    cumulative_decay = log_decay_BHT.reshape(batch, heads, chunk_count, chunk_size).cumsum(-1)
    decay_matrix = (
        (cumulative_decay.unsqueeze(-1) - cumulative_decay.unsqueeze(-2)).tril().exp().tril()
    )

    diagonal_and_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device)
    )
    transition = _unit_lower_inverse(
        -((beta_key @ key.transpose(-1, -2)) * decay_matrix).masked_fill(diagonal_and_upper, 0)
    )
    value = transition @ value
    decayed_key = transition @ (beta_key * cumulative_decay.exp()[..., None])

    state = query.new_zeros(batch, heads, key_dimension, value_dimension)
    outputs = []
    strictly_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )
    for chunk in range(chunk_count):
        chunk_query = query[:, :, chunk]
        chunk_key = key[:, :, chunk]
        chunk_value = value[:, :, chunk]
        attention = (
            (chunk_query @ chunk_key.transpose(-1, -2)) * decay_matrix[:, :, chunk]
        ).masked_fill(strictly_upper, 0)
        corrected_value = chunk_value - (decayed_key[:, :, chunk] @ state)
        prior_output = (chunk_query * cumulative_decay[:, :, chunk, :, None].exp()) @ state
        outputs.append(prior_output + attention @ corrected_value)

        final_decay = cumulative_decay[:, :, chunk, -1, None]
        state = (
            state * final_decay[..., None].exp()
            + (
                chunk_key * (final_decay - cumulative_decay[:, :, chunk]).exp()[..., None]
            ).transpose(-1, -2)
            @ corrected_value
        )

    output = torch.stack(outputs, dim=2).reshape(batch, heads, padded_length, value_dimension)
    return output[:, :, :sequence]


def ascend_chunk_gdn(
    query_BHTK: torch.Tensor,
    key_BHTK: torch.Tensor,
    value_BHTV: torch.Tensor,
    log_decay_BHT: torch.Tensor,
    beta_BHT: torch.Tensor,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """``chunk_gdn(impl="reference")``'s contract: SDPA layout, FP32 recurrence."""
    output_dtype = query_BHTK.dtype
    compute_dtype = torch.promote_types(output_dtype, torch.float32)
    query_BHTK, key_BHTK, value_BHTV, log_decay_BHT, beta_BHT = (
        t.to(compute_dtype) for t in (query_BHTK, key_BHTK, value_BHTV, log_decay_BHT, beta_BHT)
    )
    # Explicit casts do not stop autocast from selecting low-precision contractions.
    with torch.autocast(device_type=query_BHTK.device.type, enabled=False):
        output = _chunk_forward(
            query_BHTK,
            key_BHTK,
            value_BHTV,
            log_decay_BHT,
            beta_BHT,
            chunk_size=chunk_size,
        )
    return output.to(output_dtype)


def _chunk_gdn_btnk(
    q_BTNK: torch.Tensor,
    k_BTNK: torch.Tensor,
    v_BTNV: torch.Tensor,
    g_BTN: torch.Tensor,
    beta_BTN: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """``ascend_chunk_gdn`` on upstream's ``[B, T, N, *]`` layout."""
    return ascend_chunk_gdn(
        l2norm(q_BTNK).transpose(1, 2),
        l2norm(k_BTNK).transpose(1, 2),
        v_BTNV.transpose(1, 2),
        g_BTN.transpose(1, 2),
        beta_BTN.transpose(1, 2),
        chunk_size=chunk_size,
    ).transpose(1, 2)


class AscendGatedDeltaKernel(GatedDeltaKernel):
    """``GatedDeltaKernel`` on attn_gym's device-agnostic reference recurrence."""

    @dataclass(kw_only=True, slots=True)
    class Config(GatedDeltaKernel.Config):
        chunk_size: int = CHUNK_SIZE
        """See ``CHUNK_SIZE``. Lower it to compare against fla/attn_gym defaults."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.chunk_size = config.chunk_size

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
                chunk_size=self.chunk_size,
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
                    chunk_size=self.chunk_size,
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
    description="Gated DeltaNet on Ascend: torch depthwise causal conv1d + the chunk-parallel "
    "delta rule in plain torch (fla's kernels are CUDA-only)",
)
def npu_gated_delta_net(cfg: InnerGatedDeltaNet.Config) -> AscendInnerGatedDeltaNet.Config:
    """One override for the whole subtree; see kernels/kda.py for why it must be one."""
    new = derive(cfg, AscendInnerGatedDeltaNet.Config)
    new.kernel = derive(cfg.kernel, AscendGatedDeltaKernel.Config)
    return new
