"""The primitives a recipe applies to an upstream config, one delta at a time.

Nothing here decides *what* a model needs -- that judgement belongs in the model's
own ``recipes.py``, written out call by call so the reader can see which modules we
swapped and which we left alone. This module only holds the two operations that
would otherwise be re-implemented in every recipe, and the knowledge each carries:

``add_override`` / ``swap_override``   how ``override.imports`` may be edited --
    in particular that two overrides claiming the same node cannot both be active,
    so a drop-in variant is a *swap*, never a stack.
``flex_to_varlen``                     the FlexAttention -> VarlenAttention node
    surgery, its feature-detected disappearance condition, and the vision-tower
    exception. Both are load-bearing and neither is obvious.

The capability matrix builds its generic ``npu_minimal`` / ``npu_fused`` transforms
out of these same primitives (``recipes/matrix.py``); a recipe must not call those
transforms (``tests/unit/test_models_registry.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.trainer import Trainer


def add_override(config: Trainer.Config, target: str) -> bool:
    """Activate an override target, in place. Idempotent; ``True`` if it was added."""
    if target in config.override.imports:
        return False
    config.override.imports = [*config.override.imports, target]
    return True


def swap_override(config: Trainer.Config, *, remove: str, add: str) -> bool:
    """Replace one override target with a mutually-exclusive sibling, in place.

    Two overrides that claim the same ``Configurable.Config`` node cannot both be
    active (torchtitan raises a per-node conflict), so activating a drop-in
    variant means removing the original and appending the variant -- never
    stacking. Idempotent: an already-swapped list is left untouched. Returns
    ``True`` when ``remove`` was present (and thus removed).
    """
    if remove not in config.override.imports:
        return False
    config.override.imports = [imp for imp in config.override.imports if imp != remove]
    add_override(config, add)
    return True


def _flex_attention_is_usable() -> bool:
    """True when upstream's FlexAttention is a *usable* choice on this machine.

    Two conditions, and both matter:

    * torch_npu must have lifted torch's device whitelist (``_validate_device``
      rejects anything outside ``{cuda, cpu, xpu, hpu, mps}``, TORCH-1); master
      patches it and marks the module with ``_npu_device_patched``.
    * inductor must be able to lower a **document mask**. That mask indexes a
      segment-id tensor (``aten.index.Tensor``) inside flex's pointwise subgraph,
      and torch_npu only lowers indirect memory access when
      ``inductor_indirect_memory_mode`` is set -- which its config does only on
      Ascend950. On 910B2 it stays ``None``, the index falls back to an
      ExternKernel, and building a buffer inside a pointwise subgraph raises
      ``SubgraphLoweringException``; where it does not raise, flex silently
      materialises the O(T^2) score matrix and the run OOMs instead (measured:
      llama3 debugmodel asks for 20 GiB on a 60 GiB card).

    So this is a *hardware* gate, not a missing package: Triton-Ascend is part of
    the baseline and inductor compiles fine (``scripts/install_triton.sh`` proves
    fwd+bwd), yet model-level flex still has no path on 910B2.
    """
    from torch.nn.attention import flex_attention

    if not getattr(flex_attention, "_npu_device_patched", False):
        return False
    from torch_npu._inductor import config as npu_inductor_config

    return getattr(npu_inductor_config, "inductor_indirect_memory_mode", None) is not None


def flex_to_varlen(config: Trainer.Config, *, keep: Sequence[str] = ()) -> list[str]:
    """Convert FlexAttention nodes to VarlenAttention, in place. Returns the fqns converted.

    A **no-op once flex is usable on this machine** (feature detection, see
    :func:`_flex_attention_is_usable`) -- that is this delta's disappearance
    condition (P12), not a version check. Flex-only fields (block_size,
    kernel_options) are dropped by the conversion; a model that depends on
    Flex-only features (sinks, custom mask mods) fails later and gets attributed,
    which is the point of measuring it.

    ``keep`` holds fqn substrings whose nodes must stay flex. The case that
    exists is a vision tower: its attention is fed a ``BlockMask`` built by
    ``create_block_diagonal_mask``, while ``VarlenAttention.forward`` asserts
    ``isinstance(attention_masks, VarlenMetadata)`` -- so converting that node
    swaps a clear failure ("flex needs inductor") for a confusing one deep inside
    the tower ("must be VarlenMetadata but got BlockMask").

    **Never converts under Context Parallel.** Upstream implements CP's mask
    sharding for ``BlockMask`` only and raises ``NotImplementedError: Context
    Parallel is not supported with ScaledDotProductAttention or VarlenAttention``
    (``models/common/decoder.py:186``); torch's CP hooks reach flex (all-gather
    K/V, ``_cp_custom_ops.flex_cp_allgather``) and SDPA (ring attention), and
    nothing else. Converting therefore turns a CP run that *works* into a hard
    stop. Eager flex is the expensive choice, but under CP the sequence is already
    sharded, so the O(T^2) scores it materialises are O(T^2/cp) per rank.
    Measured 2026-09-02 on 910B2: ``llama3_debugmodel_cp4`` on 4 cards,
    ``step 2 loss 7.74610``, 33.10 GiB, rc=0.
    """
    if config.parallelism.context_parallel_degree > 1:
        return []
    if _flex_attention_is_usable():
        return []
    converted: list[str] = []
    for fqn, _cfg, parent, attr in list(config.traverse(FlexAttention.Config)):
        if any(marker in fqn for marker in keep):
            continue
        new = VarlenAttention.Config()
        if isinstance(parent, list):
            parent[attr] = new  # type: ignore[index]
        else:
            setattr(parent, attr, new)
        converted.append(fqn)
    return converted
