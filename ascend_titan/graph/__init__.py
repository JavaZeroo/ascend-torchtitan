"""L2: graph mode on Ascend (torchair).

torchair is Ascend's GE-graph backend for ``torch.compile``. It ships inside
torch_npu, but only when torch_npu was built without ``--disable_torchair``, so
:func:`require_torchair` says exactly how to get it rather than failing later
with an opaque dynamo error.

    from ascend_titan.graph import npu_graph
    npu_graph(config, components=["loss"])

What works today (measured 2026-08-30, torch 2.15.0.dev20260812 + torch_npu
master with torchair, CANN 9.1.0, 910B2):

    components=["loss"]   🟢  qwen3_debugmodel_npu, 10 steps
    components=["model"]  🔴  our varlen attention custom op has no GE converter
                              (OURS-13); torchair looks for an AscendIR named
                              after the op and stops.

See ``ascend_titan/graph/README.md``.
"""

from ascend_titan.graph.torchair_backend import npu_graph, require_torchair, torchair_available

__all__ = ["npu_graph", "require_torchair", "torchair_available"]
