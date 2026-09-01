import torch
import torch_npu  # noqa
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
import fla_npu  # noqa
from fla_npu.ops import ascendc as ac
names = [n for n in dir(ac) if n.startswith("npu_")]
print("fla_npu ascendc npu_* ops:", len(names))
for n in ("npu_solve_tri", "npu_chunk_local_cumsum", "npu_chunk_scaled_dot_kkt",
          "npu_recompute_w_u_fwd", "npu_chunk_gated_delta_rule_fwd_h",
          "npu_chunk_fwd_o", "npu_chunk_bwd_dv_local", "npu_chunk_gated_delta_rule_bwd_dhu",
          "npu_chunk_bwd_dqkwg", "npu_prepare_wy_repr_bwd_da", "npu_prepare_wy_repr_bwd_full"):
    print("  has", n, hasattr(ac, n))
