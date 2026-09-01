import torch
import torch_npu  # noqa
from fla_npu.ops import ascendc as ac
import inspect

B, H, T, K, V, chunk = 1, 4, 256, 64, 64, 64
dtype = torch.bfloat16
q = torch.randn(B, H, T, K, dtype=dtype, device="npu:0")
k = torch.randn(B, H, T, K, dtype=dtype, device="npu:0")
v = torch.randn(B, H, T, V, dtype=dtype, device="npu:0")
g = -torch.rand(B, H, T, device="npu:0")

h, v_new, _ = ac.chunk_gated_delta_rule_fwd_h(
    k, k, v, g=g, initial_state=None, output_final_state=False,
    chunk_size=chunk, cu_seqlens=None, chunk_indices=None)
o = ac.chunk_fwd_o(q, k, v_new, h, 1.0, g=g, cu_seqlens=None, chunk_indices=None, chunk_size=chunk)
torch.npu.synchronize()
print("fwd o", tuple(o.shape))
print("h", tuple(h.shape), "v_new", tuple(v_new.shape))

for name in ["chunk_bwd_dv_local","chunk_gated_delta_rule_bwd_dhu","chunk_bwd_dqkwg","prepare_wy_repr_bwd_da","prepare_wy_repr_bwd_full","recompute_w_u_fwd","chunk_scaled_dot_kkt","chunk_local_cumsum","solve_tri"]:
    print(name, str(inspect.signature(getattr(ac, name))))
