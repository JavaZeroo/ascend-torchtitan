import fla_npu.ops.ascendc as ac
names = ["chunk_local_cumsum", "chunk_scaled_dot_kkt", "solve_tri",
         "recompute_w_u_fwd", "chunk_gated_delta_rule_fwd_h", "chunk_fwd_o",
         "chunk_bwd_dv_local", "chunk_gated_delta_rule_bwd_dhu",
         "chunk_bwd_dqkwg", "prepare_wy_repr_bwd_da", "prepare_wy_repr_bwd_full"]
for n in names:
    print(n, hasattr(ac, n))
print("__all__" in dir(ac))
