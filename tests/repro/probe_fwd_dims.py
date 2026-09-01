import torch
import torch_npu  # noqa
from fla_npu.ops import ascendc as ac

# 0.8B GDN: per-head K=key_head_dim=64, V=value_head_dim=64; heads after expansion = 4
# Build [B, H, T, K] / [B, H, T, V] with H=4, K=64, V=64, T=128, chunk=64
B, H, T, K, V, chunk = 1, 4, 128, 64, 64, 64
dtype = torch.bfloat16
k = torch.randn(B, H, T, K, dtype=dtype, device="npu:0")
q = torch.randn(B, H, T, K, dtype=dtype, device="npu:0")
v = torch.randn(B, H, T, V, dtype=dtype, device="npu:0")
# gate: need log gate cumsum as g; fla uses g as cumulative log-gate in fwd_h
g = -torch.rand(B, H, T, dtype=torch.float32, device="npu:0").cumsum(-1)
beta = torch.rand(B, H, T, dtype=dtype, device="npu:0")

for Kd in (64, 128):
    for Vd in (64, 128, 256):
        kk = torch.randn(B, H, T, Kd, dtype=dtype, device="npu:0")
        vv = torch.randn(B, H, T, Vd, dtype=dtype, device="npu:0")
        try:
            h, v_new, _ = ac.chunk_gated_delta_rule_fwd_h(
                kk, kk, vv, g=g[:, :, :T], initial_state=None,
                output_final_state=False, chunk_size=chunk,
                cu_seqlens=None, chunk_indices=None)
            torch.npu.synchronize()
            o = ac.chunk_fwd_o(q[:, :, :T, :Kd], kk, v_new, h, 1.0,
                g=g[:, :, :T], cu_seqlens=None, chunk_indices=None, chunk_size=chunk)
            torch.npu.synchronize()
            print("K=%3d V=%3d -> OK o=%s" % (Kd, Vd, tuple(o.shape)))
        except Exception as e:
            torch.npu.synchronize()
            print("K=%3d V=%3d -> FAIL %r" % (Kd, Vd, str(e)[:140]))
