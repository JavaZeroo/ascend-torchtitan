import torch
import torch_npu  # noqa
from fla_npu.ops import ascendc as ac

B, H, T, chunk = 1, 4, 128, 64
for K, V in ((64,64),(64,128),(64,256),(128,64),(128,128),(128,256)):
    k = torch.randn(B, H, T, K, dtype=torch.bfloat16, device="npu:0")
    v = torch.randn(B, H, T, V, dtype=torch.bfloat16, device="npu:0")
    beta = torch.rand(B, H, T, dtype=torch.bfloat16, device="npu:0")
    A = torch.randn(B, H, T, chunk, dtype=torch.bfloat16, device="npu:0")
    g = -torch.rand(B, H, T, device="npu:0")
    try:
        w, u = ac.recompute_w_u_fwd(k, v, beta, A, chunk, g=g)
        torch.npu.synchronize()
        print("recompute_w_u K=%3d V=%3d -> OK w=%s u=%s" % (K, V, tuple(w.shape), tuple(u.shape)))
    except Exception as e:
        torch.npu.synchronize()
        print("recompute_w_u K=%3d V=%3d -> FAIL %r" % (K, V, str(e)[:120]))
