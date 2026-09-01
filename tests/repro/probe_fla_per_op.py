"""Find the exact K/V boundary for recompute_w_u, and whether fwd_h/fwd_o also have
the same boundary. Real qwen3.5 debugmodel is K=V=64, so this decides feasibility.
"""
import torch
import torch_npu  # noqa
from fla_npu.ops import ascendc as ac

torch.npu.set_device(0)

def run(tag, B, Hk, Hv, T, K, V, dtype):
    torch.manual_seed(0)
    k = torch.randn(B, Hk, T, K, dtype=dtype).npu()
    v = torch.randn(B, Hv, T, V, dtype=dtype).npu()
    beta = torch.randn(B, Hv, T).float().npu()
    A = torch.randn(B, Hv, T, K, dtype=dtype).npu()  # A last dim == K
    g = torch.randn(B, Hv, T).float().npu()
    try:
        w, u = ac.npu_recompute_w_u_fwd(k, v, beta, A, 64, g=g, gk=None,
                                        cu_seqlens=None, chunk_indices=None)
        torch.npu.synchronize()
        print("  OK   %-24s" % tag)
        return True
    except Exception as e:
        print("  FAIL %-24s %r" % (tag, str(e)[:60]))
        return False

print("=== sweep K with fixed V=256 (official V) ===")
for K in (32, 64, 96, 128):
    run("K=%d V=256" % K, 1, 2, 4, 256, K, 256, torch.float16)

print("=== sweep V with fixed K=128 ===")
for V in (32, 64, 96, 128, 256):
    run("K=128 V=%d" % V, 1, 2, 4, 256, 128, V, torch.float16)

print("=== sweep K with fixed V=64 ===")
for K in (64, 128):
    run("K=%d V=64" % K, 1, 2, 4, 256, K, 64, torch.float16)
