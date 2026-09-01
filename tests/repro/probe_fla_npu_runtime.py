"""Probe fla_npu runtime capabilities on the NPU (single card)."""
import torch
import torch_npu  # noqa
print("torch", torch.__version__, "torch_npu", torch_npu.__version__)

try:
    import triton
    from triton.runtime import driver
    print("triton", getattr(triton, "__version__", "?"), "driver.active", driver.active)
except Exception as e:
    print("triton import/active failed:", repr(e))

from fla_npu.ops import ascendc as ac

print("probe solve_tri (bsnd):")
for dtype in (torch.float16, torch.bfloat16):
    for d in (64, 128):
        A = torch.randn(1, 1, 64, d, dtype=dtype, device="npu:0").tril(-1).contiguous()
        try:
            out = ac.npu_solve_tri(A, cu_seqlens=None, chunk_indices=None, layout="bsnd")
            torch.npu.synchronize()
            print("  solve_tri bsnd dtype=%s D=%d: OK shape=%s dtype=%s" % (dtype, d, tuple(out.shape), out.dtype))
        except Exception as e:
            print("  solve_tri bsnd dtype=%s D=%d: FAIL %r" % (dtype, d, str(e)[:200]))

print("probe fwd_h/fwd_o:")
B, HK, H_V, T, K, V = 1, 4, 4, 128, 64, 64
chunk = 64
dtype = torch.bfloat16
k = torch.randn(B, HK, T, K, dtype=dtype, device="npu:0")
q = torch.randn(B, HK, T, K, dtype=dtype, device="npu:0")
v = torch.randn(B, H_V, T, V, dtype=dtype, device="npu:0")
g = -torch.rand(B, H_V, T, dtype=torch.float32, device="npu:0").cumsum(-1)
try:
    h, v_new, _ = ac.npu_chunk_gated_delta_rule_fwd_h(
        k, k, v, g=g, initial_state=None, output_final_state=False,
        chunk_size=chunk, cu_seqlens=None, chunk_indices=None)
    torch.npu.synchronize()
    print("  fwd_h OK h=%s v_new=%s" % (tuple(h.shape), tuple(v_new.shape)))
    o = ac.npu_chunk_fwd_o(q, k, v_new, h, 1.0,
        g=g, cu_seqlens=None, chunk_indices=None, chunk_size=chunk)
    torch.npu.synchronize()
    print("  fwd_o OK o=%s dtype=%s" % (tuple(o.shape), o.dtype))
except Exception as e:
    print("  fwd_h/fwd_o FAIL:", repr(e)[:300])

