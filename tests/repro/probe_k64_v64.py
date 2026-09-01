import torch
import torch_npu  # noqa
from ascend_titan.kernels.gdn import ascend_chunk_gdn
from ascend_titan.kernels.gdn_fla import fused_chunk_gdn

def inputs(batch, heads, seq, K, V, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)
    def randn(*s):
        return torch.randn(*s, generator=gen, dtype=torch.float32).to("npu:0")
    def unit(*s):
        return torch.nn.functional.normalize(randn(*s), dim=-1).to(dtype)
    return (unit(batch, heads, seq, K), unit(batch, heads, seq, K),
            randn(batch, heads, seq, V).to(dtype),
            -torch.rand(batch, heads, seq, generator=gen).to("npu:0"),
            torch.rand(batch, heads, seq, generator=gen).to("npu:0"))

for dtype in (torch.bfloat16, torch.float16):
    for K, V in ((64,64),(64,128),(128,128),(128,256)):
        q,k,v,g,beta = inputs(1,4,256,K,V,dtype,seed=11)
        try:
            got = fused_chunk_gdn(q,k,v,g,beta, chunk_size=64)
            want = ascend_chunk_gdn(q,k,v,g,beta, chunk_size=64)
            torch.npu.synchronize()
            diff = (got-want).abs().max().item()
            rel = (diff / want.abs().max().item())
            print("fwd dtype=%s K=%3d V=%3d -> max_abs=%.4e rel=%.2e  %s" %
                  (dtype, K, V, diff, rel, "OK" if rel < 0.05 else "MISMATCH"))
        except Exception as e:
            torch.npu.synchronize()
            print("fwd dtype=%s K=%3d V=%3d -> FAIL %r" % (dtype, K, V, str(e)[:140]))

print("--- backward K=64 V=64 bf16 ---")
base = inputs(1,4,256,64,64,torch.bfloat16,seed=7)
def grads(fn):
    q,k,v,g,beta = [t.clone().requires_grad_() for t in base]
    out = fn(q,k,v,g,beta, chunk_size=64)
    out.square().sum().backward()
    return [q.grad,k.grad,v.grad,g.grad,beta.grad]
gf = grads(fused_chunk_gdn); gt = grads(ascend_chunk_gdn)
names = ["q","k","v","g","beta"]
for n, a, b in zip(names, gf, gt, strict=True):
    print("  grad %s -> max_abs=%.4e  rel=%.2e" % (n, (a-b).abs().max().item(),
          ((a-b).abs().max()/b.abs().max()).item()))
