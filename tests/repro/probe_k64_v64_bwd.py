import torch
import torch_npu  # noqa
import ascend_titan.kernels.gdn_fla as m
from ascend_titan.kernels.gdn import ascend_chunk_gdn

m._FUSED_VALUE_DIMS = (64, 128, 256)

def inputs(batch, heads, seq, K, V, dtype, seed=0):
    gen = torch.Generator(device="npu").manual_seed(seed)
    def randn(*s):
        return torch.randn(*s, generator=gen, dtype=torch.float32, device="npu:0")
    def unit(*s):
        return torch.nn.functional.normalize(randn(*s), dim=-1).to(dtype)
    return (unit(batch, heads, seq, K), unit(batch, heads, seq, K),
            randn(batch, heads, seq, V).to(dtype),
            -torch.rand(batch, heads, seq, generator=gen, device="npu:0"),
            torch.rand(batch, heads, seq, generator=gen, device="npu:0"))

base = inputs(1, 4, 256, 64, 64, torch.bfloat16, seed=7)
def grads(fn):
    q,k,v,g,beta = [t.clone().requires_grad_() for t in base]
    out = fn(q,k,v,g,beta, chunk_size=64)
    out.square().sum().backward()
    return [q.grad,k.grad,v.grad,g.grad,beta.grad]

print("gate(V=64) now:", m._fused_shape_gate(base[0], base[1], base[2], 64))
gf = grads(m.fused_chunk_gdn); gt = grads(ascend_chunk_gdn)
torch.npu.synchronize()
names = ["q","k","v","g","beta"]
for n,a,b in zip(names, gf, gt, strict=True):
    md = (a-b).abs().max().item()
    rel = md / b.abs().max().item()
    print("grad %s max_abs=%.4e rel=%.2e %s" % (n, md, rel, "OK" if rel<0.05 else "MISMATCH"))
