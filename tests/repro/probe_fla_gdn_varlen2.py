"""干净取证（避开 K=256 崩溃）：(1) varlen 反向 (2) 预展开后的分组头 Hk=16→Hv=32。"""
import sys
from pathlib import Path
import torch
import torch_npu  # noqa
from ascend_titan.kernels.gdn import ascend_chunk_gdn

FLA_ROOT = Path("/data/ljb/projects/create-ascend-titian/flash-linear-attention-npu")
if str(FLA_ROOT) not in sys.path:
    sys.path.insert(0, str(FLA_ROOT))
from examples.flash_gated_delta_rule import flash_gated_delta_rule


def fla(q, k, v, g, beta, chunk, cu=None):
    o, _ = flash_gated_delta_rule(q, k, v, g.transpose(1,2).contiguous(),
                                  beta.transpose(1,2).contiguous(), scale=128**-0.5,
                                  use_qk_l2norm_in_kernel=False, chunk_size=chunk, cu_seqlens=cu)
    return o.transpose(1, 2).contiguous()


def rel(a, b):
    a = a.detach().float(); b = b.detach().float()
    return (a - b).abs().max().item() / b.abs().max().item()


def grads(fn, *args):
    ts = [a.detach().clone().requires_grad_() for a in args]
    fn(*ts).square().sum().backward()
    return [t.grad for t in ts]


def main():
    torch.npu.set_device(0)
    dt = torch.bfloat16

    print("=== (1) varlen bwd K=V=128, cu=[0,96,160,256] ===")
    cu = torch.tensor([0, 96, 160, 256], dtype=torch.int64)
    gen = torch.Generator(device="cpu").manual_seed(21)
    q = torch.nn.functional.normalize(torch.randn(1,16,256,128, generator=gen), dim=-1).to(dt).npu()
    k = torch.nn.functional.normalize(torch.randn(1,16,256,128, generator=gen), dim=-1).to(dt).npu()
    v = torch.randn(1,16,256,128, generator=gen).to(dt).npu()
    g = (-0.5*torch.rand(1,16,256, generator=gen)).float().npu()
    beta = torch.sigmoid(torch.randn(1,16,256, generator=gen)).float().npu()
    def ours(*ts):
        qq,kk,vv,gg,bb = ts
        return torch.cat([ascend_chunk_gdn(qq[:,:,b:e],kk[:,:,b:e],vv[:,:,b:e],gg[:,:,b:e],bb[:,:,b:e],chunk_size=64) for b,e in [(0,96),(96,160),(160,256)]], dim=2)
    og = grads(ours, q, k, v, g, beta)
    torch.npu.synchronize()
    fg = grads(lambda *ts: fla(*ts, 64, cu=cu), q, k, v, g, beta)
    torch.npu.synchronize()
    for n, a, b in zip(["dq","dk","dv","dg","dbeta"], og, fg):
        print("  %-6s rel=%.3e" % (n, rel(b, a)))

    print("=== (2) grouped after expand Hk=16→Hv=32 (4B shape) ===")
    gen = torch.Generator(device="cpu").manual_seed(33)
    Hk, Hv = 16, 32
    qk = torch.nn.functional.normalize(torch.randn(1,Hk,128,128, generator=gen), dim=-1).to(dt).npu()
    kk = torch.nn.functional.normalize(torch.randn(1,Hk,128,128, generator=gen), dim=-1).to(dt).npu()
    vv = torch.randn(1,Hv,128,128, generator=gen).to(dt).npu()
    gg = (-0.5*torch.rand(1,Hv,128, generator=gen)).float().npu()
    bb = torch.sigmoid(torch.randn(1,Hv,128, generator=gen)).float().npu()
    rpt = Hv // Hk
    qe, ke = qk.repeat_interleave(rpt,1), kk.repeat_interleave(rpt,1)
    try:
        got = fla(qe, ke, vv, gg, bb, 64)
        torch.npu.synchronize()
        want = ascend_chunk_gdn(qe, ke, vv, gg, bb, chunk_size=64)
        print("  grouped-after-expand OK rel=%.3e" % rel(got, want))
    except Exception as e:
        print("  grouped-after-expand FAIL %r" % str(e)[:120])


if __name__ == "__main__":
    main()

