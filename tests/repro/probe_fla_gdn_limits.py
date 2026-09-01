"""收尾取证：(1) 全链 K 下界（V 固定 128）；(2) varlen 反向对拍。"""
import sys
from pathlib import Path
import torch
import torch_npu  # noqa
from ascend_titan.kernels.gdn import ascend_chunk_gdn

FLA_ROOT = Path("/data/ljb/projects/create-ascend-titian/flash-linear-attention-npu")
if str(FLA_ROOT) not in sys.path:
    sys.path.insert(0, str(FLA_ROOT))
from examples.flash_gated_delta_rule import flash_gated_delta_rule


def make(B, H, T, K, V, dtype, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, H, T, K, generator=gen), dim=-1).to(dtype).npu()
    k = torch.nn.functional.normalize(torch.randn(B, H, T, K, generator=gen), dim=-1).to(dtype).npu()
    v = torch.randn(B, H, T, V, generator=gen).to(dtype).npu()
    g = (-0.5 * torch.rand(B, H, T, generator=gen)).float().npu()
    beta = torch.sigmoid(torch.randn(B, H, T, generator=gen)).float().npu()
    return q, k, v, g, beta


def fla(q, k, v, g, beta, chunk, cu=None):
    o, _ = flash_gated_delta_rule(q, k, v, g.transpose(1,2).contiguous(),
                                  beta.transpose(1,2).contiguous(), scale=q.shape[-1]**-0.5,
                                  use_qk_l2norm_in_kernel=False, chunk_size=chunk, cu_seqlens=cu)
    return o.transpose(1, 2).contiguous()


def rel(a, b):
    a = a.detach().float(); b = b.detach().float()
    return (a - b).abs().max().item() / b.abs().max().item()


def main():
    torch.npu.set_device(0)
    dt = torch.bfloat16
    print("=== (1) full-chain K sweep, V=128, H=8, T=128 ===")
    for K in (32, 64, 128, 256):
        q, k, v, g, beta = make(1, 8, 128, K, 128, dt, seed=7)
        try:
            got = fla(q, k, v, g, beta, 64)
            torch.npu.synchronize()
            want = ascend_chunk_gdn(q, k, v, g, beta, chunk_size=64)
            print("  K=%3d OK rel=%.3e" % (K, rel(got, want)))
        except Exception as e:
            print("  K=%3d FAIL %r" % (K, str(e)[:80]))

    print("=== (2) varlen backward, K=V=128 ===")
    cu = torch.tensor([0, 96, 160, 256], dtype=torch.int64)
    q, k, v, g, beta = make(1, 16, 256, 128, 128, dt, seed=11)
    names = ["dq","dk","dv","dg","dbeta"]

    def grads(fn, *args):
        ts = [a.detach().clone().requires_grad_() for a in args]
        out = fn(*ts).square().sum()
        out.backward()
        return [t.grad for t in ts]

    # ours per-segment
    def ours(*ts):
        qq,kk,vv,gg,bb = ts
        return torch.cat([ascend_chunk_gdn(qq[:,:,b:e],kk[:,:,b:e],vv[:,:,b:e],gg[:,:,b:e],bb[:,:,b:e],chunk_size=64) for b,e in [(0,96),(96,160),(160,256)]], dim=2)
    og = grads(ours, q, k, v, g, beta)
    torch.npu.synchronize()
    fg = grads(lambda *ts: fla(*ts, 64, cu=cu), q, k, v, g, beta)
    torch.npu.synchronize()
    for n, a, b in zip(names, og, fg):
        print("  varlen-bwd %-6s rel=%.3e" % (n, rel(b, a)))


if __name__ == "__main__":
    main()

