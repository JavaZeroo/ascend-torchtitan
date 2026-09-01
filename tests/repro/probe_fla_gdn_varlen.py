"""varlen (cu_seqlens) path for fla-npu 整链 at K=V=128, and grouped-head (Hk<Hv).
This decides whether we can drop the per-document Python loop (the biggest perf win).
"""
import sys
from pathlib import Path
import torch
import torch_npu  # noqa
from ascend_titan.kernels.gdn import ascend_chunk_gdn

FLA_ROOT = Path("/data/ljb/projects/create-ascend-titian/flash-linear-attention-npu")
if str(FLA_ROOT) not in sys.path:
    sys.path.insert(0, str(FLA_ROOT))
from examples.flash_gated_delta_rule import flash_gated_delta_rule


def make_inputs(B, H, T, K, V, dtype, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, H, T, K, generator=gen), dim=-1).to(dtype).to(device)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, K, generator=gen), dim=-1).to(dtype).to(device)
    v = torch.randn(B, H, T, V, generator=gen).to(dtype).to(device)
    g = (-0.5 * torch.rand(B, H, T, generator=gen)).float().to(device)
    beta = torch.sigmoid(torch.randn(B, H, T, generator=gen)).float().to(device)
    return q, k, v, g, beta


def rel(a, b):
    a = a.detach().float(); b = b.detach().float()
    return (a - b).abs().max().item() / b.abs().max().item()


def main():
    torch.npu.set_device(0)
    dtype = torch.bfloat16

    # grouped heads: Hk=2, Hv=4 (q/k have 2 heads, v has 4)
    print("=== grouped heads Hk=2 Hv=4 K=V=128 dense ===")
    Qk = torch.nn.functional.normalize(torch.randn(1, 2, 128, 128), dim=-1).to(dtype).npu()
    Kk = torch.nn.functional.normalize(torch.randn(1, 2, 128, 128), dim=-1).to(dtype).npu()
    Vv = torch.randn(1, 4, 128, 128).to(dtype).npu()
    g = (-0.5 * torch.rand(1, 4, 128)).float().npu()
    beta = torch.sigmoid(torch.randn(1, 4, 128)).float().npu()
    try:
        o, _ = flash_gated_delta_rule(Qk, Kk, Vv, g.transpose(1,2).contiguous(),
                                      beta.transpose(1,2).contiguous(),
                                      scale=128**-0.5, use_qk_l2norm_in_kernel=False,
                                      chunk_size=64)
        torch.npu.synchronize()
        want = ascend_chunk_gdn(Qk.repeat_interleave(2,1), Kk.repeat_interleave(2,1),
                                Vv, g, beta, chunk_size=64)
        print("  grouped-head forward OK, rel=%.3e" % rel(o.transpose(1,2).contiguous(), want))
    except Exception as e:
        print("  grouped-head FAIL: %r" % str(e)[:200])

    # varlen: pack 3 sequences of lengths 96, 64, 96 into one [B=1,T=256]
    print("=== varlen cu_seqlens=[0,96,160,256] K=V=128 ===")
    cu = torch.tensor([0, 96, 160, 256], dtype=torch.int64)
    T = 256
    q = torch.nn.functional.normalize(torch.randn(1, 16, T, 128), dim=-1).to(dtype).npu()
    k = torch.nn.functional.normalize(torch.randn(1, 16, T, 128), dim=-1).to(dtype).npu()
    v = torch.randn(1, 16, T, 128).to(dtype).npu()
    g = (-0.5 * torch.rand(1, 16, T)).float().npu()
    beta = torch.sigmoid(torch.randn(1, 16, T)).float().npu()
    try:
        o, _ = flash_gated_delta_rule(q, k, v, g.transpose(1,2).contiguous(),
                                      beta.transpose(1,2).contiguous(),
                                      scale=128**-0.5, use_qk_l2norm_in_kernel=False,
                                      chunk_size=64, cu_seqlens=cu)
        torch.npu.synchronize()
        # reference: per-segment torch kernel
        pieces = []
        for b, e in [(0,96),(96,160),(160,256)]:
            pieces.append(ascend_chunk_gdn(q[:,:,b:e], k[:,:,b:e], v[:,:,b:e],
                                           g[:,:,b:e], beta[:,:,b:e], chunk_size=64))
        want = torch.cat(pieces, dim=2)
        print("  varlen forward OK, rel=%.3e" % rel(o.transpose(1,2).contiguous(), want))
    except Exception as e:
        print("  varlen FAIL: %r" % str(e)[:200])


if __name__ == "__main__":
    main()

