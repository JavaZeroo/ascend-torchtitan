"""Decisive fwd retrieval: fla-npu 整链(examples 权威编排) vs 我们的 ascend_chunk_gdn，
在真实 0.8B 维度 (K=V=128, Hk=16, Hv=16) 上。dense 路径。
"""
import sys
import time
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


def fla(q, k, v, g, beta, chunk):
    o, _ = flash_gated_delta_rule(
        q, k, v,
        g.transpose(1, 2).contiguous(),
        beta.transpose(1, 2).contiguous(),
        scale=(q.shape[-1] ** -0.5),
        use_qk_l2norm_in_kernel=False,
        chunk_size=chunk,
    )
    return o.transpose(1, 2).contiguous()


def rel(a, b):
    a = a.detach().float(); b = b.detach().float()
    return (a - b).abs().max().item() / b.abs().max().item()


def main():
    torch.npu.set_device(0)
    dtype = torch.bfloat16
    print("=== forward: K=V=128 (0.8B dims) ===")
    for (B, H, T, K, V) in [(1, 16, 256, 128, 128), (1, 16, 512, 128, 128)]:
        q, k, v, g, beta = make_inputs(B, H, T, K, V, dtype, "npu:0", seed=42)
        want = ascend_chunk_gdn(q, k, v, g, beta, chunk_size=64)
        torch.npu.synchronize()
        got = fla(q, k, v, g, beta, 64)
        torch.npu.synchronize()
        print("  B=%d H=%d T=%d K=%d V=%d rel=%.3e" % (B, H, T, K, V, rel(got, want)))

    q, k, v, g, beta = make_inputs(1, 16, 512, 128, 128, dtype, "npu:0", seed=1)
    for name, fn in (("ours", lambda: ascend_chunk_gdn(q, k, v, g, beta, chunk_size=64)),
                     ("fla", lambda: fla(q, k, v, g, beta, 64))):
        fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        print("  timing %s: %.3f ms" % (name, (time.perf_counter()-t0)/5*1e3))


if __name__ == "__main__":
    main()

