"""bwd 取证：fla-npu 整链 vs ascend_chunk_gdn 的梯度，0.8B 维度 (K=V=128)。"""
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


def grads(fn, *args):
    tensors = [a.detach().clone().requires_grad_() for a in args]
    out = fn(*tensors)
    out.square().sum().backward()
    return [t.grad for t in tensors]


def rel(a, b):
    a = a.detach().float(); b = b.detach().float()
    return (a - b).abs().max().item() / b.abs().max().item()


def main():
    torch.npu.set_device(0)
    dtype = torch.bfloat16
    print("=== gradient check: K=V=128 ===")
    for (B, H, T, K, V) in [(1, 8, 128, 128, 128), (1, 16, 256, 128, 128)]:
        q, k, v, g, beta = make_inputs(B, H, T, K, V, dtype, "npu:0", seed=5)
        ours_grad = grads(lambda *a: ascend_chunk_gdn(*a, chunk_size=64), q, k, v, g, beta)
        torch.npu.synchronize()
        fla_grad = grads(lambda *a: fla(*a, 64), q, k, v, g, beta)
        torch.npu.synchronize()
        names = ["dq", "dk", "dv", "dg", "dbeta"]
        for n, og, fg in zip(names, ours_grad, fla_grad):
            print("  B=%d H=%d T=%d %-6s rel=%.3e" % (B, H, T, n, rel(fg, og)))


if __name__ == "__main__":
    main()

