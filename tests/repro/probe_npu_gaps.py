"""NPU-side probes on torch nightly + torch_npu master: which NPU-*/TORCH-*/TT-* gaps remain?
Run on the NPU box: python tests/repro/probe_npu_gaps.py (needs torchtitan for the TT-4 case)
"""

import torch
import torch_npu  # noqa: F401

dev = "npu:0"


def run(name, fn):
    try:
        r = fn()
        print(f"[OK ] {name}: {r}", flush=True)
    except Exception as e:  # noqa: BLE001
        first = str(e).strip().splitlines()[0][:200] if str(e).strip() else ""
        print(f"[ERR] {name}: {type(e).__name__}: {first}", flush=True)


print(
    "torch",
    torch.__version__,
    "| torch_npu",
    torch_npu.__version__,
    getattr(getattr(torch_npu, "version", None), "git_version", "?"),
)
run("basic add", lambda: (torch.randn(3, 4, device=dev) * 2).sum().item())
run("NPU-2 fake backend devices", lambda: torch.distributed.Backend.backend_capability.get("fake"))


def npu3():
    c = torch.view_as_complex(torch.randn(64, 32, 2, device=dev))
    idx = torch.randint(0, 64, (128,), device=dev)
    out = c[idx]
    ref = torch.view_as_complex(torch.view_as_real(c).cpu()[idx.cpu()])
    return torch.allclose(out.cpu(), ref)


run("NPU-3 complex[idx]", npu3)
run("NPU-6 zeros uint64", lambda: torch.zeros((2,), dtype=torch.uint64, device=dev).tolist())
print(
    "has_kernel _flash_attention_forward PrivateUse1:",
    torch._C._dispatch_has_kernel_for_dispatch_key("aten::_flash_attention_forward", "PrivateUse1"),
)


def npu1():
    from torch.nn.attention.varlen import varlen_attn

    T, N, D = 256, 4, 64
    q = torch.randn(T, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(T, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(T, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, 100, 256], device=dev, dtype=torch.int32)
    out = varlen_attn(q, k, v, cu, cu, 156, 156, window_size=(-1, 0))
    out.float().sum().backward()
    return tuple(out.shape), float(q.grad.float().norm())


run("NPU-1 stock varlen_attn fwd+bwd", npu1)


def flex():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    B, H, S, D = 1, 2, 128, 32
    q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16, requires_grad=True)

    def causal(b, h, qi, ki):
        return qi >= ki

    bm = create_block_mask(causal, B, H, S, S, device=dev)
    out = flex_attention(q, k, v, block_mask=bm)
    out.float().sum().backward()
    return tuple(out.shape)


run("TORCH-1 flex_attention eager fwd+bwd", flex)


def tt4():
    from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
    from torchtitan.models.common.linear import Linear

    V, Dm, T = 512, 64, 256
    with torch.device(dev):
        lm_head = Linear.Config(in_features=Dm, out_features=V, bias=False).build()
        lm_head.weight.data.normal_(std=0.02)
    loss = ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.Config(global_vocab_size=V)).build()
    loss.lm_head = lm_head
    h = torch.randn(T, Dm, device=dev, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, V, (T,), device=dev)
    gvt = torch.tensor(float(T), device=dev)
    out, _ = loss(h, labels, gvt)
    out.backward()
    return float(out)


run("TT-4 ChunkedLossWrapper 1-NPU no FSDP", tt4)
