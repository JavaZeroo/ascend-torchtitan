import pytest
import torch

pytestmark = [pytest.mark.npu, pytest.mark.titan]


def test_fused_swiglu_matches_upstream_feedforward():
    from torchtitan.models.common.feed_forward import FeedForward
    from torchtitan.models.common.linear import Linear

    from ascend_titan.kernels.swiglu import npu_fused_swiglu

    dev = torch.device("npu:0")
    cfg = FeedForward.Config(
        w1=Linear.Config(in_features=64, out_features=128, bias=False),
        w2=Linear.Config(in_features=128, out_features=64, bias=False),
        w3=Linear.Config(in_features=64, out_features=128, bias=False),
    )
    with torch.device(dev):
        ref = cfg.build()
        new = npu_fused_swiglu(cfg).build()
    torch.manual_seed(0)
    for m in (ref, new):
        for p_ in m.parameters():
            torch.nn.init.normal_(p_, std=0.05)
    # load the stock layout into the fused module through its state_dict hooks
    new.load_state_dict(ref.state_dict())
    assert set(new.state_dict()) == set(ref.state_dict())
    x = torch.randn(32, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    y1, y2 = ref(x), new(x2)
    torch.testing.assert_close(y1, y2, atol=3e-2, rtol=3e-2)
    y1.float().sum().backward()
    y2.float().sum().backward()
    torch.testing.assert_close(x.grad, x2.grad, atol=5e-2, rtol=5e-2)
