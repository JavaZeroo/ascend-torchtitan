import pytest
import torch

pytestmark = [pytest.mark.npu, pytest.mark.titan]


@pytest.mark.parametrize("linear_beta", [None, 2.0])
def test_situ_glu_matches_upstream(linear_beta):
    from ascend_titan.kernels import situ_glu as K

    if not K._AVAILABLE:
        pytest.skip("ops-nn situ_glu not installed (scripts/build_kernels.sh ops-nn)")
    from torchtitan.models.kimi_k3.moe import _situ_glu

    dev = torch.device("npu:0")
    torch.manual_seed(0)
    g = torch.randn(64, 128, device=dev, dtype=torch.bfloat16, requires_grad=True)
    u = torch.randn(64, 128, device=dev, dtype=torch.bfloat16, requires_grad=True)
    ref = _situ_glu(g, u, 1.5, linear_beta)
    out = torch.ops.cann_ops_nn.situ_glu(
        torch.cat((g, u), -1), dim=-1, beta=1.5, linear_beta=linear_beta or 0.0, activate_left=True
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    ref.float().sum().backward()
    g2, u2 = g.grad.clone(), u.grad.clone()
    g.grad = u.grad = None
    out.float().sum().backward()
    torch.testing.assert_close(g.grad, g2, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(u.grad, u2, atol=5e-2, rtol=5e-2)
