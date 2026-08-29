import pytest
import torch

pytestmark = [pytest.mark.npu, pytest.mark.titan]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_matches_upstream(dtype):
    from torchtitan.models.common.nn_modules import RMSNorm

    from ascend_titan.kernels.rms_norm import npu_rms_norm

    dev = torch.device("npu:0")
    cfg = RMSNorm.Config(normalized_shape=256, eps=1e-5)
    with torch.device(dev):
        ref = cfg.build()
        new = npu_rms_norm(cfg).build()
    torch.manual_seed(0)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(256, device=dev))
        new.weight.copy_(ref.weight)
    x = torch.randn(64, 256, device=dev, dtype=dtype, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    y1, y2 = ref(x), new(x2)
    torch.testing.assert_close(y1, y2, atol=2e-2 if dtype == torch.bfloat16 else 1e-5, rtol=2e-2)
    (y1.float() ** 2).sum().backward()
    (y2.float() ** 2).sum().backward()
    torch.testing.assert_close(
        x.grad, x2.grad, atol=5e-2 if dtype == torch.bfloat16 else 1e-4, rtol=5e-2
    )
    torch.testing.assert_close(ref.weight.grad, new.weight.grad, atol=5e-2, rtol=5e-2)
