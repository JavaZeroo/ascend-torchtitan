import torch
import torch_npu  # noqa
from fla_npu.ops import ascendc as ac
print("torch", torch.__version__, "npu", torch_npu.__version__)

for dtype in (torch.float16, torch.bfloat16):
    for chunk in (64, 128):
        # bsnd: [B, S, H, D], D = chunk size
        A = torch.randn(1, 64, 4, chunk, dtype=dtype, device="npu:0").tril(-1).contiguous()
        try:
            out = ac.solve_tri(A, layout="bsnd")
            torch.npu.synchronize()
            print("solve_tri dtype=%s chunk(D)=%d -> OK shape=%s dtype=%s"
                  % (dtype, chunk, tuple(out.shape), out.dtype))
        except Exception as e:
            torch.npu.synchronize()
            print("solve_tri dtype=%s chunk(D)=%d -> FAIL %r" % (dtype, chunk, str(e)[:160]))
