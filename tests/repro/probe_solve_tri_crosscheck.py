"""三方互证：我们的 ``_unit_lower_inverse`` vs fla-npu 的 AscendC ``npu_solve_tri``。

背景：qwen3.5 的发散最终定位在我们自己写的这个求逆上（Neumann doubling 在 C=64 溢出
fp32）。当时只有一个实现，没有第二个可比对象，导致连续三次归因错误。fla-npu 提供了
独立的 AscendC 实现，正好补上这个缺口。

符号约定不同，对拍时要抵消：
  我们   ``_unit_lower_inverse(A)`` = ``(I - A)^-1``
  fla    ``npu_solve_tri(A)``       = ``(I + A)^-1``
所以喂给 fla 的是 ``-A``。

布局映射：我们的 ``[N, C, C]`` 就是 N 个独立的 C×C 块，对应 fla 的 BNSD
``[B, H, T, BT]`` 取 ``B=1, H=N, T=BT=C``（每个 (b,h) 恰好一个 chunk）。

判据是对 float64 参考解的**相对**误差——不是两个实现互相比（两边一起错就看不出来），
也不是绝对误差：iid 正态的严格下三角阵，``(I-A)^-1`` 的元素随 C 指数增长，C=64 时参考解
本身就能到 1e10 量级，绝对误差没有意义。
"""

import argparse
import time

import torch
import torch_npu  # noqa: F401  设备注册

from ascend_titan.kernels.gdn import _unit_lower_inverse

try:
    # 必须在任何 NPU 算子执行之前导入：fla_npu 在 __init__ 里设 ASCEND_CUSTOM_OPP_PATH，
    # 而 CANN 的算子注册表一旦初始化就不再重读这个变量。延迟到第一次调用时才 import，
    # aclnnSolveTriGetWorkspaceSize 会返回 161001（NULLPTR，查不到算子）。实测确认。
    from fla_npu.ops import ascendc as ascendc_ops
except Exception as _fla_import_error:  # 探针：没装就只报我们自己的结果
    ascendc_ops = None
    FLA_IMPORT_ERROR = _fla_import_error


def reference_inverse(strictly_lower: torch.Tensor) -> torch.Tensor:
    """``(I - A)^-1``，float64 直接解，作为两个实现共同的参照。

    在 CPU 上算：昇腾没有 fp64 硬件（``aclnnEye`` 对 DT_DOUBLE 直接报 EZ1001），而参照解
    本来就应该与被测设备无关。
    """
    size = strictly_lower.shape[-1]
    lower = strictly_lower.detach().cpu().double()
    eye = torch.eye(size, dtype=torch.float64)
    return torch.linalg.solve_triangular(eye - lower, eye.expand_as(lower), upper=False)


def make_input(count, size, dtype, scale, device, kind):
    """严格下三角输入。

    ``kind="normal"`` 是 iid 正态，压测用；它比真实 GDN 苛刻得多。
    ``kind="gdn"`` 模仿真实形态：门控衰减让元素随行列距离指数衰减，谱半径远小于 1。
    """
    generator = torch.Generator(device="cpu").manual_seed(42)
    raw = torch.randn(count, size, size, generator=generator, dtype=torch.float32) * scale
    if kind == "gdn":
        rows = torch.arange(size).view(-1, 1)
        decay = torch.exp(-0.5 * (rows - torch.arange(size).view(1, -1)).clamp(min=0).float())
        raw = raw * decay
    return raw.tril(-1).to(device=device, dtype=dtype)


def timed(fn, tensor, repeat):
    fn(tensor)
    torch.npu.synchronize()
    begin = time.perf_counter()
    for _ in range(repeat):
        result = fn(tensor)
    torch.npu.synchronize()
    return result, (time.perf_counter() - begin) / repeat * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--kind", default="gdn", choices=["gdn", "normal"])
    args = parser.parse_args()

    torch.npu.set_device(0)
    dtype = getattr(torch, args.dtype)
    strictly_lower = make_input(args.count, args.chunk, dtype, args.scale, "npu", args.kind)
    reference = reference_inverse(strictly_lower)
    magnitude = reference.abs().max().item()

    print(
        f"N={args.count} C={args.chunk} dtype={args.dtype} "
        f"scale={args.scale} kind={args.kind}  max|ref|={magnitude:.3e}"
    )

    def relative(result):
        return (result.detach().cpu().double() - reference).abs().max().item() / magnitude

    ours, ours_ms = timed(_unit_lower_inverse, strictly_lower, args.repeat)
    print(f"  ours          rel={relative(ours):.3e}   {ours_ms:8.3f} ms")

    if ascendc_ops is None:
        print(f"  fla_npu 不可用，跳过对拍：{FLA_IMPORT_ERROR}")
        return
    if dtype is torch.float32:
        # solve_tri_def.cpp 把输入 dtype 写死成 {DT_FLOAT16, DT_BF16}；fp32 会以 161001 失败。
        print("  fla-npu       不支持 fp32（算子只声明 fp16/bf16），跳过")
        return

    def call_fla(tensor):
        # 符号相反，且要 BNSD [1, N, C, C]
        return ascendc_ops.npu_solve_tri(
            (-tensor).unsqueeze(0).contiguous(), cu_seqlens=None, chunk_indices=None, layout="bnsd"
        ).squeeze(0)

    theirs, theirs_ms = timed(call_fla, strictly_lower, args.repeat)
    print(f"  fla-npu       rel={relative(theirs):.3e}   {theirs_ms:8.3f} ms")

    disagreement = (
        (ours.detach().cpu().double() - theirs.detach().cpu().double()).abs().max().item()
    )
    print(f"  两实现之差    rel={disagreement / magnitude:.3e}")


if __name__ == "__main__":
    main()
