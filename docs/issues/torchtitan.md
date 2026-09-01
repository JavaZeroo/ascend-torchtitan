# torchtitan —— 问题 / 诉求（归属：pytorch/torchtitan）

在 torchtitan `13da2d77c`（2026-08-29）测得。"nightly-only" = 需要任何正式版 torch（≤ 2.13.0）都没有的特性。

## TT-1 core 导入链中无条件 `import triton` —— `draft`
- `torchtitan/models/common/token_dispatcher.py:17 → distributed/minimal_async_ep/api.py:30 → minimal_async_ep/kernels.py:10 import triton`。没有 `triton` wheel 的环境 `import torchtitan.trainer` 直接失败，尽管 MinimalAsyncEP 是可选特性。
- 诉求：在内核函数内部懒加载 triton（像 `overrides/fused_swiglu.py` 那样放在 try/except 后面），或用 `find_spec` 保护。本仓的处理：`constraints/titan-deps.txt` 加了 `triton`（纯 Python wheel；它*不是* Triton-Ascend）。

## TT-10 树内 Triton override 与 DistMuon 写死 CUDA —— `info`
- `overrides/fused_swiglu.py` / `fused_grouped_experts`：`Could not run 'torchtitan::silu_and_mul'`（Triton 内核只注册了 CUDA）；`kimi_k2_5` 的 `DistMuon requires one CUDA device per process`。这些是上游明确的 CUDA-only 组件（override README 也这么定位），记录为 `TT-KERNEL` / `TT-CUDA`，昇腾替代属于 L1（M3+）。

## TT-11 kimi_k3 在模块导入时就需要 `cutlass`（经 attn_gym 的 cute 后端）—— `draft`
- `models/kimi_k3/kda.py:15 → attn_gym.linear.kda.short_conv → activations.py:54 import cutlass`。没有 `nvidia-cutlass-dsl`（CUDA-only）的环境连 `import torchtitan.models.kimi_k3` 都失败，任何 kimi_k3 recipe/override 都无法构建；attn_gym 的 `naive` 回退在这里帮不上忙，因为失败发生在 import 期。
- 诉求：attn_gym 把 cute 后端改为懒加载（或 kimi_k3 用 `find_spec` 门控），让无 CUDA 环境退回 naive 路径。对本仓：SituGLU/KDA override 在此之前只能在 op 级验证（`tests/npu/test_kernel_situ_glu.py`），无法接入 kimi_k3 模型（矩阵：kimi_k3 = 🔴 DEP）。

## TT-9 `overrides/fused_mla.py` 使用 nightly-only 的 `torch._C.Tag.inplace` —— `draft`
- `AttributeError: type object 'torch._C.Tag' has no attribute 'inplace'`；影响 `deepseek_v3_fused_mla_swiglu_fsdp+ep`。
- 诉求：`getattr(torch.Tag, "inplace", None)` 保护，或按 torch 版本门控该 override。

## TT-3 `create_block_mask(separate_full_blocks=...)` 仅在 nightly —— `info`
- `models/common/attention.py::create_attention_mask` 自 #4106（2026-08-11）。torch 2.13.0 有，2.12.x 没有。TORCH-1 存在期间在 NPU 上无关紧要。

## TT-4 ChunkedLossWrapper 的 backward 在 NPU 上报 "data is not allocated yet" —— `investigating`
- `components/loss.py` 显式驱动 FSDP 的 lm_head `unshard()`（自 #4143，2026-08-13）。在 torch 2.12.0/2.13.0 + torch_npu 上 backward 抛 `RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet`，开不开 AC 都一样；普通 `CrossEntropyLoss` 正常。归因仍开放（torch 版本的 FSDP 语义 vs NPU 分配器/事件处理）。矩阵格 `loss/chunked` = 🔴。

## TT-6 kimi_k3 的 attention residual 是自由函数（没有 `Configurable` 节点）—— `ask, deferred`
- `models/kimi_k3/model.py:135 _apply_attention_residual`；override 它需要替换整个 `KimiK3TransformerBlock.Config`。抽成带 `sharding_config` 的 `Module` 也解决上游自己的 `TODO: Add TP Support`。等 kimi_k3 稳定（2026-08-24 落地）。

## TT-7 LM 的 `sdpa` inner attention 已移除 —— `info`
- `models/common/config_utils.py:97`。对非 CUDA 后端的后果：上游不存在 eager 注意力路径。记录在案；override 机制是官方答案。


## compiled-block-mask：flex 路径三处无条件 `torch.compile`，没有开关

`torchtitan/models/common/attention.py`::

    _compiled_create_block_mask = torch.compile(create_block_mask)
    class FlexAttention:
        _compiled_flex_attn: ClassVar[Callable] = torch.compile(flex_attention, options=inductor_configs)

`torchtitan/models/common/vision_encoder.py`::

    compiled_create_block_mask = torch.compile(create_block_mask)

三处都在 import 时求值，且都不看 `config.compile.enable`。昇腾上有两种后果
（2026-09-01 装上 Triton-Ascend 之后重新测的）：

* **确定性模式**（`--debug.deterministic`，也就是录 golden 的方式）：torch_npu 的
  inductor 拒绝未经认证的 autotune benchmark，掩码构建直接抛
  `RuntimeError: In the deterministic mode of Inductor, ...`。
* **任何模式下的 flex attention 本体**：它的 document mask 要 index 一个 segment-id
  张量，而间接寻址的 lowering 只在 Ascend950 上启用，910B2 抛
  `SubgraphLoweringException`。

两者 eager 下都工作正常（昇腾上实测 fwd + bwd 通过）。

**上游 ask**：让这三处尊重一个开关（`compile.enable`，或一个 `FlexAttention` 上的
`compile: bool = True` 字段），或者惰性编译并在编译失败时退回 eager。

按 P10，我们不向 github.com/pytorch 提 issue/PR，只在本仓记录，并用 shim
`ascend_titan/compat/shims/flex_attention_eager.py` 在本地绕开（特性探测：
triton 一有可用后端就自动让路）。
