# torch_npu —— 问题（归属：gitcode.com/Ascend/pytorch；算子内核在 gitcode.com/Ascend/op-plugin）

2026-08-29 在 torch 2.12.0 / torch_npu 2.12.0 **与** torch 2.13.0 / torch_npu 2.13.0rc1 上测得；**2026-08-30 全部在 NIGHTLY 基线（torch 2.15.0.dev20260812 + torch_npu master `15514cc70`）复现**（`outputs/nightly/probe_npu_gaps.py`）。本仓对这些问题一律不做 workaround（P1/P9）；修复见 `patches/torch_npu/`、`patches/op-plugin/`，状态见 `STATUS.md`。

## NPU-1 `aten::_flash_attention_forward` 没有 PrivateUse1 内核 —— `已修复（本地补丁）`
- 现象：`torch.nn.attention.varlen.varlen_attn` → `NotImplementedError: Could not run 'aten::_flash_attention_forward' ... PrivateUse1`（该 op 经自定义 op `torch_attn::_varlen_attn` 分发，随后调用 aten op）。
- 检查：`torch._C._dispatch_has_kernel_for_dispatch_key("aten::_flash_attention_forward", "PrivateUse1")` 在 2.12.0 与 2.13.0rc1 上均为 `False`。
- 影响：torchtitan 的 `VarlenAttention`（上游唯一的非 Flex LM 注意力）无法运行。ascend-torchtitan 改为提供基于 `npu_fusion_attention` 的 override（`ascend_titan.kernels.attention`）——这是内核替换，不是对本问题的绕过。
- 诉求：为 NPU 实现 `_flash_attention_forward/_backward`（varlen 签名：`cum_seq_q/k`、`max_q/k`、`window_size`），例如下沉到 `npu_fusion_attention` TND。这样上游 varlen 无需修改即可运行。

## NPU-2 `npu` 未注册到 `fake` 进程组后端 —— `已修复（本地补丁）`
- 现象：`--comm.mode=fake_backend`（torchtitan 的单设备干跑）→ `torch.distributed.broadcast` / 函数式 `all_reduce` 抛 `RuntimeError: No backend type associated with device type npu`。
- 检查：`torch.distributed.Backend.backend_capability["fake"]` → `['cpu', 'cuda', 'hpu', 'xpu']`（torch nightly 2.15.0.dev20260828 同样）。HPU 把自己加进去了，torch_npu 也应如此。
- 影响：所有基于 fake PG 的工具（配置干跑、上游 `fake_pg` 集成套件、流水线干跑）在 NPU 上不可用。
- 诉求：`import torch_npu` 时把 `"npu"` 注册进 `Backend.backend_capability["fake"]`，让 fake 后端接受 npu 张量。**发现（2026-08-30）**：torch_npu 已有该逻辑但只在 `torch_npu.testing._patch_backend_register_for_npu`（测试套件专用）；master 上 `_init/registry/distributed.py::register_distributed_backend_for_npu` 是死代码（无人调用），修复必须放进初始化时的 distributed patch 组。补丁：`patches/torch_npu/NPU-2-fake-process-group-npu.patch`。

## NPU-3 复数张量的高级索引失败（aclnnIndex 161002）—— `已修复（本地补丁，op-plugin）`
- 现象：`RuntimeError: index_high_dims_op_api: .../op_plugin/ops/opapi/IndexKernelNpuOpApi.cpp:40 NPU function error: call aclnnIndex failed, error code is 161002`。
- 最小复现：`c = torch.view_as_complex(torch.randn(64, 32, 2, device="npu")); c[torch.randint(0, 64, (128,), device="npu")]` 失败；同样的索引在实数张量上正常；`c.index_select(0, idx)` 与复数乘法正常。
- 影响：torchtitan 的 `ComplexRoPE`（llama3、deepseek_v3、gpt_oss、kimi、muse_glimmer 等的默认 RoPE）在 `rope.py:_reshape_for_broadcast` 的 `cache[positions]` 处失败——上游集成套件中 19 个用例。本仓用 `ascend_titan.kernels.rope`（实数缓存、数学相同）作为 override。
- 诉求：让 `aclnnIndex`/`index.Tensor` 支持 complex64（或在 op-plugin 中回退到 `index_select`）。

## NPU-4 `ArgSort` int32/int64 回退到 AiCpu（性能警告）—— `note`
- 在 `torch.distributed`/mesh 建立期间出现 `Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu`。不是正确性问题；记录下来避免重复排查。

## NPU-5 `torch_npu==2.13.0rc1` 传递性地要求 `attn-gym` ≥0.0.6？—— `check`
- 安装 torch_npu 2.13.0rc1 时拉入了 `attn-gym 0.0.6`，与 torchtitan 的 `==0.0.5` 冲突。需要确认是哪个包要求的；已在 `constraints/titan-deps.txt` 里 pin 回 0.0.5。

## NPU-6 `torch.zeros(dtype=torch.uint64)` / `zero_` 不支持 uint64 —— `已修复（本地补丁，op-plugin）`
- `aclnnInplaceZero` 的 dtype 列表不含 `DT_UINT64`（`ZerosLikeKernelNpuOpApi.cpp:26`，错误码 161002 / EZ1001）。
- 影响：`torch/nn/attention/varlen.py` 用 uint64 创建 rng_state 占位张量，即使有了 NPU-1 的内核，stock `varlen_attn` 仍在这里失败。本仓给 torch 准备了 TORCH-8 补丁（占位改 int64），torch_npu 侧的正解是 op-plugin 的 `zero_`/`fill_` 支持 uint64。

## NPU-7 torch_npu 的 `make_reduction` 覆盖缺少 torch 2.15 的 `strict_reduction` 关键字 —— `已修复（本地补丁）`
- 现象：任何含多维 `sum` 的 `torch.compile` 图在 NPU 上 lowering 失败：`torch._inductor.exc.InductorError: LoweringException: TypeError: make_reduction() got an unexpected keyword argument 'strict_reduction'`（torchtitan stock flex attention，`torch.compile(flex_attention)`，torch 2.15.0.dev20260812）。
- 根因：torch 2.15 给 `torch._inductor.lowering.make_reduction` 加了 keyword-only 的 `strict_reduction`，并从 `sum_` lowering 传入；同时 `_make_reduction_inner` 多了 `reduction_type=`、`Reduction.create` 多了 `strict_reduction=`。`torch_npu/_inductor/lowering.py:144` 用自己的 `make_reduction`（dump_fx_graph 支持）整体替换了它，签名停留在旧版。
- 修复：`patches/torch_npu/NPU-7-inductor-make-reduction-strict.patch`——接受 `strict_reduction`，并按 `inspect.signature` 探测把 `reduction_type` / `strict_reduction` 转发给 torch（各支持的 torch 版本都保持契约）。
- 诉求：torch_npu 的 inductor 覆盖层按 torch nightly 持续对齐；建议以 nightly 为 CI 基线。

## NPU-8 torch_npu 自动加载时经 `torch.distributed._tensor` 拖入 checkpoint / fsdp → 与 `spmd_types` 循环导入 —— `已修复（本地补丁）`
- 现象：`import spmd_types` 先于 `import torch` 的程序（torchtitan 的 `trainer.py:16` 就是）在 NPU 上直接死：`RuntimeError: Failed to load the backend extension: torch_npu` ← `AttributeError: partially initialized module 'spmd_types' has no attribute 'register_local_autograd_function' (most likely due to a circular import)`。
- 根因（两条链）：(1) `torch_npu/distributed/tensor/{_matrix_ops,_moe_ops,_attention}.py` 从已弃用的 `torch.distributed._tensor` 导入，该包的 `__init__` 导入 `_shards_wrapper` → `torch.distributed.checkpoint` → `fsdp`；(2) `_init/patches/distributed_patches.py::_apply_sharded_grad_scaler_patch` 为替换 `ShardedGradScaler` 直接导入 `torch.distributed.fsdp.sharded_grad_scaler`。nightly 的 `_fsdp_state.py` 在模块级 `import spmd_types`；因为 torch_npu 是在 `import torch` 内部被自动加载的，这两条链变成了每一次 `import torch` 的副作用。
- 修复：`patches/torch_npu/NPU-8-dtensor-public-imports.patch`——(1) 改用公开的 `torch.distributed.tensor{,.experimental,._dtensor_spec}`；(2) `ShardedGradScaler` 替换延后到用户导入该模块之后（新增 `torch_npu/utils/_import_hooks.py::run_after_import` meta-path 后置导入钩子）。UT 断言 `import torch` 后 `torch.distributed.{checkpoint,fsdp}` 不在 `sys.modules` 且替换仍生效。
- 诉求：torch_npu 的 import 期足迹应最小化（自动加载 = 全局副作用）。


## NPU-9：`NPUCombinedScheduling` 未构造父类的子调度器

`torch_npu/_inductor/codegen/npu_combined_scheduling.py` 的 `NPUCombinedScheduling` 继承
`CUDACombinedScheduling`，但 `__init__` 直接调 `BaseScheduling.__init__`，跳过父类构造函数，
于是 `_cutlass_scheduling` / `_rocm_cpp_scheduling` / `_cutedsl_scheduling` 以及 torch 2.15 新增的
`_nv_universal_gemm_scheduling` 都不存在。它只覆写了三个方法，其余继承方法无条件解引用这些属性。

后果：NPU 上编译 FlexAttention 抛
`AttributeError: 'NPUCombinedScheduling' object has no attribute '_nv_universal_gemm_scheduling'`。
`can_fuse_reduction_epilogue` 是 torch 2.15 新增的方法——**torch 每加一个这样的方法，就会重现一次**。

修复：`__init__` 先 `CUDACombinedScheduling.__init__`，再覆盖 NPU 自己的子调度器；父类那些
`is_*_template()` 在 NPU 上都返回 False，调度行为不变。

issue [#4447](https://gitcode.com/Ascend/pytorch/issues/4447) · PR [!45534](https://gitcode.com/Ascend/pytorch/merge_requests/45534)

## NPU-11：`NPUTritonKernel` 在非 Ascend950 上也请求 SIMT 编译模式

`torch_npu/_inductor/codegen/triton.py:931`（`NPUTritonKernel.add_npu_inductor_meta`）无条件写

```python
inductor_meta["npu_kernel_type"] = str(NPUKernelType.SIMD_SIMT_MIX)
```

项目里其它四处选内核类型的地方全都带芯片门——`triton.py:1017`（`NPUIndexTritonKernel.__init__`）、
`ir.py:1566`、`ir.py:1858`、`config.py:319`，非 A5 时 `inductor_indirect_memory_mode` 是 `None`。
**931 是唯一的漏网**，也是 910B2 上唯一会请求 SIMT 的路径。

后果不是慢，是编译直接失败。`simd_simt_mix` 进了 `inductor_meta`，autotune 就生成 SIMT_ONLY
候选（`runtime/triton_heuristics.py:3360`、`runtime/fasta_autotune.py:407`），
`compile_mode="simt_only"` 传到 Triton-Ascend 后端，后者追加 `--enable-triton-ir-compile`
与 `--pure-simt`（`triton/backends/ascend/compiler.py:1076-1077`）。已发布的
`bishengir-compile` 没有一个实现这些参数（见 TA-1），于是用户看到

```
bishengir-compile: Unknown command line argument '--pure-simt'.
MLIRCompilationError: [ConvertLinalgRToBinary] encounters error
```

这看起来像 CANN 太旧，会把人引去升级 toolkit，而升级不解决任何问题。

**触发路径**：内存访问线性化失败后回退到 `NPUNoLinearTritonScheduling`
（`codegen/scheduling.py:56`，它的 `kernel_type` 就是 `NPUTritonKernel`）。
FlexAttention 的 document mask 正落在这里。抓到的 autotune 候选带
`--num-warps=64/32/16/8`，坐实来源是 autotune。

修复：`patches/torch_npu/pending/NPU-11-simt-kernel-type-chip-gate.patch`——按 `is_ascend950` 分流，
非 A5 用 `SIMD`。

**修完不代表能跑**（已实测）：SIMT 参数消失，改走 SIMD，然后停在
`LLVM ERROR: Failed to obtain op buffer shape size which should be static.`。
indirect load + sum 在 910B2 上本来就没有可 lower 的路径。这条补丁的价值是**把报错变诚实**，
不是解锁功能。P13：不要拿它当"flex 能编了"的证据。

未提交（等 TA-1 一起）。
