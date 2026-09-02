# triton-ascend 的问题

托管在 `gitcode.com/Ascend/triton-ascend`——在 P10 的授权范围内，可以提 issue / PR。
正文可直接粘贴。状态见 `docs/issues/STATUS.md`（P11：这里不复述状态）。

---

## TA-1：3.2.2 的 Python 前端发出的 SIMT 编译参数，自带的 bishengir-compile 不认识

**版本**：triton-ascend 3.2.2（wheel 自带完整 triton 发行版）、torch 2.15.0.dev20260812、
torch_npu master（2.15.0+git15514cc）、CANN 9.1.0、Ascend 910B2。

**现象**：任何走到 SIMT 编译路径的内核都编不出来，报

```
bishengir-compile: Unknown command line argument '--enable-triton-ir-compile'.
  Did you mean '--enable-triton-kernel-compile'?
bishengir-compile: Unknown command line argument '--pure-simt'.
triton.compiler.errors.MLIRCompilationError:
  [ConvertLinalgRToBinary] encounters error
```

**根因：同一个 wheel 内部不自洽。** Python 侧在 SIMT 路径上无条件加这两个参数：

```
triton/backends/ascend/compiler.py:1074  if opt.force_simt_only:
                              :1076          _compile_option_list += ["--enable-triton-ir-compile"]
                              :1077          _compile_option_list += ["--pure-simt"]
```

而 wheel 自带的编译器二进制**完全不支持 SIMT**：

```bash
B=<site-packages>/triton/backends/ascend/bishengir/bin/bishengir-compile
$B --help | grep -ci simt                      # -> 0
$B --help | grep -i 'triton-ir-compile'        # -> 空；它只有 --enable-triton-kernel-compile
```

即 `compile_mode="simt_only"`（`compiler.py:1015-1017` 列为受支持的取值之一）
在 3.2.2 里是**声明了但无法执行**的。二进制里连隐藏选项都没有：
`strings bishengir-compile | grep -ci simt` 与 `--help-hidden | grep -ci simt` 都是 0，
SIMT 根本没编进去。

**不是"我们的 CANN 太旧"**：这台机器上三个 `bishengir-compile` 副本全是 1.2.0，
SIMT 参数都是 0 个——

| 来源 | 版本 | `--help` 里 SIMT 参数 |
|---|---|--:|
| triton-ascend 3.2.2 自带 | 1.2.0（AscendNPU-IR `042c923d`，2026-07-31） | 0 |
| CANN 9.1.0 `/usr/local/Ascend` | 1.2.0（`8796a8ac`，2026-07-31） | 0 |
| CANN 9.1.0 `/data/ljb/CANN` | 1.2.0（`8796a8ac`，2026-07-29） | 0 |

也就是说 `simt_only` 需要一个比任何已发布组件都新的 bishengir，而它没有随包发出来。

**触发路径**（实测，不是我们主动选的）：910B2 上 `torch_npu/_inductor/codegen/triton.py:931`
在 `NPUTritonKernel.add_npu_inductor_meta` 里**无条件**写

```python
inductor_meta["npu_kernel_type"] = str(NPUKernelType.SIMD_SIMT_MIX)
```

它忽略了构造函数刚按芯片算好的 `self.npu_kernel_type`（`triton.py:1018-1020`：非 A5 是
`SIMD`），也没有像同项目其它四处那样带 `is_ascend950` / `inductor_indirect_memory_mode`
门（`triton.py:1018`、`ir.py:1566`、`ir.py:1858`、`config.py:319`）。同文件里
`NPUIndexTritonKernel` 走的是 `triton.py:1885` 的 `str(self.npu_kernel_type)`，是对的——
**931 这一处是唯一的漏网**。

`SIMD_SIMT_MIX` 一旦进 `inductor_meta`，autotune 就会生成 SIMT_ONLY 候选
（`runtime/triton_heuristics.py:3360`、`runtime/fasta_autotune.py:407`），于是
`compile_mode="simt_only"` 传到 triton-ascend，`--pure-simt` 就发出去了。
实测抓到的候选串带着 `--num-warps=64/32/16/8`，正是 autotune 在扫。

走到 `NPUTritonKernel` 的是 `NPUNoLinearTritonScheduling`（`codegen/scheduling.py:56`），
即内存访问线性化失败后的回退路径——FlexAttention 的 document mask 正好落在这里。

**修了这一行会怎样（已实测）**：SIMT 参数确实消失，改走 SIMD
（`--enable-triton-kernel-compile=true`），然后停在更深的一层：

```
[ConvertLinalgRToBinary] encounters error:
LLVM ERROR: Failed to obtain op buffer shape size which should be static.
```

也就是说 **910B2 上这个 kernel 本来就没有能走通的编译路径**：indirect load + sum 需要
SIMT，而 SIMT 只在 A5 上开。修 931 不能让它跑起来，只能把报错从"参数不认识"变成诚实的
"这条路 lower 不了"。这条仍然值得修——现在的报错把人引向 CANN 版本，是错的方向。

**最小复现**（不需要模型，两行）：

```bash
B=<site-packages>/triton/backends/ascend/bishengir/bin/bishengir-compile
$B --help | grep -c simt        # 0：二进制没有任何 SIMT 选项
grep -n 'pure-simt' <site-packages>/triton/backends/ascend/compiler.py   # 1077：Python 却在发它
```

端到端复现（上游 torchtitan，无任何本仓改动）：

```bash
python -m torchtitan.train --module torchtitan.models.qwen3_5.config_registry \
    --config qwen35_debugmodel --training.steps 2
```

**期望**：要么 wheel 里带上支持 `--pure-simt` / `--enable-triton-ir-compile` 的
bishengir-compile，要么 Python 侧在这些参数不被支持时给出可操作的报错
（而不是让 `CalledProcessError` 冒到用户面前），并说明 `simt_only` 在该版本不可用。

**影响（2026-09-02 重新界定，比我们最初判断的小）**：TA-1 只挡**编译版**的
`create_block_mask`。把掩码构建改走 eager 之后，能力矩阵里被它挡住的格子是 **0 个**：

| 用例 | 9-01 | 9-02 重测 |
|---|:--:|:--:|
| `qwen3_5_fsdp+tp+varlen_attn+per_op_sac` | 🔴 TA-1 | 🟢 53s |
| `qwen3_5_moe_fsdp+tp+ep` | 🔴 TA-1 | 🟢 72s |
| `qwen3_5_moe_fsdp+tp+ep+pp` | 🔴 TA-1 | 🟢 59s |

所以这条**不是阻塞**，是一条正确性/可用性缺陷：它让 `simt_only` 这个声明为受支持的
compile_mode 实际不可执行，并且把失败暴露成一条指向 CANN 版本的误导性报错
（`Unknown command line argument`，会让人去升级 toolkit，而升级不解决任何问题）。

它会重新变成阻塞的场景：换到能 lower 间接寻址的芯片（Ascend950）之后，掩码构建与
flex 都该走编译路径，那时 SIMT 是真要用的。
