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
在 3.2.2 里是**声明了但无法执行**的。

**触发路径**（不是我们主动选的）：torch_npu 的 inductor 对 "indirect load + sum" 模式会选
`simt_only`（`torch_npu/_inductor/codegen/ir.py:1717 define_npu_kernel_type`，
取值表在 `torch_npu/_inductor/config.py:319`）。FlexAttention 的 document mask 正是这个模式。

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

**影响**：910B2 上模型级 FlexAttention 编不出来 → 上游要求 CP 必须配 flex，所以
`torch.compile` + flex 与 CP 这两条路都停在这里。
