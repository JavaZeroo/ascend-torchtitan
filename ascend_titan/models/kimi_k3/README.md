# Kimi K3 on Ascend

**状态 🔴 — 阻塞在 `cutlass`（TT-11 / DEP-CUTLASS）。** 昇腾侧的融合算子已经写好了。

| | |
|---|---|
| 上游模型包 | `torchtitan.models.kimi_k3` |
| 我们的 recipe | `ascend_titan/models/kimi_k3/recipes.py` |
| 昇腾算子 | `ascend_titan/kernels/situ_glu.py`（ops-nn `aclnnSituGlu`） |
| 阻塞 | `ModuleNotFoundError: No module named 'cutlass'` |

## 1. 现在会发生什么

```bash
$ python -c "import torchtitan.models.kimi_k3"
ModuleNotFoundError: No module named 'cutlass'
```

链路：kimi_k3 在模块级导入 attn-gym 的 cute 后端 → `cutlass`（NVIDIA CUTLASS 的 Python DSL，CUDA-only）。
记录在 `docs/issues/torchtitan.md` 的 TT-11，修复思路存 `patches/evidence/torchtitan/`——
**只读证据，永不应用**，也不向 github.com/pytorch 提 issue/PR（P10）。

## 2. 已经就绪的部分

`kernels/situ_glu.py` 把上游那段 ~8 个 elementwise 算子的 SiTU-GLU
（`beta * tanh(gate/beta) * sigmoid(gate) * [linear_beta * tanh(up/linear_beta)]`）
换成 ops-nn 的 AscendC 融合算子，前向 `situ_glu` / 反向 `situ_glu_grad` 各一个 kernel，
用 `torch.library.custom_op` 包成一个可微算子（带 `register_fake`，可 compile）。

它以 `KimiFeedForward.Config` 为 override 目标——所以只等模型包能 import。

ops-nn 是**真正可选**的加速包（要装 run 包 + JIT 构建），因此走
`_probe.optional_module()` + WARNING 降级（ADR-004）；这与 torch_npu 的硬依赖规则（P14）不冲突。

```bash
./scripts/build_kernels.sh ops-nn      # 装/构建 cann_ops_nn
```

## 3. recipe（就绪，未验证）

| 函数 | 说明 |
|---|---|
| `kimi_k3_debugmodel_npu` | 上游 `kimi_k3_debugmodel` + SiTU-GLU override + 关 checkpoint |

## 4. 解开阻塞的路径

1. 上游把 attn-gym cute 后端改成惰性导入（最干净，但我们只能记录，不能提 PR，P10）。
2. 本地：在 `patches/evidence/torchtitan/` 的证据补丁基础上，等上游自己修。
3. 若 cutlass 只在某个注意力后端里用到，可考虑请上游把它移到 `attn_backend` 分支内
   （`docs/upstream-tracking.md` 的上游 ask）。
