<!-- 手工网格，不是 tools.matrix 的用例集：上游 tests/integration_tests 里没有 kimi_k3 的并行用例，
     所以这些格子在能力矩阵里一直是 ⚪。torchtitan 13da2d77c，NIGHTLY track，8 卡。-->
# kimi_k3 的并行空白格（2026-09-04）

`models/registry.py` 里 kimi_k3 的 R2 长期是「只有 debugmodel：R1–R8 一条都没取」，
README 的场景总表把 TP / PP / EP / CP 合成一格 ❓。这里把它们逐格测掉。

命令形如：

```bash
MODULE=ascend_titan.models.kimi_k3 CONFIG=<cfg> ASCEND_RT_VISIBLE_DEVICES=<cards> NPU=<n> \
  ./scripts/run_train.sh --training.steps 2 --checkpoint.no-enable --parallelism.<...>
```

| 场景 | config | 卡 | 结果 | 证据 |
|---|---|:--:|:--:|---|
| 单卡 | `kimi_k3_debugmodel`（stock） | 1 | 🔴 | `ValueError: x must be a contiguous CUDA FP16, BF16, or FP32 tensor` |
| 单卡 | `kimi_k3_debugmodel_npu` | 1 | 🟢 | golden 逐位复现（`check_golden.sh`） |
| FSDP2 × 2 | `kimi_k3_debugmodel_npu` | 2 | 🟢 | `12.40092 → 11.23449` |
| FSDP2 × 8 | `kimi_k3_debugmodel_npu` | 8 | 🟢 | `12.40720 → 10.68999` |
| HSDP 2 × 4 | `kimi_k3_debugmodel_npu` | 8 | 🟢 | `12.45858 → 10.63290` |
| TP2 | `kimi_k3_debugmodel_npu` | 2 | 🔴 | `NotImplementedError: Kimi K3 currently supports FSDP2 data parallelism only; disable tensor parallel.` |
| PP2 | `kimi_k3_debugmodel_npu` | 2 | 🔴 | `RuntimeError: Pipeline Parallel is enabled but kimi_k3 does not support pipelining` |
| EP2 | `kimi_k3_debugmodel_npu` | 2 | 🔴 | `NotImplementedError: ... disable expert parallel.` |
| CP2 | `kimi_k3_debugmodel_npu` | 2 | 🔴 | `ValueError: Context Parallel requires parallelism.spmd_backend='spmd_types', got 'partial_dtensor'.` |
| `--compile.enable` | `kimi_k3_debugmodel_npu` | 1 | 🔴 | `NotImplementedError: Kimi K3 does not support model compilation yet.` |

## 五个红格全部是上游按设计不支持，与昇腾无关

`torchtitan/models/kimi_k3/parallelize.py:36` 的 `parallelize_kimi_k3` 开头就把
TP / PP / CP / EP 四个维度一起挡掉：

```python
raise NotImplementedError(
    "Kimi K3 currently supports FSDP2 data parallelism "
    f"only; disable {', '.join(unsupported_parallelisms)}."
)
```

同一个函数下面还有 `if compile_config.enable and "model" in compile_config.components:
raise NotImplementedError("Kimi K3 does not support model compilation yet.")`。
代码里**没有任何设备判断**，CUDA 上跑同样报这四条。

CP 那格撞的是更早的一道门：`kimi_k3_debugmodel` 的配置把 `spmd_backend` 钉成
`partial_dtensor`（上游注释：`TODO: Kimi K3 has no spmd_types annotations yet`），
而 CP 要求 `spmd_types`。所以 CP 连 `parallelize_kimi_k3` 都没走到。两道门都是上游的。

**这些不该记成红格**，也不该算进「让 kimi_k3 支持更多并行」的收益里——那件事的前提是
上游先支持。上游 main 已经在 2026-08 之后落地了 EP（`[Kimi K3] Expert Parallelism support`
#4314），等 `constraints/torchtitan.sha` 升上去之后 EP 这一格要重测。

## stock 那格红在 KDA，不在并行

`kimi_k3_debugmodel` 零 override 时死在 `x must be a contiguous CUDA FP16, BF16, or FP32
tensor`——KDA 子树里的 `l2norm` 是 attn_gym 的 Triton 实现，只认 CUDA 张量；再往里
`KDAKernel.forward` 还有一条 Blackwell SM100/SM103 的断言。这正是 `kernels/kda.py`
那条 override 存在的理由，和 qwen3.5 的 GDN 是同一类。

## 过程中的自查

三个绿格都只跑 2 步，只能证明**能跑**，不能证明数值对。多卡的数值证据在单卡 golden
（逐位冻结）+ `tests/unit/test_kernel_kda.py` 的算子级对拍上；FSDP2 各路的 loss 之间
不能互相比较（数据分片不同），也不能拿来和单卡 golden 比。
