# Kimi K3 on Ascend

**状态 🟢** —— 多模态 + KDA + MoE，2026-08-30 在 910B2 上跑通 10 步。**性能极低**（flex 走 eager，是 910B2 的硬件门），不要拿它做性能结论。

| | |
|---|---|
| 上游模型包 | `torchtitan.models.kimi_k3` |
| 我们的 recipe | `ascend_titan/models/kimi_k3/recipes.py` |
| 昇腾算子 | `kernels/kda.py`（KDA + causal conv1d）、`kernels/situ_glu.py`（ops-nn `aclnnSituGlu`） |
| 额外依赖 | `nvidia-cutlass-dsl`（有 aarch64 wheel；只 import 不执行） |
| 实测 | 单卡 10 步 `loss 4.10312 / grad_norm 4.5312`，显存 12.88 GiB，**tps 47**（LM 走昇腾融合 varlen 注意力，视觉塔保留 flex） |

## 1. 跑

```bash
pip install -c constraints/nightly.txt nvidia-cutlass-dsl   # kimi_k3 额外依赖
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
MODULE=ascend_titan.models.kimi_k3 CONFIG=kimi_k3_debugmodel_npu ./scripts/run_train.sh
```

## 2. 上游为什么在昇腾上跑不了，以及我们做了什么

| 上游的东西 | 为什么在昇腾上不行 | 我们的做法 |
|---|---|---|
| `KDAKernel.forward` | 显式要求 CUDA 张量 + Blackwell SM100/SM103，否则 `raise` | override 走 attn_gym 自己的 `impl="reference"`（文档写明是"differentiable eager PyTorch in FP32"） |
| `l2norm` | 来自 attn_gym 的 Triton 实现 | override 里用等价的 torch 实现（FP32 统计） |
| `causal_conv1d` | attn_gym 的 CuTeDSL（cutlass）内核 | `kernels/kda.py::ascend_causal_conv1d`：W 次移位读 + 边界掩码，dense 与 packed（`cu_seqlens`）都精确 |
| `attn_gym` 顶层 import `cutlass` | 曾被记为 TT-11 阻塞 | **不是阻塞**：`nvidia-cutlass-dsl` 有 aarch64 wheel，装上即可 import；所有会真正执行 cute 内核的节点都被 override 掉了 |
| 无条件 `torch.compile` 的 flex attention | Triton-Ascend 已在基线里、inductor 能编，但 document mask 的间接寻址在 910B2 上 lower 不了（`SubgraphLoweringException`） | shim `flex_attention_eager` 让 flex 走 eager；**特性探测**：探 `inductor_indirect_memory_mode`，换到 Ascend950 自动让路。掩码构建那两条 shim 2026-09-01 已删——装上 Triton-Ascend 后编译版构建器本身是好的 |
| `_apply_attention_residual` | 纯 torch，昇腾上直接能跑 | 不动。ops-transformer 的 `block_attn_res_update` 只有前向、没有反向，训练接不进来（见"待办"） |

一条 override 覆盖整棵 KDA 子树：`InnerKDA.Config` 持有 `kernel: KDAKernel.Config`，
torchtitan 拒绝"祖先已被别的 override 认领"的嵌套 override，所以嵌套的 kernel 配置在
工厂里用 `derive` 就地换掉。

## 3. recipe

`kimi_k3_debugmodel_npu` 的四条增量在 `recipes.py` 里逐条写着（一条 `# DELTA n` 一句理由），
读那个函数就知道我们换了什么：

| # | 换了什么 | 换成什么 | 为什么 | 什么时候能删 |
|---|---|---|---|---|
| 1 | 语言塔 6 个 `FlexAttention` 节点（layers 3/7/11/15/19/23，其余层是 KDA） | `VarlenAttention` | flex 掩码走 `create_block_mask`，是 `torch.compile` 的，910B2 编不出它的 document mask（间接寻址只有 Ascend950 才 lower） | 换到能 lower 间接寻址的芯片后自动失效（特性探测） |
| 2 | 上面转出来的 varlen 节点 | `kernels/attention.py` 昇腾融合注意力 | stock varlen 要 `aten::_flash_attention_forward`，torch_npu 没有（NPU-1） | NPU-1 合入后可换回 stock |
| 3 | 整棵 KDA 子树 | `kernels/kda.py`（attn_gym reference + torch 短卷积） | 上游内核要 CUDA/Blackwell | 有昇腾 KDA 融合算子后换实现，override 仍在 |
| 4 | `checkpoint.enable` | `False` | 冒烟运行不做 DCP I/O | DCP on NPU 是独立矩阵格 |

**没有动的**（同样是结论）：视觉塔的 `FlexAttention` 保持原样（它吃 `BlockMask`，转成
varlen 只会在塔内部炸一个更难懂的类型错误）、RoPE 走上游实现（kimi_k3 没有 `ComplexRoPE`
节点，用不上 NPU-3 的实数 cache override）、MoE 路由、`_apply_attention_residual`、loss、
并行策略、优化器全是上游默认。

| 函数 | 说明 |
|---|---|
| `kimi_k3_debugmodel_npu` | 参考路径：上表四条增量 |
| `kimi_k3_debugmodel_npu_fused` | 再叠加 ops-nn SiTU-GLU（需要 ops-nn run 包 + `cann_ops_nn_<vendor>`）。🟢 实测 `step: 10 loss 4.29434`，tps 48。首次调用要 JIT 编译算子（约 1 分钟） |

## 4. 性能

tps 45 / MFU 0.05%。三个原因，按影响排序：

1. **flex attention 走 eager**（掩码构建也是 eager），比编译版慢一个数量级以上——910B2 上编不了 document mask，是硬件门。
2. **KDA 走 attn_gym 的 reference 实现**，FP32 逐块 python 循环——等昇腾侧的 KDA 融合算子（fla-npu / ops-nn，M4 后续）。
3. MoE、SiTU-GLU 未启用融合算子。

按 P13，在这三条解决之前不记性能基线：数字会误导。

## 5. 待办

- KDA 融合算子（fla-npu 或 ops-nn）替换 reference 实现。
- ~~SiTU-GLU 端到端~~ **已跑通**（`loss 4.29434`）。需要 `patches/ops-nn/OPSNN-1-cxx-std.patch`
  （上游构建器硬编码 `-std=c++17`，torch 2.15 要 C++20）。
  之前"编译永远不结束"是**被 kill 的进程留下的 stale lock**，不是慢：
  `rm -f /root/.cache/torch_extensions/*/situ_glu/lock` 或换 `TORCH_EXTENSIONS_DIR` 即可。
- 融合 SiTU-GLU 目前**不改变吞吐**（tps 47 → 48）：瓶颈在 eager flex 与 reference KDA，不在 MoE 的激活。
- `attn_res`：ops-transformer 的 `block_attn_res_update` 缺反向，只能用于推理；要么等它补反向（gitcode 提 issue，P9），要么自己写反向。
- `flex_attention_eager` shim 的消失条件是**硬件**（能 lower 间接寻址的芯片），不是装包；它已按该开关做特性探测。
