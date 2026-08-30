# Kimi K3 on Ascend

**状态 🟢** —— 多模态 + KDA + MoE，2026-08-30 在 910B2 上跑通 10 步。**性能极低**，Triton-Ascend 到位前不要拿它做性能结论。

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
| 三处无条件 `torch.compile`（flex 掩码 ×2、flex attention ×1） | 昇腾上 inductor 需要 Triton-Ascend，否则 `0 active drivers` | shim `flex_block_mask_eager` 换回上游未编译的函数；**特性探测**：triton 一有可用后端就自动让路 |
| `_apply_attention_residual` | 纯 torch，昇腾上直接能跑 | 不动。ops-transformer 的 `block_attn_res_update` 只有前向、没有反向，训练接不进来（见"待办"） |

一条 override 覆盖整棵 KDA 子树：`InnerKDA.Config` 持有 `kernel: KDAKernel.Config`，
torchtitan 拒绝"祖先已被别的 override 认领"的嵌套 override，所以嵌套的 kernel 配置在
工厂里用 `derive` 就地换掉。

## 3. recipe

| 函数 | 说明 |
|---|---|
| `kimi_k3_debugmodel_npu` | 参考路径：`npu_minimal` + KDA override + 关 checkpoint |
| `kimi_k3_debugmodel_npu_fused` | 再叠加 ops-nn SiTU-GLU（需要 ops-nn run 包 + `cann_ops_nn_<vendor>`）。⚠️ 首次调用要 JIT 编译算子，很慢；本轮未跑完 |

## 4. 性能

tps 45 / MFU 0.05%。三个原因，按影响排序：

1. **flex attention 走 eager**（掩码构建也是 eager），比编译版慢一个数量级以上——等 Triton-Ascend（M5）。
2. **KDA 走 attn_gym 的 reference 实现**，FP32 逐块 python 循环——等昇腾侧的 KDA 融合算子（fla-npu / ops-nn，M4 后续）。
3. MoE、SiTU-GLU 未启用融合算子。

按 P13，在这三条解决之前不记性能基线：数字会误导。

## 5. 待办

- KDA 融合算子（fla-npu 或 ops-nn）替换 reference 实现。
- SiTU-GLU：`cann_ops_nn_ascend_titan_nn` 已装进 NIGHTLY venv（需要 `patches/ops-nn/OPSNN-1-cxx-std.patch`：
  上游构建器硬编码 `-std=c++17`，torch 2.15 要 C++20），算子注册成功、override 能落到 12 个
  `KimiFeedForward` 节点；但**首次调用的 JIT 编译在本轮没跑完**（>20 分钟未结束），
  所以 `kimi_k3_debugmodel_npu_fused` 还没有端到端结果。下次直接跑
  `python /tmp/situprobe.py` 那种单算子调用把 JIT 缓存烤热，再跑训练。
- `attn_res`：ops-transformer 的 `block_attn_res_update` 缺反向，只能用于推理；要么等它补反向（gitcode 提 issue，P9），要么自己写反向。
- Triton-Ascend 到位后复测，并撤掉 `flex_block_mask_eager` shim（它会自己让路）。
