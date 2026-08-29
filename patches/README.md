# patches —— 上游修复方案

两类目录，政策完全不同（P9 / P10）：

| 目录 | 归属 | 政策 |
|---|---|---|
| `torch_npu/` | gitcode.com/Ascend/pytorch | **在途修复**。每个补丁 = 一个 `fix/<ID>` 分支的 `format-patch`，头部注明问题编号、根因、验证命令与结果；提交 PR 后在头部加 PR 链接。`WITH_PATCHES=1 scripts/build_torch_npu.sh` 把它们叠加进 NIGHTLY 构建（默认要求 PR 链接存在；提 PR 前的本地验证用 `REQUIRE_PR_LINK=0`）。**上游合入即删**，同时升 `constraints/torch_npu.sha`。 |
| `op-plugin/` | gitcode.com/Ascend/op-plugin（torch_npu 的算子子模块） | 同上，应用于 `third_party/op-plugin`。 |
| `evidence/{torchtitan,pytorch}/` | github.com/pytorch/*（**不提上游**，P10） | **只读证据**：记录"如果上游要修应该怎么修"，**永不应用于安装路径**，不被任何脚本读取。只保留在 NIGHTLY 基线上仍然存在的问题（纯版本差的补丁已删：TT-2 / TT-8 / TT-9）。 |

当前内容：

| 补丁 | 问题 | 状态 |
|---|---|---|
| `torch_npu/NPU-1-flash-attention-privateuse1.patch` | `aten::_flash_attention_forward/_backward` 无 NPU 内核 → stock `varlen_attn` 不可用 | 已提交：issue [4439](https://gitcode.com/Ascend/pytorch/issues/4439) · PR [!45527](https://gitcode.com/Ascend/pytorch/merge_requests/45527) |
| `torch_npu/NPU-2-fake-process-group-npu.patch` | fake 进程组不接受 npu 张量 | 已提交：issue [4438](https://gitcode.com/Ascend/pytorch/issues/4438) · PR [!45526](https://gitcode.com/Ascend/pytorch/merge_requests/45526) |
| `torch_npu/NPU-7-inductor-make-reduction-strict.patch` | torch_npu 的 `make_reduction` 覆盖缺 torch 2.15 的 `strict_reduction` 关键字 → 含多维 sum 的 compile 图 lowering 失败 | 已提交：issue [4440](https://gitcode.com/Ascend/pytorch/issues/4440) · PR [!45528](https://gitcode.com/Ascend/pytorch/merge_requests/45528) |
| `torch_npu/NPU-8-dtensor-public-imports.patch` | torch_npu 自动加载时经 `torch.distributed._tensor` 拖入 checkpoint/fsdp → 先 `import spmd_types` 再 `import torch` 的程序循环导入 | 已提交：issue [4441](https://gitcode.com/Ascend/pytorch/issues/4441) · PR [!45529](https://gitcode.com/Ascend/pytorch/merge_requests/45529) |
| `op-plugin/NPU-3-index-complex.patch` | `aclnnIndex` 不支持复数 → 复数张量高级索引失败（ComplexRoPE） | 已提交：issue [466](https://gitcode.com/Ascend/op-plugin/issues/466) · PR [!5800](https://gitcode.com/Ascend/op-plugin/merge_requests/5800) |
| `op-plugin/NPU-6-zero-unsigned.patch` | `aclnnInplaceZero` 不支持 uint16/32/64 → `varlen_attn` 的 uint64 rng_state 占位失败 | 已提交：issue [467](https://gitcode.com/Ascend/op-plugin/issues/467) · PR [!5801](https://gitcode.com/Ascend/op-plugin/merge_requests/5801) |
| `evidence/pytorch/0001-TORCH-7-*.patch` | `opcheck` autograd 检查不认 privateuse1 | 证据；本仓 NPU 测试用数值梯度代替 |
| `evidence/pytorch/0002-TORCH-8-*.patch` | `varlen.py` 用 uint64 占位 | 证据；昇腾侧正解是 NPU-6（op-plugin） |
| `evidence/torchtitan/0004-TT-1-*.patch` | core 无条件 `import triton` | 证据；本仓装纯 Python `triton` |
| `evidence/torchtitan/0005-TT-11-*.patch` | kimi_k3 导入需 `cutlass` | 证据；kimi_k3 在无 CUDA 环境保持 🔴 DEP |

生成方式：在 `../ascend-pytorch` / `../ascend-op-plugin` 的 `fix/<ID>` 分支上提交（含 UT），`git format-patch -1 --stdout fix/<ID> > patches/<dir>/<ID>-<slug>.patch`。
