# constraints

| 文件 | 作用 |
|---|---|
| `nightly.txt` | **NIGHTLY**（默认、唯一门禁，ADR-006）：torch nightly（日期 = torch_npu master `requirements_2.15.txt` 的 pin）+ 源码构建的 torch_npu（`==2.15.0` 匹配 `2.15.0+git<sha>`）。 |
| `torch_npu.sha` | torch_npu（gitcode Ascend/pytorch）master 的 commit；`scripts/build_torch_npu.sh` 检出它构建（子模块 op-plugin 由该 commit 决定）。 |
| `torchtitan.sha` | 固定的 torchtitan commit。 |
| `npu.txt` | **RELEASE**（信息性，不门禁）：PyPI 最新 torch_npu（2.13.0rc1）+ torch 2.13.0。 |
| `npu-stable.txt` | 已废弃（torch 2.12 / torch_npu 2.12.0）；保留一个发布周期后删除。 |
| `npu-triton.txt` | RELEASE + Triton-Ascend（inductor / fla-npu 实验）；待改为只含增量的 `extras/triton.txt`。 |
| `titan-deps.txt` | torchtitan 的依赖列表，**去掉** `attn-gym[linear]` extra（它会拉 `nvidia-cutlass-dsl[cu13]`）。 |

三元组（`nightly.txt` 的 torch 行 + `torch_npu.sha` + `torchtitan.sha`）是"我们在哪个版本上"的唯一事实来源（P11），一起升级、一个 PR 附矩阵结果（P5）。

为什么不直接 `pip install torchtitan`？它的 `requirements.txt` 硬 pin 了 `attn-gym[linear]==0.0.5`，`[linear]` extra 会往 aarch64/NPU 机器上装 CUDA-only 的 wheel。`scripts/install.sh` 用 `--no-deps` 装 torchtitan，再装这份列表。

attn_gym 的 KDA 有纯 torch 的 `naive` 回退，所以没有该 extra 时 kimi_k3 仍能跑（慢）。

**安装时永远带 `-c`。** 不带 `-c constraints/<track>.txt` 的 `pip install torchvision` 会升级 `torch`，破坏 torch_npu 的 ABI（`undefined symbol ... deleteNode`、"Failed to load the backend extension: torch_npu"）。`scripts/install.sh` 已经做对了（NIGHTLY 下 torchvision 取 nightly 的 CPU 包且 `--no-deps`）；手工安装时每次都要传 constraints 文件。
