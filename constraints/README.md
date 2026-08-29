# constraints

| 文件 | 作用 |
|---|---|
| `torchtitan.sha` | 固定的 torchtitan commit。 |
| `npu.txt` | **NEXT** track（默认）：torch 2.13.0 + torch_npu 2.13.0rc1。纯 pip 语法。与 `torchtitan.sha` 一起是"我们在哪个版本上"的唯一事实来源。 |
| `npu-stable.txt` | **STABLE** track：torch 2.12.0 + torch_npu 2.12.0（GA）。同一个 torchtitan SHA。用 `CONSTRAINTS=constraints/npu-stable.txt ./scripts/install.sh` 选择。 |
| `titan-deps.txt` | torchtitan 的依赖列表，**去掉** `attn-gym[linear]` extra（它会拉 `nvidia-cutlass-dsl[cu13]`）。 |

为什么不直接 `pip install torchtitan`？它的 `requirements.txt` 硬 pin 了 `attn-gym[linear]==0.0.5`，`[linear]` extra 会往 aarch64/NPU 机器上装 CUDA-only 的 wheel。`scripts/install.sh` 用 `--no-deps` 装 torchtitan，再装这份列表。

attn_gym 的 KDA 有纯 torch 的 `naive` 回退，所以没有该 extra 时 kimi_k3 仍能跑（慢）。

两条 track 每晚都验证；一条绿一条红，按定义就是 torch 版本问题（见 docs/baseline.md）。

**安装时永远带 `-c`。** 不带 `-c constraints/<track>.txt` 的 `pip install torchvision` 会升级 `torch`，破坏 torch_npu 的 ABI（`undefined symbol ... deleteNode`、"Failed to load the backend extension: torch_npu"）。`scripts/install.sh` 已经做对了；手工安装时每次都要传 constraints 文件。
