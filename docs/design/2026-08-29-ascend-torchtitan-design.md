# ascend-torchtitan —— 设计

状态：已采纳 2026-08-29（v1，M0/M1 实测后修订）。维护者：ascend-torchtitan maintainers。
配套文档：`docs/PRINCIPLES.md`、`docs/roadmap.md`、`docs/adr/`、`docs/upstream-tracking.md`、`docs/baseline.md`。

## 1. 目标与非目标

**目标**
1. `pip install` 本包 + `scripts/install.sh` ⇒ torchtitan 在昇腾 NPU 上以 eager 模式运行。
2. 承载昇腾性能工作——融合算子（KDA、causal_conv1d、SituGLU、attn_res、fusion attention、norm）、并行策略（moonep EP、CP）、图模式（torchair）——全部是**可插拔、按需开启**的组件。
3. **不 fork torchtitan**，并能跟上它每月约 170 个 commit 的节奏。
4. 保持可读：新人从一个 recipe 文件就能看出一次运行用了哪些昇腾组件。

**非目标**
- vendor 无关的中间层（ADR-005）。
- 绕过 torch_npu 缺陷（P1）。
- 跟踪 torchtitan 的 `experiments/`（上游自己也不让 experiments 阻塞 core）。

## 2. 上游已经提供的扩展点（对照 torchtitan @ 13da2d77c 验证）

| 扩展点 | 证据 | 不打补丁就能做什么 |
|---|---|---|
| `--module <完整路径>` | `torchtitan/config/manager.py:126-135` | 树外 `config_registry` 模块 → 我们的 recipe |
| `config.build()` 通过 `Config._owner` 反查 | `train.py:43`、`docs/extension.md` | 需要时子类化 `Trainer`/`Trainer.Config` |
| `override.imports` | `torchtitan/config/override.py`、`overrides/README.md` | 替换**任意** `Configurable.Config` 节点；`fqns` 精确定位；per-node 冲突检测；`derive()` 抗字段漂移；per-entry kwargs |
| `ModelSpec` 可调用字段 | `protocols/model_spec.py:33` | 替换 `parallelize_fn`、`pipelining_fn`、`state_dict_adapter` |
| 设备探测 | `tools/utils.py:54-60`、`distributed/utils.py:498` | `device_type` 来自 `torch._utils._get_available_device_type()`（privateuse1 → npu）；通信后端来自 `Backend.default_device_backend_map`（→ hccl） |
| 已有保护的 CUDA 路径 | `distributed/cudagraph.py:342`、`configs.py:296-306`、量化模块 | CUDA graph 在非 CUDA 上自动退回 eager；`compile.backend` 是普通字符串；float8/mx/nvfp4 是 opt-in |

上游 override README 直接写明了意图：vendor 内核应放在外部包里、由 `override.imports` 激活；`experiments/README.md` §4 禁止树内放 vendor 代码。我们的布局就是上游要求的形状。

## 3. 架构

```
python -m ascend_titan.train  ──setup()──►  导入 torch_npu + L0 shim  ──►  torchtitan.train.main()
                                                                                  │
      recipe (L3)：cfg = 上游 registry 函数(); 增量; cfg.override.imports=[L1 模块]
                                                                                  │
                     Trainer.__init__: apply_overrides → 导入 ascend_titan.kernels.* (L1)
                                        ModelSpec.parallelize_fn ← ascend_titan.parallel (L2)
                                        compile.backend="torchair" ← ascend_titan.graph (L2)
```

| 层 | 包 | 机制 | 健康度指标 |
|---|---|---|---|
| L0 compat | `ascend_titan.compat` | 受治理的 monkeypatch 注册表；只由 `setup()` 应用 | shim 数量 → 0 |
| L1 kernels | `ascend_titan.kernels` | `@override` 工厂 + `torch.library.custom_op` | 每个算子：opcheck + 对齐测试 |
| L2 parallel / graph | `ascend_titan.parallel`、`.graph` | `ModelSpec` 可调用字段、`compile.backend` | 端到端性能基线 |
| L3 recipes | `ascend_titan.recipes` | 上游 registry 函数 + 增量；`transforms.npu_baseline` 通用变换 | 带基线的 🟢 格子数 |
| L4 tools | `ascend_titan.tools` | doctor、matrix 扫描、provenance、对齐辅助 | — |

### 3.1 引导与导入顺序约束（F4）
`torchtitan/tools/utils.py:60` 在模块导入时求值 `device_type`。因此 `setup()` 必须先于任何 torchtitan 导入；`ascend_titan.train` 的存在只为保证这个顺序。M0 实测：torch 通过 `torch._import_device_backends` 自动加载 torch_npu（显式导入前 privateuse1 已是 `npu`），所以 F4 的风险只剩 shim 的应用顺序。

`import ascend_titan` 无副作用（有测试），因为 torchtitan 会在 `Trainer.__init__` 内部导入我们的 L1 模块。

### 3.2 Configurable 节点判据（P6）
对每个想融合的计算：上游是否有一个 `forward` 包含该计算的 `Configurable.Config` 节点？

| 计算 | 上游节点 | 结论 |
|---|---|---|
| inner attention（所有 LM） | `attention.py:222 FlexAttention.Config`、`:126 VarlenAttention.Config` | 直接 override（`kernels/attention.py`，`npu_fusion_attention`）。**M1 就需要**，因为上游删了 eager 的 `sdpa` 路径（`config_utils.py:97`），flex 被 torch 拒绝（TORCH-1），varlen 缺 NPU 内核（NPU-1） |
| RoPE（复数缓存） | `rope.py:179 ComplexRoPE.Config` | 直接 override（`kernels/rope.py`）：torch_npu 不能对复数张量做高级索引（NPU-3），改为实数缓存，数学完全一致 |
| `chunk_kda` | `kda.py:48 KDAKernel.Config` | 直接 override（M4） |
| SituGLU | `moe.py:41 SiTUFeedForward` / `:60 SiTUGroupedExperts` | override 父节点（与上游 `fused_swiglu.py` 同粒度） |
| `causal_conv1d` | 在 `InnerKDA.forward` 内（`kda.py:150`） | override `InnerKDA.Config`；粒度偏粗但自洽 |
| attn_res | `model.py:135 _apply_attention_residual`，自由函数 | **没有节点** → 上游 ask（抽成 `Module` 也顺带解决上游自己的 `TODO: Add TP Support`） |

### 3.3 shim 治理（ADR-002）
`@shim(target, reason, upstream, kind, why_not_wrap)`；注册表在 import 时拒绝缺 `upstream` 或 replace-without-reason。`kind="polyfill"` 只在属性缺失时添加，torch 一旦自带就自动 no-op。幂等应用，每条一行日志。`doctor` 列出已注册 shim。包装型 shim 自动继承上游变更；只有替换型 shim 才值得考虑源码指纹（推迟）。

### 3.4 降级与 provenance（ADR-004，P7）
算子依赖缺失 ⇒ L1 模块打 WARNING 且不注册 override ⇒ 上游 eager 运行。provenance（M3）按节点记录实际生效的后端；benchmark 必须附带。

## 4. 依赖与版本管理（F1–F3）—— 2026-08-29 由 M0 在 NPU 上验证（见 docs/baseline.md）

实测事实：
- torchtitan main 面向 PyTorch **nightly**；连 release 也 pin nightly 日期（`docs/release.md:10-12`）。torch_npu 面向 torch **正式版**。
- 最新 release v0.2.2（2026-02-20）没有 override 机制（2026-06-10）也没有 kimi_k3（2026-08-24）。⇒ 按 SHA 固定（ADR-003）。
- `attn-gym[linear]==0.0.5` 被硬 pin；extra 会拉 `nvidia-cutlass-dsl[cu13]`。⇒ `scripts/install.sh` 用 `--no-deps` + `constraints/titan-deps.txt`。attn_gym 的 KDA 有 `naive` 回退，没有 extra 时 kimi_k3 仍可运行。
- 使用中的 nightly-only API：`torch.distributed.set_timeout`、`PipelineSchedule.step(arg_mbs=...)`、`torch._C.Tag.inplace`（fused_mla）、`create_block_mask(separate_full_blocks=)`（2.13 已有）。

机制：
- `constraints/npu.txt` + `constraints/npu-stable.txt` + `constraints/torchtitan.sha` = 四方版本元组（交付物）。
- `scripts/probe_compat.sh` = 最远可 import 的 SHA / 第一个破坏性 commit。
- CI 两条腿：**pinned**（门禁）与 **main / 最远 SHA**（漂移探针，允许红）。判读：🟢/🟢 正常 · 🟢/🔴 上游漂移（下次升级前修）· 🔴/🔴 我们的 bug · 🔴/🟢 上游已修。

**M0 结论（2026-08-29）：GO。** NPU 上实测：
- torch_npu 由 `import torch` 自动加载。
- torchtitan main（`13da2d77c`）在**正式版** torch 2.12.0 / 2.13.0 + torch_npu 2.12.0 / 2.13.0rc1 上运行，nightly-API 缺口只有两处（`set_timeout`、PP `step(arg_mbs=)`），都以 shim 弥合。其它 nightly-only 用法要么在配置后面（`spmd_types`），要么在 NPU 上本来就够不着（Flex），要么是单独的待办（ChunkedLossWrapper，TT-4）。
- `import torchtitan.trainer` 需要 `triton` wheel（TT-1）；`attn-gym[linear]` 用 `--no-deps` 规避。
- 上游两种注意力后端都跑不了（TORCH-1、NPU-1）→ inner-attention override 从 M3 提前到 M1，这也是 M1 能过的原因。
- fake 进程组没有 `npu`（NPU-2）→ `--comm.mode=fake_backend` 是 torch_npu 侧的 🔴，不是我们的。

## 5. 测试策略

| 层 | 位置 | 运行环境 | 门禁？ |
|---|---|---|---|
| 单测：import 纯净性、shim 注册表、doctor、recipe 构建、变换、算子 CPU 数值（RoPE 与上游逐位一致） | `tests/unit` | CPU，每个 PR，两条 CI 腿 | 是（pinned 腿） |
| 每条 shim / 每个算子的 CPU 测试 | `tests/unit` | CPU | 是 |
| 上游集成套件（`features`、`models`；57 例）经 `npu_baseline` 变换 | `tools/matrix.py` | NPU nightly，64 卡 / 8 小时预算 | pinned 腿：是 |
| 数值：opcheck + 对上游 eager 的对齐；golden loss（`check_golden.sh`） | `tests/npu` | NPU nightly | 是 |
| 带 provenance 的性能基线 | `benchmarks/`（M4） | NPU nightly | 阈值告警 |
| nightly 红格自动 bisect | 脚本（M4） | NPU，每个红格约 4 卡·小时 | — |

8 小时窗口超时时的优先级：冒烟 → pinned 矩阵 → 最远 SHA 矩阵 → bisect → 性能 → 版本矩阵；砍尾部并在报告里注明。

## 6. 协作模型
一个上游，多个互不依赖的 vendor 包。归属判据：*第二家硬件也需要它吗？* 需要 ⇒ 上游（设备能力查询、device graph 抽象、attention 后端注册表、attn_res 抽 `Module`）。不需要 ⇒ 本仓。上游 PR **只在"不提就得永久背 fork 或 patch"时**才提；平时用漂移腿产生的 bug 报告做贡献——上游零 review 成本。

## 7. 风险

| 风险 | 可能性 | 缓解 |
|---|---|---|
| 没有同时满足 torchtitan@SHA 与 torch_npu 的 torch 版本（F1） | **已关闭**（M0：交集存在，2 条 shim） | 每次 SHA 升级由 nightly 探针腿复查 |
| flex/varlen 都跑不了（上游没有 eager LM 注意力） | **已发生** | inner-attention override 在 M1 交付；CP/LSE 尾部仍开放（OURS-2） |
| kimi_k3 上游剧烈变动（刚落地，仅 eager） | 高 | M4 前不对 kimi_k3 做 override；attn_res 的上游 ask 等它稳定 |
| FSDP2/DTensor 在 HCCL 上大面积红 | **已排除**（M1 FSDP2×2 🟢） | — |
| 响亮但被忽略的降级导致性能静默失真 | 中 | benchmark 强制 provenance；nightly 性能阈值 |
| shim 数量增长 | 中 | P4 import 时强制、doctor 报告、`upstream-sync` skill 第 6 步 |

## 8. 待定事项（等有数据再定）
- 替换型 shim 的源码指纹——只在真出现时做。
- `AscendTrainer` 子类——尚无必要场景。
- attn_gym 的 Triton KDA fwd 能否在 Triton-Ascend 下编译——M3 花 1–2 天探。
- 首次对外发布前决定发布节奏与版本号。
