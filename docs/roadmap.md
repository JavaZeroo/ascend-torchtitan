# 路线图

每个里程碑只引入它需要的机制。目录从第一天就建好（带 README），后续工作有处可放；代码不预建。

| M | 目标 | 新引入的机制 | 验收 |
|---|---|---|---|
| **M0** ✅ 2026-08-29 | 兼容性探测 | `tools/doctor.py`、`scripts/probe_compat.sh`、`constraints/` | 四方版本元组写入 `constraints/`；最远可 import 的 torchtitan SHA 已知；torch_npu 自动加载有结论；缺失 API 提 issue。**go/no-go 在此决定。** |
| **M1** ✅ 2026-08-29 | Qwen3 跑通 | `setup()`、shim 注册表、`models/qwen3/recipes.py`、`kernels/attention.py` | `qwen3_debugmodel_npu` 10 步：① 单卡 eager ② `--comm.mode=fake_backend` ③ FSDP2×2。NPU golden 冻结。**如预测发生：** flex 与 varlen 在 NPU 上都失败，inner-attention override 因此提前到 M1；fake_backend 为 🔴 NPU-2，等 torch_npu 注册 fake 后端。 |
| **M2** ✅ 2026-08-29 | 能力矩阵 | 上游 `tests/integration_tests` 用例在 NPU 上扫描（`tools/matrix.py`）、矩阵三态记录、nightly CI 两条腿（pinned + 最远 SHA）、归因自动初判 | M2 各轴（并行 × 注意力 × AC × compile）无 ⚪；每个 🔴 有归因。**结果：** NEXT 24 🟢 / STABLE 18 🟢，红格归入 6 个根因（TT-5 spmd_types 需 nightly FSDP2、OURS-8 compile、DEP、TT-KERNEL/TT-CUDA、OURS-2、OURS-9）；新增 RoPE override 与 PP shim；矩阵扫描工具进入 nightly。 |
| **M3** ✅ 2026-08-30 | override 机制加固 | provenance 表（`ascend-titan-provenance`，并接入矩阵报告 `--provenance`）、注意力 custom_op + `register_fake`（compile 可追踪）、`opcheck`（autograd 检查在 NPU 上 NYI，TORCH-7）、第二个 override RMSNorm、注意力的 LSE 尾部（CP / attention sinks）、SituGLU（ops-nn） | 每次运行日志带 provenance；两个 override 带对齐测试。**结果：** 均已落地；`npu_baseline` 拆成 `npu_minimal` + `npu_fused`（P12）。 |
| **M4** ✅ 2026-08-30 | Kimi-K3 + 融合算子 | `kernels/kda.py`（KDA + causal conv1d override）、`models/kimi_k3/`、`tools/bench.py`（性能基线附 provenance）、`tools/bisect.py`（上游回归自动二分） | kimi_k3 recipe 带 KDA / conv1d / SituGLU，性能基线附 provenance。**结果：** `kimi_k3_debugmodel_npu` 单卡 10 步 loss 4.10312（多模态 + KDA + MoE，视觉塔保留 flex、LM 走昇腾融合 varlen）。TT-11 证伪：`cutlass` 就是 `nvidia-cutlass-dsl`，有 aarch64 wheel。attn_res 的昇腾算子缺反向，只能推理用。性能极低（tps 47），等 Triton-Ascend 与 KDA 融合算子。 |
| **M5** ✅ 2026-08-30（FP8 override 除外，见右） | 图模式 / 低精度 / 多模态 | `graph/`（torchair）、多模态轴、低精度轴 | 矩阵轴扩展；不新增机制类型。**结果：** 三条轴都进了能力矩阵并有实测：① 图模式 —— torchair `components=["loss"]` 🟢（qwen3 10 步 loss 5.11634），`["model"]` 🔴 OURS-13（自定义算子无 GE converter）；② 多模态 —— kimi_k3 debugmodel 🟢（视觉塔 + KDA + MoE）；③ 低精度 —— float8 张量分配 🟢（NPU-6 扩展），但转换 `aclnnInplaceCopy 561103`、`_scaled_mm` 明确要求 Ascend950，**910B2 的 CANN 对 float8 只支持按字节存储**。**因此 post-converter 树上的 FP8 override 没有写**：这台硬件上无从验证，按 P13 与"不加投机性代码"，等 Ascend950 / A3 再做。 |

有意推迟、等有数据再定的决策：替换型 shim 的源码指纹（只在真出现替换型 shim 时做）、`AscendTrainer` 子类（尚无必要场景）、vendor 无关中间层（不做）。

## 2026-08-30：M3 / M4 收尾，M5 进行中

已完成
- provenance 接入矩阵报告（`--provenance`）与性能基线（`tools/bench.py`）。
- KDA / causal_conv1d override（`kernels/kda.py`）——不需要 fla-npu 也能跑：走 attn_gym 自己的 reference 实现。
- ops-nn 的 python 扩展已装进 NIGHTLY venv（构建器硬编码 `-std=c++17`，torch 2.15 要 C++20，已本地修复，见下）。
- 图模式（torchair）与多模态轴。
- 上游回归自动二分：`python -m ascend_titan.tools.bisect --config <cfg> --good <sha> --bad origin/main`（在 scratch clone 上做，绝不动 `../torchtitan`）。

未完成 / 阻塞
- **OURS-10（gpt_oss + TP）二分**：8 张卡里 0–5 全程被其它作业占着（AICore ~100%），2 卡复现只得到 HCCL EI0006（HARNESS），等卡空再做。
- **CP**：上游要求 CP 必须配 FlexAttention（`Context Parallel is not supported with ScaledDotProductAttention or VarlenAttention`），而模型级 flex 需要 inductor，所以 CP 完全卡在 Triton-Ascend 上——`parallel/` 里加机制解决不了，保持空目录。
- **Triton-Ascend 装到 NIGHTLY**：`/opt/venv_ta` 里的 `triton_ascend 3.2.2` 绑 triton 3.5.0 + torch 2.13，而 nightly 是 triton 3.7.1；上游只有 `release/3.5.x`/`3.6.x` 分支，要从源码构建（含 LLVM），本轮没做。它同时挡着 inductor、CP、模型级 flex、fla-npu。
- **OURS-13**：自定义算子没有 GE converter，整模型进不了 torchair 图。
- **ops-nn `-std=c++17`**：远端是 `gitcode.com/cann/ops-nn`，不在授权范围（P10 只允许 `gitcode.com/ascend/*`），因此只本地修复 + 存 `patches/ops-nn/`，不提 issue。
- **attn_res 融合算子**：ops-transformer 的 `block_attn_res_update` 只有前向，训练接不进来。

## 2026-08-30：基线切换（ADR-006）
- 基线改为 NIGHTLY（torch nightly + torch_npu master 源码构建 + torchtitan main）；两条 shim、三个 torchtitan 补丁、14 个 CP 红格被证实为版本差。
- 六个昇腾侧问题在 torch_npu / op-plugin 修复并带 UT（`patches/`），按 P9 提 gitcode PR；M3 的后续项（Triton-Ascend、kimi_k3）不变。
- 行动清单见 `docs/design/2026-08-30-architecture-review.md` §8。
