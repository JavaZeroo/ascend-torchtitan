# 路线图

每个里程碑只引入它需要的机制。目录从第一天就建好（带 README），后续工作有处可放；代码不预建。

| M | 目标 | 新引入的机制 | 验收 |
|---|---|---|---|
| **M0** ✅ 2026-08-29 | 兼容性探测 | `tools/doctor.py`、`scripts/probe_compat.sh`、`constraints/` | 四方版本元组写入 `constraints/`；最远可 import 的 torchtitan SHA 已知；torch_npu 自动加载有结论；缺失 API 提 issue。**go/no-go 在此决定。** |
| **M1** ✅ 2026-08-29 | Qwen3 跑通 | `setup()`、shim 注册表、`recipes/qwen3.py`、`kernels/attention.py` | `qwen3_debugmodel_npu` 10 步：① 单卡 eager ② `--comm.mode=fake_backend` ③ FSDP2×2。NPU golden 冻结。**如预测发生：** flex 与 varlen 在 NPU 上都失败，inner-attention override 因此提前到 M1；fake_backend 为 🔴 NPU-2，等 torch_npu 注册 fake 后端。 |
| **M2** ✅ 2026-08-29 | 能力矩阵 | 上游 `tests/integration_tests` 用例在 NPU 上扫描（`tools/matrix.py`）、矩阵三态记录、nightly CI 两条腿（pinned + 最远 SHA）、归因自动初判 | M2 各轴（并行 × 注意力 × AC × compile）无 ⚪；每个 🔴 有归因。**结果：** NEXT 24 🟢 / STABLE 18 🟢，红格归入 6 个根因（TT-5 spmd_types 需 nightly FSDP2、OURS-8 compile、DEP、TT-KERNEL/TT-CUDA、OURS-2、OURS-9）；新增 RoPE override 与 PP shim；矩阵扫描工具进入 nightly。 |
| **M3** 🔄 | override 机制加固 | provenance 表 ✅（`ascend-titan-provenance`）、注意力 custom_op + `register_fake`（compile 可追踪）✅、`opcheck` ✅（autograd 检查在 NPU 上 NYI，TORCH-7）、第二个 override RMSNorm ✅（+30% tps）、注意力的 LSE 尾部（CP）⏳、SituGLU ⏳ | 每次运行日志带 provenance；两个 override 带对齐测试。 |
| **M4** | Kimi-K3 + 融合算子 | `parallel/`（moonep、CP）、nightly 红格自动 bisect、性能基线 | kimi_k3 recipe 带 KDA / conv1d / SituGLU / attn_res，性能基线附 provenance。 |
| **M5** | 图模式 / 低精度 / 多模态 | `graph/`（torchair）、post-converter 树上的 FP8 override、多模态轴 | 矩阵轴扩展；不新增机制类型。 |

有意推迟、等有数据再定的决策：替换型 shim 的源码指纹（只在真出现替换型 shim 时做）、`AscendTrainer` 子类（尚无必要场景）、vendor 无关中间层（不做）。
