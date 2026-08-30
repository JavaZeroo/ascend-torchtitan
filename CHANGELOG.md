# 变更日志

所有值得记录的变更都在这里。格式：[Keep a Changelog](https://keepachangelog.com/)；0.1.0 之后遵循 SemVer。

## [Unreleased]
### 变更（2026-08-30 · 硬依赖、模型目录、README）
- **P14 / ADR-007：基础依赖硬导入。** 删掉全部 `try: import torch_npu` + `_AVAILABLE` 降级开关；新增 `kernels/_probe.py`（`require_op()` 缺算子即抛 `MissingNpuOpError`，`optional_module()` 只给真正可选的加速包）。`setup()` 去掉 `require_npu` 参数（本来就不该可选）。`tests/unit/test_kernel_import_safety.py` 从"缺 torch_npu 也能安全 import"翻转为"必须抛错"；CPU 单测改用 `npu_stub` / `no_torch_npu` / `npu_stub_missing_op` fixture 提供依赖。
- **按模型重组 L3**：新增 `ascend_titan/models/<model>/`（`recipes.py` + 可选 `probes.py` + **必需的 `README.md`**）。`recipes/qwen3.py` → `models/qwen3/`（4 个矩阵探针拆到 `probes.py`），`recipes/stock.py` → `models/llama3/`；新增 `models/qwen3_5/`、`models/kimi_k3/`（recipe 就绪，状态 🔴 + 归因）、`models/registry.py`（模型状态表，纯数据）、`models/_template/`（新模型骨架）。`ascend_titan/recipes/` 只保留跨模型机制（`transforms.py`、`matrix.py`）。`tests/unit/test_models_registry.py` 强制"每个模型包有 README 且已登记"。
  - 入口路径变化：`--module ascend_titan.recipes.qwen3` → `--module ascend_titan.models.qwen3`（`--config` 函数名不变，golden 不受影响）。
- 补录 NIGHTLY 的 `qwen3_debugmodel_npu_fused` / `_fused_fsdp2` golden：与 torch 2.13 的对应曲线逐位一致；四条 golden 在 NIGHTLY 上齐备。
- **README 重写**：banner / 架构图 / golden loss 曲线（`docs/assets/*.svg`，曲线由冻结的 golden 数据生成）、徽章、模型支持表、特性支持表、上游修复表（六个 gitcode issue/PR）。

### 变更（2026-08-30，nightly-first，ADR-006）
- **基线改为 NIGHTLY**：torch 2.15.0.dev20260812（torch_npu master 钉的日期）+ torch_npu master 源码构建（`scripts/build_torch_npu.sh`，`constraints/torch_npu.sha`）+ torchtitan `13da2d77c`。`constraints/nightly.txt` 成为默认；`install.sh` 识别 `.dev` pin 走 nightly index 与本地 torch_npu wheel。NIGHTLY 上 `ASCEND_TITAN_SKIP_SHIMS=1` 的 golden（单卡 / FSDP2×2）与 2.13 golden 逐位一致。
- 架构评审 `docs/design/2026-08-30-architecture-review.md`；原则 P8–P13（`docs/PRINCIPLES.md`）；`CLAUDE.md` 重写为 nightly-first 工作流。
- `npu_baseline`：spmd_types 步骤改为特性探测（nightly 保留上游默认）；**删除 ChunkedLossWrapper 展开**（TT-4 在 NIGHTLY 不复现；此前是 P1 违规）。
- `patches/` 重组：`torch_npu/`、`op-plugin/` = 在途修复（须带 PR 链接），`evidence/` = torchtitan / pytorch 只读证据；删除版本差补丁 TT-2 / TT-8 / TT-9。
- torch_npu / op-plugin 修复（含 UT，NIGHTLY 验证；已提交 gitcode：Ascend/pytorch !45526–!45529 / issue #4438–#4441，Ascend/op-plugin !5800–!5801 / issue #466–#467）：NPU-1 `_flash_attention_forward/_backward` PrivateUse1 内核、NPU-2 fake 进程组、NPU-3 复数索引（op-plugin C++）、NPU-6 uint16/32/64 `zero_`（op-plugin C++）、NPU-7 inductor `make_reduction` 签名、NPU-8 DTensor 公开导入（spmd_types 循环导入）。
- `ascend_titan/recipes/stock.py`（零 override 的上游 llama3 配置）、`tests/repro/`（最小复现 / 探测脚本）。
- 文档：`docs/issues/STATUS.md` 成为唯一状态来源；`baseline.md`、`upstream-tracking.md`、`capability-matrix.md`、ADR-003 / 设计文档 §4 更正"torch_npu 面向正式版"的错误前提。

### 新增（M3，进行中）
- `kernels/swiglu.py`：复用上游 `FusedSwiGLU`（融合 w13、TP 交错布局、checkpoint hook），激活换成 `torch_npu.npu_swiglu`；`kernels/rope.py` 新增 `npu_rotary_cossin`（CosSinRoPE 旋转用 `npu_rotary_mul`），ComplexRoPE 在 NPU 上也走该内核。
- `qwen3_debugmodel_npu_fused`：RMSNorm + SwiGLU + rotary 三个零构建融合内核，**tps 55k → 77k（+40%）**，显存 2.38 → 1.89 GiB。
- `scripts/build_kernels.sh`：ops-nn / ops-transformer / fla-npu 源码构建入口（ops-nn 需在本地盘构建，NFS 上 flock 超时）。
- ops-transformer `block_attn_res_update` 已从源码构建并注册（算子级），训练接入推迟到 M4（前向流式算子，无反向；见矩阵）。
- `kernels/situ_glu.py`：Kimi-K3 SiTU-GLU 走 ops-nn AscendC 算子（`aclnnSituGlu` 前向 + `situ_glu_grad` 反向，封装为可微 custom_op）；op 级前向与上游 fp32 参考误差 0。kimi_k3 模型本身在无 `cutlass` 环境下不可导入（TT-11）。
- `kernels/attention.py` 改为 `torch.library.custom_op` 前向/反向 + `register_fake`：`cu_seq` 以张量传入、D2H 在算子内部，`torch.compile(fullgraph=True)` 可追踪（OURS-8 关闭）；数值与 golden 逐位不变。
- `kernels/rms_norm.py`：RMSNorm → `torch_npu.npu_rms_norm`（drop-in，Meta/autograd 齐全）；recipe 变体 `qwen3_debugmodel_npu_fused_norm`；`npu_baseline` 默认启用。
- `ascend-titan-provenance`：按节点列出实际生效的实现（ascend / upstream-override / upstream），P7 要求的审计表。
- `npu_baseline` 在上游 override 已 claim attention 块时跳过 RoPE override（OURS-9）。
- 注意力 LSE 尾部（attention sinks / CP 所需）：从 kernel 的 softmax 统计量按文档还原，NPU 测试对 `logsumexp` 对齐；`gpt_oss_pp+fsdp+ep+sacop` 由 🔴 变 🟢。
- 归因：`DEP-INDUCTOR`（inductor 在 NPU 上需要 Triton-Ascend/torchair）。

### 新增
- 包骨架：无副作用的 import、`setup()` 引导、`python -m ascend_titan.train` 入口。
- shim 注册表，支持 `wrap` / `replace` / `polyfill` 三种类型；P3/P4 在 import 时强制。
- `ascend-titan-doctor` 环境探测。
- L1 override：`kernels/attention.py`（VarlenAttention → `torch_npu.npu_fusion_attention`）、`kernels/rope.py`（ComplexRoPE 改为实数缓存，torch_npu 不能索引复数张量）。
- shim：`torch.distributed.set_timeout` 的 polyfill（torch ≤ 2.13）；PP `step(arg_mbs=...)` 预切分 microbatch 转发。
- Qwen3 recipe（`qwen3_debugmodel_npu`、`_fsdp2`、矩阵变体）；`recipes/transforms.py::npu_baseline` 通用变换；`recipes/matrix.py` 动态 recipe 模块。
- `tools/matrix.py`：把上游集成用例搬到 NPU 上扫描并自动归因（`--retriage` 离线重归因；`HARNESS`/`CLI` 环境类归因）。
- M2 结果：`docs/matrix/2026-08-29_{stable,next}.md`，`docs/capability-matrix.md` 按轴汇总。
- 基线（`docs/baseline.md`）：NEXT（torch 2.13.0 / torch_npu 2.13.0rc1）与 STABLE（2.12.0 / 2.12.0），torchtitan `13da2d77c`；golden loss 曲线与 `scripts/check_golden.sh`。
- 能力矩阵、问题清单（`docs/issues/`）、ADR-001..005、Claude Code skill 与规则。
- 开源工程：Apache-2.0、CONTRIBUTING、行为准则、SECURITY、issue/PR 模板、pre-commit、CPU CI、NPU nightly workflow 骨架。
