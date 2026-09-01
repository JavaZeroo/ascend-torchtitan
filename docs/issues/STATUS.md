# 问题处理状态（逐条，唯一事实来源 —— P11）

基线：**NIGHTLY**（torch 2.15.0.dev20260812+cpu + torch_npu master `15514cc70` 源码构建 + torchtitan `13da2d77c`），2026-08-30。
规则：torch_npu / op-plugin 的问题 → 修复 + UT → `patches/` → 验证 → gitcode issue + PR（P9）；torchtitan / pytorch 的问题**不提上游**（P10），只在 NIGHTLY 仍存在时记录，修复方案存 `patches/evidence/`；只在正式版 torch 上出现的问题**关闭**（P8）。
状态：`已关闭（版本差）` · `已确认` · `已修复（本地补丁）` · `已修复（本仓）` · `无需处理` · `阻塞` · `已提交 <URL>` · `已合入`

| 编号 | 问题 | 状态 | 方案 / 位置 | NIGHTLY 验证 |
|---|---|---|---|---|
| NPU-1 | `_flash_attention_forward/_backward` 无 NPU 内核 | 已提交：issue [4439](https://gitcode.com/Ascend/pytorch/issues/4439) · PR [!45527](https://gitcode.com/Ascend/pytorch/merge_requests/45527)（CLA ✅，CI 运行中） | `patches/torch_npu/NPU-1-flash-attention-privateuse1.patch`（`torch_npu/utils/patch_flash_attention.py` + UT） | 复现 ✅（`probe_npu_gaps.py`：`Could not run 'aten::_flash_attention_forward'`）；修复验证：见 §NIGHTLY 第二轮 |
| NPU-2 | fake 进程组未注册 `npu` | 已提交：issue [4438](https://gitcode.com/Ascend/pytorch/issues/4438) · PR [!45526](https://gitcode.com/Ascend/pytorch/merge_requests/45526)（CLA ✅，CI 运行中） | `patches/torch_npu/NPU-2-fake-process-group-npu.patch`（7 行 + UT） | 复现 ✅（`--comm.mode=fake_backend` → `No backend type associated with device type npu`）；修复验证 ✅（见下表） |
| NPU-3 | 复数张量高级索引失败（aclnnIndex 161002） | 已提交：issue [466](https://gitcode.com/Ascend/op-plugin/issues/466) · PR [!5800](https://gitcode.com/Ascend/op-plugin/merge_requests/5800)（CLA ✅，CI 运行中） | `patches/op-plugin/NPU-3-index-complex.patch`（`op_api::index` 经实数视图 + UT；取代旧的 torch_npu Python 层 `__getitem__` 绕行） | 复现 ✅；修复验证 ✅（见下表） |
| NPU-6 | uint16/32/64 **与 float8** 的 `zero_` / `zeros` 无内核（aclnnInplaceZero 161002） | 已提交：issue [467](https://gitcode.com/Ascend/op-plugin/issues/467) · PR [!5801](https://gitcode.com/Ascend/op-plugin/merge_requests/5801)（CLA ✅，CI 运行中） | `patches/op-plugin/NPU-6-zero-unsigned.patch`（同宽整数视图 + UT）；取代 torch 侧的 TORCH-8 补丁 | 复现 ✅；修复验证 ✅（见下表）。2026-08-30 扩展到 float8：FP8 张量在 910B2 上同样撞 161002，补丁与 UT 已推到同一 PR 并在 PR 上留言说明 |
| NPU-9 | `NPUCombinedScheduling` 未构造父类子调度器，torch 2.15 上编译 FlexAttention 抛 `'_nv_universal_gemm_scheduling'` AttributeError | 已提交：issue [4447](https://gitcode.com/Ascend/pytorch/issues/4447) · PR [!45534](https://gitcode.com/Ascend/pytorch/merge_requests/45534) | `patches/torch_npu/NPU-9-inductor-combined-scheduling.patch`（`__init__` 先调父类构造，再覆盖 NPU 自己的子调度器）+ UT | 复现 ✅（triton-ascend 环境）；修复验证 ✅（编译版 flex 前反向通过） |
| NPU-10 | `ASCEND_RT_VISIBLE_DEVICES` 非升序时 `device_count()` 静默返回 0，上层退回 cuda 后在很远的地方炸 | 已提交：issue [4448](https://gitcode.com/Ascend/pytorch/issues/4448) · PR [!45536](https://gitcode.com/Ascend/pytorch/merge_requests/45536) | `patches/torch_npu/NPU-10-visible-devices-order.patch`（保持行为，加明确告警）+ UT；本仓的 `CardPool` 同时改为 `sorted()` | 复现 ✅（`1,0` → 0 设备）；修复验证 ✅（告警文案已核对） |
| NPU-7 | torch_npu inductor `make_reduction` 覆盖缺 torch 2.15 的 `strict_reduction` | 已提交：issue [4440](https://gitcode.com/Ascend/pytorch/issues/4440) · PR [!45528](https://gitcode.com/Ascend/pytorch/merge_requests/45528)（CLA ✅，CI 运行中） | `patches/torch_npu/NPU-7-inductor-make-reduction-strict.patch` | 复现 ✅（stock flex：`LoweringException: TypeError: make_reduction() got an unexpected keyword argument 'strict_reduction'`）；修复验证 ✅（见下表） |
| NPU-8 | torch_npu 自动加载经 `torch.distributed._tensor` 拖入 checkpoint/fsdp → `spmd_types` 循环导入 | 已提交：issue [4441](https://gitcode.com/Ascend/pytorch/issues/4441) · PR [!45529](https://gitcode.com/Ascend/pytorch/merge_requests/45529)（CLA ✅，CI 运行中） | `patches/torch_npu/NPU-8-dtensor-public-imports.patch` | 复现 ✅（`python -c "import spmd_types"` → `Failed to load the backend extension: torch_npu`）；修复验证 ✅（见下表） |
| NPU-4 | ArgSort int 回退 AiCpu（性能警告） | 无需处理 | 记录 | — |
| TORCH-1 | FlexAttention 设备白名单 | 已确认（torch 侧仍在）；**torch_npu master 已绕开** | torch_npu master `utils/patch_flexattention.py`；NIGHTLY 上 `flex_attention` eager 前反向 ✅。模型级 stock flex 走 `torch.compile` → inductor：NPU-7 + Triton-Ascend（`CANN/硬件（document mask 间接寻址，Ascend950 才有）`） | eager ✅；compile 见第二轮 |
| TORCH-2 | fake 后端不可扩展 | 无需处理 | torch_npu 侧追加即可（NPU-2） | — |
| TORCH-7 | `opcheck` autograd 检查不支持 privateuse1 | 已确认（NIGHTLY 仍在） | `patches/evidence/pytorch/0001-TORCH-7-*.patch`；本仓 NPU 测试用数值梯度 | — |
| TORCH-8 | `varlen.py` rng_state 占位用 uint64 | 已确认（NIGHTLY 仍在）；**昇腾侧由 NPU-6 解决** | `patches/evidence/pytorch/0002-TORCH-8-*.patch` 仅作证据 | — |
| TT-1 | core 无条件 `import triton` | 已确认（NIGHTLY 仍在） | `constraints/titan-deps.txt` 装纯 Python `triton`；`patches/evidence/torchtitan/0004-TT-1-*.patch` | ✅ import OK |
| TT-3 | `separate_full_blocks` 仅 nightly | 已关闭（版本差） | — | — |
| TT-4 | ChunkedLossWrapper backward "not allocated" | **已关闭（NIGHTLY 不复现）** | `npu_baseline` 不再展开 loss（此前是 P1/P9 违规） | ✅ 单卡 step10 loss 5.10291；FSDP2×2 step10 loss 5.07796（与 CE golden 差 bf16 级） |
| TT-6 | kimi_k3 attn_res 无 Configurable 节点 | 已确认 | 上游 ask（不提），等 kimi_k3 稳定 | — |
| TT-7 | LM 移除 sdpa | 无需处理 | override 机制已覆盖 | ✅ |
| TT-10 | 树内 Triton override / DistMuon 写死 CUDA | 无需处理 | 昇腾替代在 L1 | ✅（swiglu 已替代） |
| TT-11 | kimi_k3 导入需要 `cutlass` | **已关闭（证伪）** | `cutlass` 就是 `nvidia-cutlass-dsl`，有 aarch64 wheel；装上即可 import，会执行 cute 内核的节点由 `kernels/kda.py` override 掉 | ✅ kimi_k3 debugmodel 单卡 10 步 loss 4.10312 |
| OURS-1 | attention host offsets D2H | 已修复（本仓） | custom_op 内部；每步一次 D2H 仍在 | — |
| OURS-2/4/8/9 | LSE / provenance / compile graph break / override 冲突 | 已修复（本仓） | 见 git log | ✅ |
| OURS-3 | 滑窗 `sparse_mode=4` 未测 | 待确认 | 补 NPU 测试 | 待做 |
| OURS-5 | 未与 GPU golden 对比 | 阻塞 | 需 GPU 机器 | — |
| OURS-6 | issue 未提交 | 已关闭 | torch_npu / op-plugin 六项已按 P9 提交（见 NPU-* 行）；torchtitan / pytorch 按 P10 不提 | — |
| OURS-7 | 扫描期间同卡 HCCL 冲突 | 无需处理 | HARNESS | — |
| OURS-10 | gpt_oss × TP | 已确认 | 排查中（NIGHTLY 上复测待做） | 待做 |
| OURS-11 | fla-npu / inductor 需要 Triton-Ascend | **已关闭（2026-09-01）** | **DEP-FLA 证伪**：`fla-core` 有 aarch64 wheel，import 正常；挡住的只是它的 CUDA Triton 内核，gated delta rule 与 causal conv1d 已由 `kernels/gdn.py` 的 override 接管，后者已进一步换成 `kernels/gdn_fla.py` 的 AscendC 融合（R5，NPU 对拍 🟢）。**已关闭（2026-09-01）**：Triton-Ascend 3.2.2 已装进基线（`scripts/install_triton.sh`），inductor 能编前反向；模型级 flex 剩下的是硬件门（document mask 的间接寻址只有 Ascend950） | ✅ `qwen35_debugmodel_npu_text` 10 步 loss 3.54783；`qwen35_0_8b_npu` 真实尺寸可跑 |
| OURS-12 | **`npu_baseline` 违反 P1/P9**：展开 ChunkedLossWrapper（TT-4）、性能 override 混入 baseline | **已关闭（2026-08-30）** | TT-4 展开已删除；`npu_baseline` 拆成 `npu_minimal`（矩阵默认）+ `npu_fused`（opt-in），`npu_rms_norm` 已移出基线 | `tests/unit/test_matrix.py`：minimal 不含 RMSNorm override、fused 含 |
| OURS-13 | 矩阵工具在 `setup()` 之前导入 torchtitan（F4 顺序），暴露了 NPU-8 | 已确认 | NPU-8 修复后可运行；工具本身也应先 `setup()`（待做） | 第二轮 |
| OURS-14 | fla-npu 融合 GDN（R5）模型级已跑通，但 **run-to-run 非确定性**（单卡 0.8B step-1 loss 三次 12.93595 / 12.93635 / 12.93662，纯 torch 稳定 12.93624），无法按 `check_golden.sh` 冻结逐位 golden | 已确认 | AscendC `chunk_*` 内核的归约/原子序不定；对拍改走「纯 torch 参考 ± bf16 容差」断言（`tests/npu/test_kernel_gdn_fla.py`），模型级 golden 状态记为「无逐位 golden，用文档容差」 | 三次 0.8B 训练 + 算子级对拍 4 passed |

## NIGHTLY 修复验证（含六个补丁的 torch_npu 构建，2026-08-30）

| 检查 | 结果 |
|---|---|
| `tests/repro/probe_npu_gaps.py` | NPU-1 / NPU-2（列表）/ NPU-3 / NPU-6 `[OK ]`；`_flash_attention_forward` PrivateUse1 = True；flex eager OK；TT-4 OK |
| stock varlen（qwen3，零 override）10 步 | 🟢 loss 5.10302 / grad_norm 3.3060 |
| stock llama3（`ascend_titan.models.llama3`，零 override：stock ComplexRoPE + ChunkedLoss + spmd_types）10 步 | 🟢 单卡 4.01820 / 1.7382；FSDP2×2 3.97774 / 1.7523 |
| `import torch` 不再拖入 fsdp/checkpoint；`import spmd_types` 先行 | 🟢（NPU-8） |
| 矩阵工具可运行；`pp_1f1b` | 🟢（无 shim） |
| `cp` / `fsdp+cp` / `deepseek_v3_fused_mla_swiglu` | 🔴 CANN/硬件（document mask 间接寻址，Ascend950 才有）（Triton-Ascend） |
| stock flex 模型级 | lowering 通过（NPU-7）→ Triton-Ascend 已装、inductor 能编 → 仍红：document mask 的间接寻址只有 Ascend950 能 lower（硬件门） |
| UT：`test_autoload.py` / op-plugin `test_index_complex.py` / inductor 签名 | 4 OK / 6 OK / 1 passed |
| UT（最终 wheel）：`test_fake_process_group_npu.py` 1 OK / `test_flash_attention_privateuse1.py` 4 OK / `test_autoload.py` 4 OK / op-plugin `test_index_complex.py` 6 OK / `test_zero_unsigned.py` 3 OK / inductor 签名 1 passed | 🟢 |
| `--comm.mode=fake_backend`（单卡模拟 8 卡干跑，NPU-2 第三版：包装 `Backend.register_backend`） | 🟢 `step: 1  loss: 7.66238`，exit 0 |
