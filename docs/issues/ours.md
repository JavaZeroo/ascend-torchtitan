# ascend-torchtitan —— 已知缺口（归属：本仓）

| 编号 | 缺口 | 计划 |
|---|---|---|
| OURS-1 | `AscendFusionAttention` 每步做一次 D2H 同步把 `cu_seq_*` 转成 host 整数（`_host_offsets`）。 | 请上游在 inner attention 声明需要 host offsets 时于 `get_attention_masks` 传 `include_host_offsets=True`；或按 positions 张量缓存。 |
| OURS-2 | `AscendFusionAttention` 拒绝 `out_transform`（LSE 尾部）→ 暂不支持 context parallel / attention sinks（gpt_oss）。 | M4：由 `softmax_max/softmax_sum` 还原 LSE。 |
| OURS-3 | 滑窗（`window_size=(W,0)`）路径使用 `sparse_mode=4`，未测试。 | 启用任何需要它的模型前先加 NPU 测试。 |
| OURS-4 | ~~还没有 provenance 表~~ **已解决（M3）**：`ascend-titan-provenance --module … --config …` 输出每个节点的实际实现。 | 关闭；后续接入 matrix 报告 |
| OURS-5 | NPU golden 已冻结（`tests/assets/losses/npu/`，NEXT/STABLE 逐位一致），但**尚未与同一 recipe 的 GPU 运行对比**；上游的 `qwen3_a10g.txt` 对应不同配置（MoE param-groups，fsdp2+tp2+cp2+ep8）。 | 在 GPU 机器上跑一次 `qwen3_debugmodel_npu` 的增量并记录宽容差对比。 |
| OURS-6 | shim 的 `upstream` 链接在 issue 提交前是 `draft:` 指针。 | 提交 TT-1、TT-2、TT-8、TT-9、NPU-1、NPU-2、NPU-3、TORCH-1；替换指针。 |
| OURS-8 | ~~`torch.compile` 下 dynamo 追踪进 `npu_fusion_attention`……~~ **已解决（M3）**：前向/反向封装为 `custom_op` + `register_fake`，`cu_seq` 张量入参、D2H 在算子内；`fullgraph=True` + `aot_eager` 通过。`*_compile` 用例现在停在 inductor 后端（`DEP-INDUCTOR`：需要 Triton-Ascend/torchair，M5）。 | 关闭 |
| OURS-9 | ~~`npu_baseline` 无条件加 RoPE override，与上游 `fused_mla` 冲突~~ **已解决（M3）**：检测到 `torchtitan.overrides.fused_mla.*` 时跳过 RoPE override。 | 关闭 |
| OURS-7 | 矩阵扫描要求扫描期间没有其它 HCCL 作业占用同一张卡（EI0020 端口冲突记为 `HARNESS`）。 | nightly 用专用卡；或按用例设置 `HCCL_IF_BASE_PORT`。 |
