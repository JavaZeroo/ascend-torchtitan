# 性能基线

```bash
python -m ascend_titan.tools.bench --cards 0 --out docs/perf
python -m ascend_titan.tools.bench --cards 0-1 --out docs/perf \
    --recipe ascend_titan.models.qwen3:qwen3_debugmodel_npu_fsdp2:2
```

每次生成 `<日期>_<版本元组>.md`（表格）与同名 `.json`（含每一步的原始数字）。

规矩（P7）：**每行必须带 provenance**——这一跑到底是融合算子生效了，还是悄悄退回上游 eager，
数字长得一模一样。provenance 收集失败的行不给数字，直接标 ⚠️。

跑法固定 `--debug.seed 42 --debug.deterministic`，所以 loss 列同时是一次正确性检查
（应与 `tests/assets/losses/npu/` 的 golden 一致）；吞吐取后半程步的中位数，避开 step 1 的预热，
但共享机器上仍会有几个百分点的抖动，所以 `.json` 保留每步原始值。
