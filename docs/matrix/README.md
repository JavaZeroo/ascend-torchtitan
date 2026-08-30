# 矩阵扫描

`python -m ascend_titan.tools.matrix` 把**上游**的集成测试配置（固定 SHA 的 torchtitan 中的
`tests/integration_tests/{features,models}.py`）搬到 NPU 上运行，对每个配置施加 `npu_minimal`
（`ascend_titan/recipes/transforms.py`），把每个失败自动归因到一个代码，并输出 `report.md` + `results.json`。

```bash
TITAN_DIR=../torchtitan python -m ascend_titan.tools.matrix --list            # 会跑哪些
python -m ascend_titan.tools.matrix --cards 0-7 --jobs 4 --out outputs/matrix/$(date +%F)
python -m ascend_titan.tools.matrix --filter 'pp_|cp' --cards 0-3            # 子集
python -m ascend_titan.tools.matrix --stock --filter default                  # 上游配置原样运行
python -m ascend_titan.tools.matrix --retriage outputs/matrix/<dir>           # 用新规则离线重归因
```

报告以 `<日期>_<track>.md` 提交到这里（`report.md` 的副本）。`docs/capability-matrix.md` 里的汇总表由人工从报告整理：扫描告诉你*哪个用例*以*哪个代码*失败；矩阵说明它对每个轴意味着什么。

归因代码见 `CLAUDE.md`；`ascend_titan/tools/matrix.py` 里的 `TRIAGE` 是正则表——出现新的失败特征时，在那里加一行**并**在 `docs/issues/` 里加条目。

已知的环境类归因：`HARNESS`（HCCL 端口冲突 EI0020：同一张卡上有另一个作业，重跑即可）、`HANG`（超时）、`CLI`（tyro 参数解析错误——子命令如 `activation-checkpoint:none` 必须放在所有 `--flag` 之后）。
