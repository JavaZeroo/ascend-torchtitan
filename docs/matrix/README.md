# 矩阵扫描

`python -m ascend_titan.tools.matrix` 把**上游**的集成测试配置（固定 SHA 的 torchtitan 中的
`tests/integration_tests/{features,models}.py`）搬到 NPU 上运行，把每个失败自动归因到一个代码，
输出 `report.md` + `results.json`。

```bash
TITAN_DIR=../torchtitan python -m ascend_titan.tools.matrix --list            # 会跑哪些
python -m ascend_titan.tools.matrix --cards 0-7 --jobs 2 --provenance \
    --out outputs/matrix/$(date +%F)
python -m ascend_titan.tools.matrix --filter 'pp_|cp' --cards 0-3            # 子集
python -m ascend_titan.tools.matrix --mode stock --filter default            # 上游配置原样运行
python -m ascend_titan.tools.matrix --retriage outputs/matrix/<dir>          # 用新规则离线重归因
```

## `--mode`：对每个上游配置施加什么

| mode | 施加 | 用途 |
|---|---|---|
| `minimal`（默认） | `npu_minimal` —— 只含"不加就跑不起来"的增量 | 门禁口径：红格意味着**上游这个特性在昇腾上不行**，而不是我们的内核把它弄坏了（P12） |
| `stock` | 什么都不加 | 测量原样的上游 |
| `fused` | `npu_minimal` + `npu_fused`（drop-in 融合算子） | 测量融合算子的收益 |

## `--provenance`

在报告末尾附 P7 的审计表：每个可 override 的节点实际由谁支撑（`ascend` = 我们的 override 生效了）。
每个配置在隔离的 override 注册表里构建，构建不起来的配置跳过（它们本来就是红格）。

## 几个会咬人的地方

- **`--jobs` 不要开太大**：8 卡用例会串行等卡，`--jobs 2` 已经够。同一张卡上不能有两个 HCCL 作业（EI0020 → `HARNESS`）。
- **卡表必须升序**：`ASCEND_RT_VISIBLE_DEVICES=4,5,0,1` 会让 torch_npu 报告 0 个设备（NPU-10）。
  `CardPool` 已经强制 `sorted()`，但手动跑的时候要自己注意。
- **别在扫描进行时用同一批卡跑别的东西**：会把结果污染成 `HARNESS`。

## 归因

规则表在 `ascend_titan/tools/matrix/triage.toml`（数据，不是代码；第一条匹配的胜出）。
出现 `UNKNOWN` 时：读日志 → 定代码 → 在 `triage.toml` 加一条 `[[rule]]` → 在 `docs/issues/` 加条目
→ `--retriage <目录>` 重跑归因（不必重跑用例）。

已知的环境类归因：`HARNESS`（HCCL 端口冲突 EI0020，或卡表非升序导致 torch 看不到 NPU；重跑即可）、
`HANG`（超时）、`CLI`（tyro 参数解析——子命令如 `activation-checkpoint:none` 必须放在所有 `--flag` 之后）。

## 报告

报告以 `<日期>_<track>.md` 提交到这里（`report.md` 的副本）。`docs/capability-matrix.md` 的汇总表
由人工从报告整理：扫描告诉你*哪个用例*以*哪个代码*失败；矩阵说明它对每个轴意味着什么。
