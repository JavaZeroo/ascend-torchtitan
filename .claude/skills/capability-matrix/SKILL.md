---
name: capability-matrix
description: 把一次 NPU 运行的结果记入 docs/capability-matrix.md 并给出正确归因（TT / NPU / CANN / DEP / TORCH），或运行整套矩阵扫描。用于任何训练运行、矩阵扫描、CI nightly 之后，以及排查红格时。
---
# capability-matrix

## 跑一次扫描
`python -m ascend_titan.tools.matrix --cards 0-7 --jobs 4 --out outputs/matrix/<日期>`（见 `docs/matrix/README.md`）。
它对每个上游测试配置施加 `npu_baseline`，并用 `TRIAGE` 正则表自动归因。`UNKNOWN` 表示新的失败特征：读日志、定代码、在 `TRIAGE` 加一行正则、在 `docs/issues/<owner>.md` 加条目，然后 `--retriage`（`results.json` 保留日志路径）。扫描期间同一张卡上不能有其它 HCCL 作业（EI0020 → `HARNESS`）。

## 翻转一个格子
1. 确定行（特性/轴）和验证元组（`constraints/`）。
2. 🟢：写日期和命令。🔴：归因必填（CLAUDE.md 的表），附 issue 链接。⚪ → 一旦测过绝不改回 ⚪。
3. 归因决定负责人：
   - **TT / TORCH** → 先确认 NIGHTLY 上存在（P8）；存在则记入 `docs/issues/` + `patches/evidence/`（不提上游，P10），必要时包装型/polyfill shim（skill `shim-authoring`）。
   - **NPU / NPU-OP** → skill `torch-npu-fix`（P9：复现 → 修 torch_npu/op-plugin → 构建 → 验证 → gitcode issue + PR）；**不 workaround**（P1）。L1 override 只能作为性能项，不能作为掩盖 NPU 缺陷的手段（P12）。
   - **CANN** → 记录错误码；不再投入。
   - **DEP** → 记录包名；若是内核，昇腾替代是 M3+ 任务。

## 快速排查红格
```
grep -m1 -nE "torch_npu/|torchtitan/|attn_gym|helion|deep_ep|cutlass|EZ[0-9]{4}|EI[0-9]{4}" <log>
```
第一个命中通常就是归因。多个格子同一失败 ⇒ 同一根因；归因一次，其余引用。

## 批处理，不要流水线
扫描时先全部跑完，再一起归因：红格按根因聚类。
