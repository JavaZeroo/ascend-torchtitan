---
name: torch-npu-fix
description: 归因为 NPU（torch_npu / op-plugin）的失败的完整处理流程（P9）：最小复现 → 在 ../ascend-pytorch 或 ../ascend-op-plugin 的 fix/<ID> 分支修复 + UT → 源码构建 → NPU 验证 → patches/ 存 format-patch → 用 gitcode-pr-rfc-pipeline 提 issue + PR → STATUS.md 记 URL。用于任何 traceback 首个非框架帧落在 torch_npu/、op_plugin/ 或 aclnnXxx failed 的情况。绝不绕过。
---
# torch-npu-fix

红线：**torch_npu 的缺陷只能修，不能绕**（P1/P9）。在本仓任何位置（recipe、baseline、shim、换 loss、换算子）绕过都是违规。

## 1. 复现
- 写最小复现到 `tests/repro/<ID>_<slug>.py`（或加进 `probe_npu_gaps.py`），在 NIGHTLY（`/opt/venv-nightly`）上跑出 `[ERR]` 行，记下完整错误串（`docs/issues/torch_npu.md` 新条目：现象 / 检查 / 影响 / 诉求）。
- 判断修复位置：Python 层行为（注册、导入、dispatch、inductor 覆盖）→ `../ascend-pytorch`（torch_npu）；算子内核 / `aclnnXxx` 报错 → `../ascend-op-plugin`（`op_plugin/ops/opapi/*.cpp`），修 C++ 而不是在 Python 层绕（NPU-3 的教训）。

## 2. 修复 + UT（分支 `fix/<ID>`，从 master 拉）
- torch_npu UT 放 `test/npu/`、`test/distributed/`、`test/_inductor/`；op-plugin UT 放 `test/test_base_ops/`。用 `torch_npu.testing.testcase.TestCase`，与 CPU 对齐。
- 提交信息：`fix: <一句话>` + 现象（含错误串）+ 根因 + 方案 + `Test:` 行。**不要写任何 AI 署名**。

## 3. 构建 + 验证
```bash
git -C ../ascend-pytorch format-patch -1 --stdout fix/<ID> > patches/torch_npu/<ID>-<slug>.patch     # 或 ../ascend-op-plugin → patches/op-plugin/
source /usr/local/Ascend/cann-9.1.0/set_env.sh
WITH_PATCHES=1 REQUIRE_PR_LINK=0 VENV=/opt/venv-nightly ./scripts/build_torch_npu.sh   # 只允许一个构建在跑
pip install --no-deps --force-reinstall /opt/wheels/torch_npu-*.whl
python tests/repro/probe_npu_gaps.py            # 目标行变 [OK ]
(cd /opt/build/torch_npu && python test/<...>.py) # 新 UT 通过
ASCEND_TITAN_SKIP_SHIMS=1 ... ./scripts/check_golden.sh ...   # 相关格子翻绿，golden 不变
```

## 4. 提交上游（只允许 gitcode.com/ascend/*，P10）
- 用 skill `gitcode-pr-rfc-pipeline`：先搜候选 issue，再用 `docs/issues/torch_npu.md` 的条目开 issue，push `fix/<ID>` 到 fork（`jimmyisme1/pytorch`；op-plugin 需先 fork），按 `.gitcode/PULL_REQUEST_TEMPLATE.md` 开 PR 并关联 issue。功能验证一栏贴第 3 步的真实输出（不贴本地路径、不贴大段日志）。
- PR URL 写进补丁头部（`build_torch_npu.sh` 默认要求）与 `docs/issues/STATUS.md`。
- 合入后：删补丁、升 `constraints/torch_npu.sha`、重跑 golden。

## 5. 记录
`STATUS.md` 一行（状态 + 位置 + 验证），`capability-matrix.md` 只引用 ID。
