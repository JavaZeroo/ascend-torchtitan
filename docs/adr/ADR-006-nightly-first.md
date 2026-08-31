# ADR-006：三个 main 对齐 —— 以 torch nightly + torch_npu master 为基线（nightly-first）

## 状态
已采纳（2026-08-30），取代 ADR-003 背景中"torch_npu 追 torch 正式版"的前提与 `docs/baseline.md` 的 NEXT/STABLE 双 track。

## 背景
torchtitan main 面向 torch nightly（README；CI 装 `--pre torch … /whl/nightly/cpu`）。我们此前以 torch 正式版 2.12 / 2.13 为基线，
理由是"torch_npu 只发正式版配套"。2026-08-30 实测推翻了这个前提：torch_npu master 的 `requirements.txt` 钉 `torch==2.14.0.dev20260719+cpu`、
`requirements_2.15.txt` 钉 `torch==2.15.0.dev20260812+cpu`，`version.txt` 列 `2.15.0`，`ci/build.sh --torch=2.15.0` 就是官方构建路径；
在开发容器里对 torch 2.15.0.dev20260812 源码构建 master `15514cc70` 耗时 8 分 28 秒、零报错。
正式版基线的代价：多条只为版本差存在的 shim 与补丁，以及的 1 个、矩阵里 14 个 CP 红格（TT-5）
全部只是 torch 版本差；这些"接口补丁"没有一条是昇腾问题。

## 决定
1. 唯一门禁 track = **NIGHTLY**：torch nightly（日期 = torch_npu master `requirements_<line>.txt` 钉的日期）+ torch_npu master 源码构建
   （`constraints/torch_npu.sha`，`scripts/build_torch_npu.sh`）+ torchtitan main（`constraints/torchtitan.sha`）。三者作为一个元组一起升级，一个 PR 附矩阵结果（P5 扩展）。
2. **RELEASE** track（torch_npu 最新发布版 + 其配套 torch）只做信息性报告，不门禁、不写 shim。STABLE（2.12）删除。
3. 只在正式版 torch 上出现、nightly 上不存在的问题**不是问题**（P8）：不写 shim/补丁、不记 issue。
4. torch_npu 的缺陷按 P9 走"修复 → 构建 → 验证 → gitcode issue + PR"；在途修复以 `patches/torch_npu/*.patch`（必须带 PR 链接）叠加进构建（`WITH_PATCHES=1`），合入即删。

## 备选
- **继续正式版基线 + shim 层**：否决。shim 层维护的是 torch 的历史，不是昇腾的现在；每次 torchtitan 升级都会新增一批。
- **等 torch_npu 发布 nightly wheel**：无此渠道（PyPI 最新 2.13.0rc1）；源码构建 8.5 分钟，可接受，且正是 torch_npu 自己的 CI 路径。
- **固定 torch nightly 日期为最新**：否决。日期跟随 torch_npu master 的 pin，保证 C++ ABI 与 torch_npu 的 CI 一致；需要更新时先升 torch_npu SHA。

## 后果
- 环境搭建多一步构建（`scripts/build_torch_npu.sh`，产物缓存于 `/opt/wheels/`，元数据含源码/op-plugin SHA 与 sha256）。
- `constraints/nightly.txt` 成为默认；`scripts/install.sh` 识别 `.dev` pin 走 nightly index 与本地 torch_npu wheel。
- `npu_baseline` 的版本差增量改为特性探测（`_torch_fsdp_reads_spmd_types`），在 nightly 上自动消失。
- golden 只按 NIGHTLY 元组记录。
