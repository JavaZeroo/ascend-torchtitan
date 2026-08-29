#!/usr/bin/env bash
# 从源码构建 torch_npu（gitcode.com/Ascend/pytorch）对当前 venv 里的 torch nightly —— NIGHTLY track 的 torch_npu 来源。
#   source /usr/local/Ascend/cann-9.1.0/set_env.sh
#   VENV=/opt/venv-nightly ./scripts/build_torch_npu.sh              # 固定 SHA（constraints/torch_npu.sha）
#   WITH_PATCHES=1 ./scripts/build_torch_npu.sh                        # 叠加 patches/torch_npu/*.patch（在途修复，P9）
#   REF=fix/npu-2 SRC=../ascend-pytorch ./scripts/build_torch_npu.sh   # 构建本地分支（开发中的修复）
# 环境变量：BUILD（默认 /opt/build/torch_npu，必须是本地盘：NFS 上 flock/构建都不可靠）、WHEELS（默认 /opt/wheels）、
#           PY（默认 3.12）、MAX_JOBS（默认 nproc）、TORCHAIR=1 保留 torchair（默认 --disable_torchair，eager 用不到）。
# 产物：$WHEELS/torch_npu-<line>+git<sha>-*.whl 与同名 .json（源码 SHA、op-plugin SHA、torch 版本、补丁列表、sha256）。
# 实测：master 15514cc70 对 torch 2.15.0.dev20260812+cpu，256 核 gcc 11.4，8 分 28 秒。
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-/opt/build/torch_npu}
VENV=${VENV:-/opt/venv-nightly}
PY=${PY:-3.12}
WHEELS=${WHEELS:-/opt/wheels}
WITH_PATCHES=${WITH_PATCHES:-0}
SRC=${SRC:-https://gitcode.com/Ascend/pytorch.git}
REF=${REF:-$(tr -d '[:space:]' < "$HERE/constraints/torch_npu.sha")}
: "${ASCEND_HOME_PATH:?source /usr/local/Ascend/cann-<ver>/set_env.sh first}"
export PATH="$VENV/bin:$PATH"
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
LINE=$(python -c "import torch; print(torch.__version__.split('+')[0].split('.dev')[0])")   # 2.15.0
echo "torch ${TORCH_FULL} -> building torch_npu line ${LINE}"

# 1. 源码：本地盘克隆，检出 REF，子模块浅克隆（op-plugin / torchair / acl 头文件来源）
if [ ! -d "$BUILD/.git" ]; then
  git clone https://gitcode.com/Ascend/pytorch.git "$BUILD"
fi
if [ -d "$SRC" ]; then   # 本地仓库（如 ../ascend-pytorch）：从它取分支
  git -C "$BUILD" fetch -q "$SRC" "$REF"
  git -C "$BUILD" checkout -q FETCH_HEAD
else
  git -C "$BUILD" fetch -q origin
  git -C "$BUILD" checkout -q "$REF"
fi
git -C "$BUILD" reset -q --hard && git -C "$BUILD" clean -qfd -e build -e dist -e '*.egg-info'
git -C "$BUILD" submodule sync -q && git -C "$BUILD" submodule update --init --recursive --depth 1
SHA=$(git -C "$BUILD" rev-parse HEAD)
OPP=$(git -C "$BUILD/third_party/op-plugin" rev-parse HEAD)

# 2. 在途补丁（torch_npu → patches/torch_npu，op-plugin → patches/op-plugin）。默认要求每个补丁头部带
#    gitcode PR 链接（P9：没有 PR 的补丁不算修复）；提 PR 前的本地验证用 REQUIRE_PR_LINK=0。
APPLIED=()
# 子模块工作树也要干净：上一次叠加的补丁（含其新增的 UT 文件）会让 git apply 拒绝
git -C "$BUILD/third_party/op-plugin" checkout -q -- . && git -C "$BUILD/third_party/op-plugin" clean -qfd
if [ "$WITH_PATCHES" = 1 ]; then
  for pair in "torch_npu:$BUILD:Ascend/pytorch" "op-plugin:$BUILD/third_party/op-plugin:Ascend/op-plugin"; do
    IFS=: read -r sub dir repo <<< "$pair"
    for p in "$HERE"/patches/$sub/*.patch; do
      [ -e "$p" ] || continue
      if [ "${REQUIRE_PR_LINK:-1}" = 1 ]; then
        grep -qE "gitcode\.com/$repo/(pull|merge_requests)/[0-9]+" "$p" \
          || { echo "refusing $(basename "$p"): no gitcode PR link in the patch header (P9; REQUIRE_PR_LINK=0 for pre-PR verification)"; exit 1; }
      fi
      git -C "$dir" apply "$p" || { echo "patch failed: $p"; exit 1; }
      APPLIED+=("$sub/$(basename "$p")")
    done
  done
fi

# 3. 构建
pip install -q pyyaml setuptools auditwheel wheel packaging
cd "$BUILD"
ARGS=(--python="$PY" --torch="$LINE")
[ "${TORCHAIR:-0}" = 1 ] || ARGS+=(--disable_torchair)
MAX_JOBS=${MAX_JOBS:-$(nproc)} bash ci/build.sh "${ARGS[@]}"
WHL=$(ls -t dist/torch_npu-*.whl | head -1)

# 4. 产物与元数据
mkdir -p "$WHEELS"
cp "$WHL" "$WHEELS/"
python - "$WHEELS/$(basename "$WHL")" "$SHA" "$OPP" "$TORCH_FULL" "${APPLIED[*]:-}" <<'PYEOF'
import hashlib, json, sys
whl, sha, opp, torch_full, patches = sys.argv[1:6]
meta = {
    "wheel": whl.split("/")[-1],
    "sha256": hashlib.sha256(open(whl, "rb").read()).hexdigest(),
    "torch_npu_sha": sha, "op_plugin_sha": opp, "torch": torch_full,
    "patches": patches.split() if patches else [],
}
json.dump(meta, open(whl + ".json", "w"), indent=2)
print(json.dumps(meta, indent=2))
PYEOF
echo "install: pip install --no-deps --force-reinstall $WHEELS/$(basename "$WHL")"
