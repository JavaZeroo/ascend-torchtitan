#!/usr/bin/env bash
# 下载 release 级 recipe 需要的真实资产（tokenizer 与数据）到本地盘。
#   ./scripts/fetch_assets.sh tokenizer Qwen/Qwen3-0.6B
#   ./scripts/fetch_assets.sh c4 2                       # 下载 2 个真实 C4 分片（每个约 300MB）
#   ./scripts/fetch_assets.sh all                        # 两个模型的 tokenizer + 2 个分片
#
# 放在 $ASCEND_TITAN_ASSETS（默认 /opt/assets，**本地盘**：HF 的 filelock 在 NFS 上会报
# "No locks available"）。这台机器访问不到 huggingface.co，走 HF_ENDPOINT 镜像。
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${ASCEND_TITAN_ASSETS:-/opt/assets}
TITAN_DIR=${TITAN_DIR:-"$HERE/../torchtitan"}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/opt/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
mkdir -p "$ROOT/hf" "$ROOT/c4" "$HF_HOME"

fetch_tokenizer() {
  local repo=$1 name; name=$(basename "$repo")
  [ -f "$ROOT/hf/$name/tokenizer.json" ] && { echo "have $name"; return; }
  ( cd "$TITAN_DIR" && python scripts/download_hf_assets.py \
      --repo_id "$repo" --local_dir "$ROOT/hf_tmp" --assets tokenizer )
  mv "$ROOT/hf_tmp/$name" "$ROOT/hf/$name"
  echo "fetched $name -> $ROOT/hf/$name"
}

fetch_c4() {
  local n=${1:-2}
  python - "$n" "$ROOT" <<'PY'
import sys, os
from huggingface_hub import hf_hub_download
n, root = int(sys.argv[1]), sys.argv[2]
for i in range(n):
    f = f"en/c4-train.{i:05d}-of-01024.json.gz"
    p = hf_hub_download("allenai/c4", filename=f, repo_type="dataset", local_dir=f"{root}/c4")
    print("have", p, os.path.getsize(p) >> 20, "MB")
PY
}

case "${1:-all}" in
  tokenizer) fetch_tokenizer "${2:?repo id, e.g. Qwen/Qwen3-0.6B}" ;;
  c4)        fetch_c4 "${2:-2}" ;;
  all)       fetch_tokenizer Qwen/Qwen3-0.6B; fetch_tokenizer Qwen/Qwen3.5-0.8B; fetch_c4 2 ;;
  *) echo "usage: $0 {tokenizer <repo>|c4 [n]|all}"; exit 2 ;;
esac
