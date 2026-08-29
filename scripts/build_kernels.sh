#!/usr/bin/env bash
# 从源码构建昇腾算子库（pip 源上没有 wheel）。在开发容器内、已 source CANN set_env.sh 的 shell 中运行。
#   ./scripts/build_kernels.sh ops-nn        # AscendC：situ_glu / situ_glu_grad（Kimi-K3 SituGLU）
#   ./scripts/build_kernels.sh ops-transformer  # AscendC：block_attn_res_update（Kimi-K3 attn_res）
#   ./scripts/build_kernels.sh fla-npu       # 需要 triton-ascend（见 docs/issues/ours.md OURS-11）
# 环境变量：SOC（默认 ascend910b；A3 用 ascend910_93）、JOBS（默认 nproc）、SRC（默认 ../）
# 注意：ops-nn 的代码生成用 flock 文件锁，在 NFS（local_lock=none）上会轮询超时（"[ES-GEN] Polling timeout"）。
# 源码在 NFS 上时先拷到本地盘：SRC=/opt/build（rsync --exclude build --exclude build_out）。还需要 pip install build ninja。
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=${SRC:-"$HERE/.."}
SOC=${SOC:-ascend910b}
JOBS=${JOBS:-$(nproc)}
: "${ASCEND_HOME_PATH:?source /usr/local/Ascend/cann-<ver>/set_env.sh first}"

case "${1:-}" in
  ops-nn)
    cd "$SRC/ops-nn"
    bash build.sh --pkg --soc="$SOC" --ops=situ_glu,situ_glu_grad --vendor_name=ascend_titan -j"$JOBS"
    ls build_out/*.run
    echo "install the run package, then the python side:"
    echo "  ./build_out/cann-ops-nn-*linux*.run && source \$ASCEND_HOME_PATH/opp/vendors/ascend_titan/bin/set_env.bash"
    echo "  pip install ninja && TORCH_EXTENSION_OPS=situ_glu,situ_glu_grad TORCH_EXTENSION_VENDOR=ascend_titan pip install ./torch_extension"
    ;;
  ops-transformer)
    cd "$SRC/ops-transformer"
    bash build.sh --pkg --soc="$SOC" --ops=block_attn_res_update --vendor_name=ascend_titan -j"$JOBS"
    ls build_out/*.run
    echo "  ./build_out/cann-ops-transformer-*linux*.run --force"
    echo "  TORCH_EXTENSION_OPS=block_attn_res_update TORCH_EXTENSION_VENDOR=ascend_titan pip install ./torch_extension"
    echo "  export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/opp/vendors/ascend_titan_transformer/op_api/lib:\$LD_LIBRARY_PATH"
    ;;
  fla-npu)
    cd "$SRC/flash-linear-attention-npu"
    python scripts/check_npu_env.py   # 需要 triton-ascend；缺失时在此停下
    FLA_NPU_SOC="$SOC" python -m pip wheel . -w dist --no-deps
    ls dist/*.whl
    ;;
  *) echo "usage: $0 {ops-nn|ops-transformer|fla-npu}"; exit 2 ;;
esac
