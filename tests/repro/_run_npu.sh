#!/bin/bash
set -e
. /opt/venv-nightly/bin/activate
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /data/ljb/projects/create-ascend-titian/ascend-torchtitan
export ASCEND_RT_VISIBLE_DEVICES=0
python "$1" 2>&1
