#!/usr/bin/env bash
# Fine-tune Time-MoE-50M once per dataset -> ${CKPT_ROOT}/timemoe/<DATASET>.
#   bash scripts/finetune/timemoe.sh [DATASET ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

for data in $(select_datasets "$@"); do
  $PYTHON -m finetune.timemoe \
    -d "$(data_path "$data")" -m Maple728/TimeMoE-50M -o "${CKPT_ROOT}/timemoe/${data}"
done
