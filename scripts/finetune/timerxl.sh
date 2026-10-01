#!/usr/bin/env bash
# Fine-tune Timer-XL once per dataset -> ${CKPT_ROOT}/timerxl/<DATASET>/epoch-1.
#   bash scripts/finetune/timerxl.sh [DATASET ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

for data in $(select_datasets "$@"); do
  extra=()
  # illness is too short for the default 2880-step context.
  if [[ "$data" == illness ]]; then extra=(--seq_len 96 --pred_len 24); fi
  $PYTHON -m finetune.timerxl \
    -d "$(data_path "$data")" --data "$data" -m thuml/timer-base-84m \
    --output_dir "${CKPT_ROOT}/timerxl" --pool_number 10000 "${extra[@]}"
done
