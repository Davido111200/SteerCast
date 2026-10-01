#!/usr/bin/env bash
# FT baseline on Time-MoE: the fine-tuned checkpoint without retrieval or steering.
# (For Timer-XL and TimesFM, FT is the SteerCast runner with --lam 0.)
#   bash scripts/baselines/ft_timemoe.sh [fixed|various] [DATASET ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

PROTOCOL=${1:-fixed}; shift || true

for data in $(select_datasets "$@"); do
  read -r -a preds <<< "$(horizons "$data")"
  read -r -a seqs <<< "$(lookbacks "$data" "$PROTOCOL")"
  bss=(512 32 32 16)
  for i in "${!preds[@]}"; do
    $PYTHON -m baselines.ft_timemoe \
      -m "${CKPT_ROOT}/timemoe/${data}" -d "$(data_path "$data")" --data "$data" \
      -c "${seqs[$i]}" -p "${preds[$i]}" -b "${bss[$i]}" --output_dir "${RESULTS_ROOT}/${PROTOCOL}"
  done
done
