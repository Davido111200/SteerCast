#!/usr/bin/env bash
# SteerCast on fine-tuned TimesFM 2.5 (200M).
#
#   bash scripts/steercast/timesfm.sh fixed   [DATASET ...]   # look-back 512 (Table 2)
#   bash scripts/steercast/timesfm.sh various [DATASET ...]   # look-back 512/1024/2048/3072 (Table 1)
#
# TimesFM is fine-tuned separately for every horizon (and look-back), so every run builds its own
# database with its own checkpoint: ${CKPT_ROOT}/timesfm/<DATASET>_p<H> (fixed) or
# ${CKPT_ROOT}/timesfm/<DATASET>_c<L>_p<H> (various); see scripts/finetune/timesfm.sh.
# Set LAMBDA=0 to evaluate the fine-tuned model without steering (FT).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

PROTOCOL=${1:-fixed}
shift || true
LAM=${LAMBDA:-0.01}

# Number of retrieved neighbours (k) per dataset; tune with --split val for new settings.
declare -A K=(
  [ETTh1]=4 [ETTh2]=4 [ETTm1]=4 [ETTm2]=4 [exchange_rate]=4 [weather]=4
  [illness]=1 [us_births]=4 [saugeenday_dataset]=4 [sunspots]=4
)
# Evaluation batch size per horizon (retrieval is done once per mini-batch).
declare -A BS=(
  [ETTh1]="512 512 512 512"
  [ETTh2]="512 512 32 16"
  [ETTm1]="512 512 32 16"
  [ETTm2]="512 512 32 16"
  [exchange_rate]="512 512 32 16"
  [weather]="512 512 32 16"
  [illness]="1024 1024 1024 1024"
  [us_births]="128 128 32 32"
  [saugeenday_dataset]="1024 1024 1024 512"
  [sunspots]="512 128 16 16"
)

for DATA in $(select_datasets "$@"); do
  read -r -a PREDS <<< "$(horizons "$DATA")"
  read -r -a SEQS <<< "$(lookbacks "$DATA" "$PROTOCOL")"
  read -r -a BSS <<< "${BS[$DATA]}"
  for i in "${!PREDS[@]}"; do
    if [[ "$PROTOCOL" == various ]]; then CKPT="${CKPT_ROOT}/timesfm/${DATA}_c${SEQS[$i]}_p${PREDS[$i]}"
    else CKPT="${CKPT_ROOT}/timesfm/${DATA}_p${PREDS[$i]}"; fi
    $PYTHON run_steercast_timesfm.py \
      -m "$CKPT" -d "$(data_path "$DATA")" --data "$DATA" \
      -c "${SEQS[$i]}" -p "${PREDS[$i]}" -b "${BSS[$i]}" \
      --k "${K[$DATA]}" --lam "$LAM" --beta 0.0 \
      --cache_dir "$CACHE_ROOT" --output_dir "$RESULTS_ROOT/$PROTOCOL"
  done
done
