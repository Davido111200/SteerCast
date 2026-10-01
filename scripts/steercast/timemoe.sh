#!/usr/bin/env bash
# SteerCast on fine-tuned Time-MoE (50M).
#
#   bash scripts/steercast/timemoe.sh fixed   [DATASET ...]   # look-back 512, one 512->96 database (Table 2)
#   bash scripts/steercast/timemoe.sh various [DATASET ...]   # look-back 512/1024/2048/3072, one database per horizon (Table 1)
#
# Expects fine-tuned checkpoints in ${CKPT_ROOT}/timemoe/<DATASET> (see scripts/finetune/timemoe.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

PROTOCOL=${1:-fixed}
shift || true

# Per-dataset hyper-parameters: k lambda beta. Set LAMBDA=0 to evaluate the fine-tuned model (FT).
declare -A HP=(
  [ETTh1]="1 0.01 0.0"
  [ETTh2]="1 0.01 0.0"
  [ETTm1]="4 0.01 0.0"
  [ETTm2]="6 0.01 1.0"
  [exchange_rate]="1 0.01 0.8"
  [weather]="1 0.01 0.0"
  [illness]="1 0.01 0.0"
  [us_births]="15 0.01 0.0"
  [saugeenday_dataset]="2 0.01 0.0"
  [sunspots]="5 0.05 0.7"
)
# Evaluation batch size per horizon. Retrieval is done once per mini-batch, so keep these to reproduce the paper.
declare -A BS_FIXED=(
  [ETTh1]="512 512 32 16"
  [ETTh2]="512 512 512 512"
  [ETTm1]="512 512 512 512"
  [ETTm2]="512 512 512 512"
  [exchange_rate]="512 512 512 512"
  [weather]="512 512 512 512"
  [illness]="1024 1024 1024 1024"
  [us_births]="128 128 32 32"
  [saugeenday_dataset]="1024 1024 1024 512"
  [sunspots]="512 128 16 16"
)
declare -A BS_VARIOUS=(
  [ETTh1]="512 512 16 16"
  [ETTh2]="512 512 16 16"
  [ETTm1]="512 128 16 16"
  [ETTm2]="512 128 16 16"
  [exchange_rate]="512 128 8 8"
  [weather]="512 128 16 16"
  [illness]="1024 16 16 16"
  [us_births]="128 128 32 32"
  [saugeenday_dataset]="1024 512 16 16"
  [sunspots]="512 128 16 16"
)

for DATA in $(select_datasets "$@"); do
  read -r K LAM BETA <<< "${HP[$DATA]}"
  LAM=${LAMBDA:-$LAM}
  read -r -a PREDS <<< "$(horizons "$DATA")"
  read -r -a SEQS <<< "$(lookbacks "$DATA" "$PROTOCOL")"
  if [[ "$PROTOCOL" == various ]]; then read -r -a BSS <<< "${BS_VARIOUS[$DATA]}"; else read -r -a BSS <<< "${BS_FIXED[$DATA]}"; fi

  for i in "${!PREDS[@]}"; do
    # Fixed protocol: every horizon reuses the database built for the shortest horizon.
    DB_PRED=${PREDS[$i]}
    if [[ "$PROTOCOL" == fixed ]]; then DB_PRED=${PREDS[0]}; fi
    $PYTHON run_steercast_timemoe.py \
      -m "${CKPT_ROOT}/timemoe/${DATA}" -d "$(data_path "$DATA")" --data "$DATA" \
      -c "${SEQS[$i]}" -p "${PREDS[$i]}" -b "${BSS[$i]}" \
      --k "$K" --lam "$LAM" --beta "$BETA" --db_pred_len "$DB_PRED" \
      --cache_dir "$CACHE_ROOT" --output_dir "$RESULTS_ROOT/$PROTOCOL"
  done
done
