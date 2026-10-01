#!/usr/bin/env bash
# SteerCast on fine-tuned Timer-XL (84M).
#
#   bash scripts/steercast/timerxl.sh fixed   [DATASET ...]   # look-back 512, one 512->96 database (Table 2)
#   bash scripts/steercast/timerxl.sh various [DATASET ...]   # look-back 512/1024/2048/3072, one database per horizon (Table 1)
#
# Expects fine-tuned checkpoints in ${CKPT_ROOT}/timerxl/<DATASET>/epoch-1 (see scripts/finetune/timerxl.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

PROTOCOL=${1:-fixed}
shift || true

# Per-dataset hyper-parameters: k lambda beta. Set LAMBDA=0 to evaluate the fine-tuned model (FT).
declare -A HP=(
  [ETTh1]="1 0.01 0.0"
  [ETTh2]="16 0.05 0.0"
  [ETTm1]="15 0.01 0.0"
  [ETTm2]="4 0.01 0.0"
  [exchange_rate]="4 0.01 0.0"
  [weather]="4 0.01 0.0"
  [illness]="1 0.01 0.0"
  [us_births]="2 0.1 0.9"
  [saugeenday_dataset]="13 0.01 1.0"
  [sunspots]="10 0.1 0.3"
)

for DATA in $(select_datasets "$@"); do
  read -r K LAM BETA <<< "${HP[$DATA]}"
  LAM=${LAMBDA:-$LAM}
  read -r -a PREDS <<< "$(horizons "$DATA")"
  read -r -a SEQS <<< "$(lookbacks "$DATA" "$PROTOCOL")"
  # Evaluation batch size per horizon (retrieval is done once per mini-batch).
  if [[ "$DATA" == illness ]]; then BSS=(128 128 128 128)
  elif [[ "$PROTOCOL" == various ]]; then BSS=(1024 128 16 16)
  else BSS=(1024 1024 1024 1024); fi

  for i in "${!PREDS[@]}"; do
    # Fixed protocol: every horizon reuses the database built for the shortest horizon.
    DB_PRED=${PREDS[$i]}
    if [[ "$PROTOCOL" == fixed ]]; then DB_PRED=${PREDS[0]}; fi
    $PYTHON run_steercast_timerxl.py \
      -m "${CKPT_ROOT}/timerxl/${DATA}/epoch-1" -d "$(data_path "$DATA")" --data "$DATA" \
      -c "${SEQS[$i]}" -p "${PREDS[$i]}" -b "${BSS[$i]}" \
      --k "$K" --lam "$LAM" --beta "$BETA" --db_pred_len "$DB_PRED" \
      --cache_dir "$CACHE_ROOT" --output_dir "$RESULTS_ROOT/$PROTOCOL"
  done
done
