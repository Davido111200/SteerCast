#!/usr/bin/env bash
# RAF baseline on Timer-XL (fine-tuned checkpoints from scripts/finetune/timerxl.sh).
#   bash scripts/baselines/raf_timerxl.sh [fixed|various] [DATASET ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

PROTOCOL=${1:-fixed}; shift || true

declare -A CHANNELS=([ETTh1]=7 [ETTh2]=7 [ETTm1]=7 [ETTm2]=7 [exchange_rate]=8 [weather]=21 [illness]=7
                     [us_births]=1 [saugeenday_dataset]=1 [sunspots]=1)
declare -A BATCH_FIXED=([ETTh1]="512 512 512 512" [ETTh2]="512 512 32 16" [ETTm1]="512 512 32 16"
                        [ETTm2]="512 512 32 16" [exchange_rate]="512 512 32 16" [weather]="512 512 32 16"
                        [illness]="1024 1024 1024 1024" [us_births]="128 128 32 32"
                        [saugeenday_dataset]="1024 1024 1024 512" [sunspots]="512 128 16 16")
declare -A BATCH_VARIOUS=([ETTh1]="512 32 32 16" [ETTh2]="512 32 32 16" [ETTm1]="512 32 32 16"
                          [ETTm2]="512 32 32 16" [exchange_rate]="512 32 32 16" [weather]="512 32 32 16"
                          [illness]="1024 16 16 16" [us_births]="128 128 32 32"
                          [saugeenday_dataset]="1024 32 32 16" [sunspots]="512 128 16 16")

for data in $(select_datasets "$@"); do
  read -r -a preds <<< "$(horizons "$data")"
  read -r -a seqs <<< "$(lookbacks "$data" "$PROTOCOL")"
  if [[ "$PROTOCOL" == various ]]; then read -r -a bss <<< "${BATCH_VARIOUS[$data]}"; else read -r -a bss <<< "${BATCH_FIXED[$data]}"; fi
  for i in "${!preds[@]}"; do
    $PYTHON -m baselines.raf_timerxl \
      -m "${CKPT_ROOT}/timerxl/${data}/epoch-1" -d "$(data_path "$data")" --data "$data" \
      --num_channels "${CHANNELS[$data]}" -c "${seqs[$i]}" -p "${preds[$i]}" -b "${bss[$i]}" \
      --cache_dir "$CACHE_ROOT" --output_dir "${RESULTS_ROOT}/${PROTOCOL}"
  done
done
