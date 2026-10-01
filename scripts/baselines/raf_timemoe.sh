#!/usr/bin/env bash
# RAF baseline on Time-MoE (fine-tuned checkpoints from scripts/finetune/timemoe.sh).
#   bash scripts/baselines/raf_timemoe.sh [fixed|various] [DATASET ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

PROTOCOL=${1:-fixed}; shift || true

declare -A CHANNELS=([ETTh1]=7 [ETTh2]=7 [ETTm1]=7 [ETTm2]=7 [exchange_rate]=8 [weather]=21 [illness]=7
                     [us_births]=1 [saugeenday_dataset]=1 [sunspots]=1)
# Time-MoE attends over every time step, so the doubled RAF context needs small batches.
declare -A BATCH_FIXED=([ETTh1]="64 64 32 16" [ETTh2]="64 64 32 16" [ETTm1]="64 64 32 16"
                        [ETTm2]="64 64 32 16" [exchange_rate]="64 64 32 16" [weather]="64 64 32 16"
                        [illness]="1024 1024 1024 1024" [us_births]="128 128 32 32"
                        [saugeenday_dataset]="64 64 32 16" [sunspots]="128 64 16 8")
declare -A BATCH_VARIOUS=([ETTh1]="64 16 8 4" [ETTh2]="64 16 8 4" [ETTm1]="64 16 8 4"
                          [ETTm2]="64 16 8 4" [exchange_rate]="64 16 8 4" [weather]="64 16 8 4"
                          [illness]="1024 16 16 16" [us_births]="128 16 8 4"
                          [saugeenday_dataset]="64 16 8 4" [sunspots]="128 16 8 4")

for data in $(select_datasets "$@"); do
  read -r -a preds <<< "$(horizons "$data")"
  read -r -a seqs <<< "$(lookbacks "$data" "$PROTOCOL")"
  if [[ "$PROTOCOL" == various ]]; then read -r -a bss <<< "${BATCH_VARIOUS[$data]}"; else read -r -a bss <<< "${BATCH_FIXED[$data]}"; fi
  for i in "${!preds[@]}"; do
    $PYTHON -m baselines.raf_timemoe \
      -m "${CKPT_ROOT}/timemoe/${data}" -d "$(data_path "$data")" --data "$data" \
      --num_channels "${CHANNELS[$data]}" -c "${seqs[$i]}" -p "${preds[$i]}" -b "${bss[$i]}" \
      --cache_dir "$CACHE_ROOT" --output_dir "${RESULTS_ROOT}/${PROTOCOL}"
  done
done
