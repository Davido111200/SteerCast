#!/usr/bin/env bash
# RAFT baseline on Timer-XL (fine-tuned checkpoints from scripts/finetune/timerxl.sh).
#   bash scripts/baselines/raft_timerxl.sh [fixed|various] [DATASET ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

PROTOCOL=${1:-fixed}; shift || true
K=20

declare -A CHANNELS=([ETTh1]=7 [ETTh2]=7 [ETTm1]=7 [ETTm2]=7 [exchange_rate]=8 [weather]=21 [illness]=7
                     [us_births]=1 [saugeenday_dataset]=1 [sunspots]=1)
# Mixing weight of the retrieval forecast.
declare -A GAMMA=([ETTh1]=0.5 [ETTh2]=0.1 [ETTm1]=0.1 [ETTm2]=0.1 [exchange_rate]=0.3 [weather]=0.1 [illness]=0.1
                  [us_births]=0.7 [saugeenday_dataset]=0.1 [sunspots]=0.1)
GAMMA_VARIOUS=0.1  # longer look-backs of the "various" protocol
declare -A BATCH_FIXED=([ETTh1]="1024 32 32 16" [ETTh2]="1024 512 32 16" [ETTm1]="1024 512 32 16"
                        [ETTm2]="1024 512 32 16" [exchange_rate]="1024 512 32 16" [weather]="1024 512 32 16"
                        [illness]="128 128 128 128" [us_births]="1024 512 32 16"
                        [saugeenday_dataset]="1024 512 32 16" [sunspots]="1024 512 32 16")
declare -A BATCH_VARIOUS=([ETTh1]="1024 32 32 16" [ETTh2]="1024 32 32 16" [ETTm1]="1024 32 32 16"
                          [ETTm2]="1024 32 32 16" [exchange_rate]="1024 32 32 16" [weather]="1024 32 32 16"
                          [illness]="128 128 16 16" [us_births]="1024 32 32 16"
                          [saugeenday_dataset]="1024 32 32 16" [sunspots]="1024 32 32 16")

for data in $(select_datasets "$@"); do
  read -r -a preds <<< "$(horizons "$data")"
  read -r -a seqs <<< "$(lookbacks "$data" "$PROTOCOL")"
  if [[ "$PROTOCOL" == various ]]; then read -r -a bss <<< "${BATCH_VARIOUS[$data]}"; else read -r -a bss <<< "${BATCH_FIXED[$data]}"; fi
  for i in "${!preds[@]}"; do
    gamma=${GAMMA[$data]}
    # The first horizon uses the same 512 (illness: 96) look-back in both protocols.
    if [[ "$PROTOCOL" == various && $i -gt 0 ]]; then gamma=$GAMMA_VARIOUS; fi
    $PYTHON -m baselines.raft_timerxl \
      -m "${CKPT_ROOT}/timerxl/${data}/epoch-1" -d "$(data_path "$data")" --data "$data" \
      --num_channels "${CHANNELS[$data]}" -c "${seqs[$i]}" -p "${preds[$i]}" -b "${bss[$i]}" \
      --k "$K" --gamma "$gamma" --cache_dir "$CACHE_ROOT" --output_dir "${RESULTS_ROOT}/${PROTOCOL}"
  done
done
