#!/usr/bin/env bash
# RAFT baseline on TimesFM 2.5 (per-horizon fine-tuned checkpoints from scripts/finetune/timesfm.sh).
#   bash scripts/baselines/raft_timesfm.sh [fixed|various] [DATASET ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

PROTOCOL=${1:-fixed}; shift || true
K=4

declare -A CHANNELS=([ETTh1]=7 [ETTh2]=7 [ETTm1]=7 [ETTm2]=7 [exchange_rate]=8 [weather]=21 [illness]=7
                     [us_births]=1 [saugeenday_dataset]=1 [sunspots]=1)
# Mixing weight of the retrieval forecast.
declare -A GAMMA=([ETTh1]=0.2 [ETTh2]=0.2 [ETTm1]=0.2 [ETTm2]=0.2 [exchange_rate]=0.3 [weather]=0.1 [illness]=0.1
                  [us_births]=0.7 [saugeenday_dataset]=0.1 [sunspots]=0.1)
declare -A BATCH=([ETTh1]="512 512 512 512" [ETTh2]="512 512 32 16" [ETTm1]="512 512 32 16"
                  [ETTm2]="512 512 32 16" [exchange_rate]="512 512 32 16" [weather]="512 512 32 16"
                  [illness]="1024 512 32 16" [us_births]="128 128 32 32"
                  [saugeenday_dataset]="1024 1024 32 16" [sunspots]="512 128 16 16")

# Fine-tuned checkpoint for (dataset, look-back, horizon).
timesfm_ckpt() {
  if [[ "$PROTOCOL" == various ]]; then echo "${CKPT_ROOT}/timesfm/$1_c$2_p$3"; else echo "${CKPT_ROOT}/timesfm/$1_p$3"; fi
}

for data in $(select_datasets "$@"); do
  read -r -a preds <<< "$(horizons "$data")"
  read -r -a seqs <<< "$(lookbacks "$data" "$PROTOCOL")"
  read -r -a bss <<< "${BATCH[$data]}"
  for i in "${!preds[@]}"; do
    $PYTHON -m baselines.raft_timesfm \
      -m "$(timesfm_ckpt "$data" "${seqs[$i]}" "${preds[$i]}")" -d "$(data_path "$data")" --data "$data" \
      --num_channels "${CHANNELS[$data]}" -c "${seqs[$i]}" -p "${preds[$i]}" -b "${bss[$i]}" \
      --k "$K" --gamma "${GAMMA[$data]}" --cache_dir "$CACHE_ROOT" --output_dir "${RESULTS_ROOT}/${PROTOCOL}"
  done
done
