#!/usr/bin/env bash
# Fine-tune TimesFM 2.5 once per (dataset, look-back, horizon):
#   fixed   -> ${CKPT_ROOT}/timesfm/<DATASET>_p<H>
#   various -> ${CKPT_ROOT}/timesfm/<DATASET>_c<L>_p<H>
#   bash scripts/finetune/timesfm.sh [fixed|various] [DATASET ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

PROTOCOL=${1:-fixed}; shift || true
TIMESFM_MODEL=${TIMESFM_MODEL:-google/timesfm-2.5-200m-pytorch}
EPOCHS=1
LR=1e-5
WEIGHT_DECAY=0.01

# Stride between training windows.
declare -A STRIDE=([ETTh1]=8 [ETTh2]=8 [ETTm1]=32 [ETTm2]=32 [exchange_rate]=8 [weather]=32 [illness]=1
                   [us_births]=1 [saugeenday_dataset]=1 [sunspots]=1)

for data in $(select_datasets "$@"); do
  read -r -a preds <<< "$(horizons "$data")"
  read -r -a seqs <<< "$(lookbacks "$data" "$PROTOCOL")"
  if [[ "$data" == illness ]]; then bss=(32 32 16 16); else bss=(16 8 4 2); fi
  for i in "${!preds[@]}"; do
    if [[ "$PROTOCOL" == various ]]; then
      out="${CKPT_ROOT}/timesfm/${data}_c${seqs[$i]}_p${preds[$i]}"
    else
      out="${CKPT_ROOT}/timesfm/${data}_p${preds[$i]}"
    fi
    $PYTHON -m finetune.timesfm \
      -m "$TIMESFM_MODEL" -d "$(data_path "$data")" --data "$data" --out "$out" \
      -c "${seqs[$i]}" -p "${preds[$i]}" -b "${bss[$i]}" \
      --epochs "$EPOCHS" --lr "$LR" --weight_decay "$WEIGHT_DECAY" --stride "${STRIDE[$data]}"
  done
done
