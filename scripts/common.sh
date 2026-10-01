# Shared settings for the reproduction scripts (sourced, not executed).
# Every variable can be overridden from the environment, e.g.
#   DATA_ROOT=/data/ts CKPT_ROOT=/ckpt bash scripts/steercast/timemoe.sh fixed ETTh1

DATA_ROOT=${DATA_ROOT:-./dataset}
CKPT_ROOT=${CKPT_ROOT:-./checkpoints}
CACHE_ROOT=${CACHE_ROOT:-./cache}
RESULTS_ROOT=${RESULTS_ROOT:-./results}
PYTHON=${PYTHON:-python}

# The ten benchmarks of the paper, by their --data name.
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather illness us_births saugeenday_dataset sunspots)

# CSV file of a dataset under DATA_ROOT.
data_path() {
  case "$1" in
    ETTh1|ETTh2|ETTm1|ETTm2) echo "${DATA_ROOT}/ETT-small/$1.csv" ;;
    exchange_rate)           echo "${DATA_ROOT}/exchange_rate/exchange_rate.csv" ;;
    weather)                 echo "${DATA_ROOT}/weather/weather.csv" ;;
    illness)                 echo "${DATA_ROOT}/illness/national_illness.csv" ;;
    us_births)               echo "${DATA_ROOT}/us_births_dataset.csv" ;;
    saugeenday_dataset)      echo "${DATA_ROOT}/saugeenday_dataset.csv" ;;
    sunspots)                echo "${DATA_ROOT}/sunspot_dataset_without_missing_values.csv" ;;
    *) echo "Unknown dataset: $1" >&2; return 1 ;;
  esac
}

# Forecast horizons of a dataset.
horizons() {
  if [[ "$1" == illness ]]; then echo "24 36 48 60"; else echo "96 192 336 720"; fi
}

# Look-back lengths matching horizons(): "fixed" (Table 2) or "various" (Table 1).
lookbacks() {
  local dataset=$1 protocol=$2
  if [[ "$dataset" == illness ]]; then
    if [[ "$protocol" == various ]]; then echo "96 192 256 336"; else echo "96 96 96 96"; fi
  else
    if [[ "$protocol" == various ]]; then echo "512 1024 2048 3072"; else echo "512 512 512 512"; fi
  fi
}

# Datasets given on the command line, or all of them.
select_datasets() {
  if [[ $# -gt 0 ]]; then echo "$@"; else echo "${DATASETS[@]}"; fi
}
