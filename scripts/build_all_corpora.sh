#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT=${1:-workspace}
SUITE=${2:-main}
RAW_ROOT=${RAW_ROOT:-${OUT_ROOT}/raw_public}
SEED=${SEED:-1}
CORRUPTION_RATE=${CORRUPTION_RATE:-0.25}
MIXTURE_PROFILE=${MIXTURE_PROFILE:-default}

mkdir -p "${OUT_ROOT}/logs"

echo "[build] OUT_ROOT=${OUT_ROOT} SUITE=${SUITE} RAW_ROOT=${RAW_ROOT} SEED=${SEED}"

run_build () {
  local dataset="$1"
  local name="$2"
  shift 2
  echo "[build] ${dataset} -> ${name}"
  python scripts/build_gdpr_corpus.py \
    --dataset "${dataset}" \
    --name "${name}" \
    --out_root "${OUT_ROOT}" \
    --raw_root "${RAW_ROOT}" \
    --seed "${SEED}" \
    --corruption_rate "${CORRUPTION_RATE}" \
    --mixture_profile "${MIXTURE_PROFILE}" \
    "$@" 2>&1 | tee "${OUT_ROOT}/logs/build_${name}.log"
}

run_build lastfm_asia gdpr_lastfm_asia
run_build yelp gdpr_yelp --no_edges_csv

if [[ "${SUITE}" == "extended" ]]; then
  run_build pubmed gdpr_pubmed
  run_build flickr gdpr_flickr --no_edges_csv
  run_build ogbn-arxiv gdpr_ogbn_arxiv --no_edges_csv
fi

if [[ "${SUITE}" == "large" ]]; then
  run_build pubmed gdpr_pubmed
  run_build flickr gdpr_flickr --no_edges_csv
  run_build ogbn-arxiv gdpr_ogbn_arxiv --no_edges_csv
  run_build ogbn-products gdpr_ogbn_products --no_csv
fi

echo "[done] Corpora in ${OUT_ROOT}/corpora and NPZ files in ${OUT_ROOT}/data"
ls -lh "${OUT_ROOT}/data"
