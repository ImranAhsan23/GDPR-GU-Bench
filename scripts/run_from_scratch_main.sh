#!/usr/bin/env bash
set -euo pipefail

# End-to-end main GDPRBench run.
# This script builds corpora, validates typed forget sets, creates matched controls,

OUT_ROOT=${1:-workspace}
DEVICE=${DEVICE:-cuda}
SEEDS=${SEEDS:-1,2,3}
EPOCHS=${EPOCHS:-300}
UNLEARN_EPOCHS=${UNLEARN_EPOCHS:-60}
HIDDEN_DIM=${HIDDEN_DIM:-128}
METHODS=${METHODS:-retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser}
FORGETS=${FORGETS:-all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label}

mkdir -p "${OUT_ROOT}/logs"

# 1) Build paper-scope corpora.
bash scripts/build_all_corpora.sh "${OUT_ROOT}" main

# 2) Construct-validity diagnostics.
for npz in "${OUT_ROOT}/data/gdpr_lastfm_asia.npz" "${OUT_ROOT}/data/gdpr_yelp.npz"; do
  python scripts/validate_benchmark.py \
    --npz "$npz" \
    --out_dir "${OUT_ROOT}/analysis/validity" \
    --groups "${FORGETS}" 2>&1 | tee -a "${OUT_ROOT}/logs/validity.log"
done

# 3) Matched controls.
for npz in "${OUT_ROOT}/data/gdpr_lastfm_asia.npz" "${OUT_ROOT}/data/gdpr_yelp.npz"; do
  python scripts/create_matched_controls.py \
    --npz "$npz" \
    --out_dir "${OUT_ROOT}/matched_controls" \
    --targets "${FORGETS}" \
    --modes uniform,label,degree,label_degree,label_degree_profile \
    --seed 42 2>&1 | tee -a "${OUT_ROOT}/logs/matched_controls.log"
done

# 4) Main unlearning runs.
python scripts/run_grid.py \
  --npzs "${OUT_ROOT}/data/gdpr_lastfm_asia.npz" \
  --out_dir "${OUT_ROOT}/runs/main_lastfm" \
  --methods "${METHODS}" \
  --backbones gcn,sage,gat \
  --forget_names "${FORGETS}" \
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --unlearn_epochs "${UNLEARN_EPOCHS}" \
  --eraser_epochs 200 \
  --hidden_dim "${HIDDEN_DIM}" \
  --heads 2 \
  --device "${DEVICE}" \
  --quiet \
  --continue_on_error 2>&1 | tee "${OUT_ROOT}/logs/runs_main_lastfm.log"

python scripts/run_grid.py \
  --npzs "${OUT_ROOT}/data/gdpr_yelp.npz" \
  --out_dir "${OUT_ROOT}/runs/main_yelp" \
  --methods "${METHODS}" \
  --backbones gcn,sage \
  --forget_names "${FORGETS}" \
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --unlearn_epochs "${UNLEARN_EPOCHS}" \
  --eraser_epochs 150 \
  --hidden_dim "${HIDDEN_DIM}" \
  --heads 1 \
  --device "${DEVICE}" \
  --quiet \
  --continue_on_error 2>&1 | tee "${OUT_ROOT}/logs/runs_main_yelp.log"

# 5) Optional matched-control grid; enable by setting RUN_MATCHED_GRID=1.
if [[ "${RUN_MATCHED_GRID:-0}" == "1" ]]; then
  python scripts/run_matched_controls_grid.py \
    --summary_csv "${OUT_ROOT}/matched_controls/lastfm_asia/matched_controls_summary.csv" \
    --out_dir "${OUT_ROOT}/runs/matched_lastfm" \
    --methods "${METHODS}" \
    --backbones gcn,sage \
    --seeds "${SEEDS}" \
    --epochs "${EPOCHS}" \
    --unlearn_epochs "${UNLEARN_EPOCHS}" \
    --eraser_epochs 200 \
    --hidden_dim "${HIDDEN_DIM}" \
    --device "${DEVICE}" \
    --quiet \
    --continue_on_error 2>&1 | tee "${OUT_ROOT}/logs/runs_matched_lastfm.log"

  python scripts/run_matched_controls_grid.py \
    --summary_csv "${OUT_ROOT}/matched_controls/yelp/matched_controls_summary.csv" \
    --out_dir "${OUT_ROOT}/runs/matched_yelp" \
    --methods "${METHODS}" \
    --backbones gcn,sage \
    --seeds "${SEEDS}" \
    --epochs "${EPOCHS}" \
    --unlearn_epochs "${UNLEARN_EPOCHS}" \
    --eraser_epochs 150 \
    --hidden_dim "${HIDDEN_DIM}" \
    --device "${DEVICE}" \
    --quiet \
    --continue_on_error 2>&1 | tee "${OUT_ROOT}/logs/runs_matched_yelp.log"
fi

# 6) Summaries and figures.
python scripts/summarize_results.py --run_dir "${OUT_ROOT}/runs/main_lastfm" --out_dir "${OUT_ROOT}/analysis/results_main_lastfm"
python scripts/summarize_results.py --run_dir "${OUT_ROOT}/runs/main_yelp" --out_dir "${OUT_ROOT}/analysis/results_main_yelp"
if [[ "${RUN_MATCHED_GRID:-0}" == "1" ]]; then
  python scripts/summarize_results.py --run_dir "${OUT_ROOT}/runs/matched_lastfm" --out_dir "${OUT_ROOT}/analysis/results_matched_lastfm"
  python scripts/summarize_results.py --run_dir "${OUT_ROOT}/runs/matched_yelp" --out_dir "${OUT_ROOT}/analysis/results_matched_yelp"
fi

echo "[done] End-to-end pipeline completed. Results are under ${OUT_ROOT}/analysis"
