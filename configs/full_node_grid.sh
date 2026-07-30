#!/usr/bin/env bash
set -euo pipefail
NPZS=${1:-data/gdpr_lastfm.npz,data/gdpr_yelp.npz}
OUT=${2:-runs_fresh_full}
python scripts/run_grid.py \
  --npzs "$NPZS" \
  --out_dir "$OUT" \
  --methods retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser \
  --backbones gcn,sage,gat \
  --forget_names all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label \
  --seeds 1,2,3 \
  --epochs 300 \
  --unlearn_epochs 60 \
  --eraser_epochs 200 \
  --hidden_dim 128 \
  --heads 2 \
  --device cuda \
  --quiet
python scripts/summarize_results.py --run_dir "$OUT" --out_dir analysis_results_full
