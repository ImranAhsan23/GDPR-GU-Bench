#!/usr/bin/env bash
set -euo pipefail
NPZ=${1:-data/gdpr_lastfm.npz}
OUT=${2:-runs_fresh_smoke}
python scripts/validate_benchmark.py --npz "$NPZ" --out_dir analysis_validity_smoke
python scripts/create_matched_controls.py --npz "$NPZ" --out_dir matched_controls_smoke --seed 42
python scripts/run_grid.py \
  --npzs "$NPZ" \
  --out_dir "$OUT" \
  --methods retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser \
  --backbones gcn,sage \
  --forget_names all_corrupted,wrong_fact \
  --seeds 1 \
  --epochs 20 \
  --unlearn_epochs 5 \
  --eraser_epochs 10 \
  --hidden_dim 64 \
  --device auto \
  --quiet
python scripts/summarize_results.py --run_dir "$OUT" --out_dir analysis_results_smoke
