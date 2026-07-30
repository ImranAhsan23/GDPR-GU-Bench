# GDPRBench end-to-end commands

This file gives the complete start-to-finish workflow: environment, corpus/NPZ generation, validation, matched controls, main unlearning runs, membership-inference/locality/efficiency summaries, optional extended datasets, and optional official-repository comparison setup.

---

## 1. Environment setup

```bash
conda create -n gdprbench python=3.10 -y
conda activate gdprbench

# Select the PyTorch command that matches your CUDA version.
# CUDA 12.1 example:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# Required only for public topology downloads/builders:
pip install torch-geometric ogb
```

Check GPU:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
```

---

## 2. Generate GDPRBench NPZ and CSV artifacts

The builder downloads the public graph topology, attaches the GDPRBench semi-synthetic personal-record schema, injects typed inaccuracies, and writes both audit artifacts and the `.npz` used by the unlearning runner.

### 2.1 Smoke-test corpus

```bash
python scripts/build_gdpr_corpus.py \
  --dataset synthetic \
  --name gdpr_synth_smoke \
  --out_root workspace_smoke \
  --synthetic_nodes 1000 \
  --synthetic_avg_degree 8 \
  --seed 1 \
  --no_edges_csv
```



### 2.2 Main paper corpora: LastFMAsia and Yelp

```bash
bash scripts/build_all_corpora.sh workspace main
```


By default, `edges.csv` is skipped for Yelp in `build_all_corpora.sh` to avoid a very large CSV. The `.npz` still contains the full `edge_index`. For final artifact release, build Yelp with edge CSV:

```bash
python scripts/build_gdpr_corpus.py \
  --dataset yelp \
  --name gdpr_yelp \
  --out_root workspace_release \
  --raw_root workspace/raw_public \
  --seed 1
```

```bash
bash scripts/build_all_corpora.sh workspace extended
```


Optional very-large corpus:

```bash
bash scripts/build_all_corpora.sh workspace large
```

This additionally creates `workspace/data/gdpr_ogbn_products.npz`; by default it is NPZ-only because the CSV artifacts are very large.

### 2.3 Individual dataset builder commands

```bash
# LastFMAsia
python scripts/build_gdpr_corpus.py --dataset lastfm_asia --name gdpr_lastfm_asia --out_root workspace --seed 1

# Yelp
python scripts/build_gdpr_corpus.py --dataset yelp --name gdpr_yelp --out_root workspace --seed 1 --no_edges_csv

# PubMed
python scripts/build_gdpr_corpus.py --dataset pubmed --name gdpr_pubmed --out_root workspace --seed 1

# Flickr
python scripts/build_gdpr_corpus.py --dataset flickr --name gdpr_flickr --out_root workspace --seed 1 --no_edges_csv

# OGBN-Arxiv
python scripts/build_gdpr_corpus.py --dataset ogbn-arxiv --name gdpr_ogbn_arxiv --out_root workspace --seed 1 --no_edges_csv

# OGBN-Products, optional large-scale stress test
python scripts/build_gdpr_corpus.py --dataset ogbn-products --name gdpr_ogbn_products --out_root workspace --seed 1 --no_csv
```

---

## 3. Verify each NPZ before experiments

```bash
python - <<'PY'
import glob, numpy as np
for p in sorted(glob.glob('workspace/data/*.npz')):
    d=np.load(p)
    print('\n', p)
    for k in ['x','y','edge_index','train_mask','val_mask','test_mask','forget_idx','forget_idx_all','is_corrupted','error_code']:
        print(k, d[k].shape, d[k].dtype)
PY
```

## 4. Construct-validity diagnostics

Run this for every corpus used in the paper:

```bash
FORGETS=all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label

for npz in workspace/data/gdpr_lastfm_asia.npz workspace/data/gdpr_yelp.npz workspace/data/gdpr_pubmed.npz workspace/data/gdpr_ogbn_arxiv.npz; do
  [ -f "$npz" ] || continue
  python scripts/validate_benchmark.py \
    --npz "$npz" \
    --out_dir workspace/analysis/validity \
    --groups "$FORGETS"
done
```


## 5. Matched-control forget sets

Generate controls for all main corpora:

```bash
FORGETS=all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label

for npz in workspace/data/gdpr_lastfm_asia.npz workspace/data/gdpr_yelp.npz; do
  python scripts/create_matched_controls.py \
    --npz "$npz" \
    --out_dir workspace/matched_controls \
    --targets "$FORGETS" \
    --modes uniform,label,degree,label_degree,label_degree_profile \
    --seed 42
done
```


## 6. One-run smoke test of training/unlearning

```bash
python scripts/run_experiment.py \
  --npz workspace_smoke/data/gdpr_synth_smoke.npz \
  --method finetune_retain \
  --backbone gcn \
  --forget_name wrong_fact \
  --epochs 5 \
  --unlearn_epochs 2 \
  --hidden_dim 16 \
  --patience 5 \
  --eval_every 1 \
  --device auto \
  --out_dir workspace_smoke/runs
```


## 7. Main unlearning experiments

### 7.1 Full LastFMAsia grid

```bash
python scripts/run_grid.py \
  --npzs workspace/data/gdpr_lastfm_asia.npz \
  --out_dir workspace/runs/main_lastfm \
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
  --quiet \
  --continue_on_error
```



### 7.2 Full Yelp grid, memory-conservative

```bash
python scripts/run_grid.py \
  --npzs workspace/data/gdpr_yelp.npz \
  --out_dir workspace/runs/main_yelp \
  --methods retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser \
  --backbones gcn,sage \
  --forget_names all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label \
  --seeds 1,2,3 \
  --epochs 300 \
  --unlearn_epochs 60 \
  --eraser_epochs 150 \
  --hidden_dim 128 \
  --heads 1 \
  --device cuda \
  --quiet \
  --continue_on_error
```


### 7.3 Optional Yelp-GAT run

Only run this if GPU memory is sufficient:

```bash
python scripts/run_grid.py \
  --npzs workspace/data/gdpr_yelp.npz \
  --out_dir workspace/runs/main_yelp_gat \
  --methods retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite \
  --backbones gat \
  --forget_names all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label \
  --seeds 1,2,3 \
  --epochs 200 \
  --unlearn_epochs 40 \
  --hidden_dim 64 \
  --heads 1 \
  --device cuda \
  --quiet \
  --continue_on_error
```

### 7.4 Extended-dataset grid

```bash
python scripts/run_grid.py \
  --npzs workspace/data/gdpr_pubmed.npz,workspace/data/gdpr_ogbn_arxiv.npz \
  --out_dir workspace/runs/extended \
  --methods retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite \
  --backbones gcn,sage \
  --forget_names all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label \
  --seeds 1,2,3 \
  --epochs 300 \
  --unlearn_epochs 60 \
  --hidden_dim 128 \
  --device cuda \
  --quiet \
  --continue_on_error
```

---

## 8. Matched-control unlearning experiments

```bash
python scripts/run_matched_controls_grid.py \
  --summary_csv workspace/matched_controls/lastfm_asia/matched_controls_summary.csv \
  --out_dir workspace/runs/matched_lastfm \
  --methods retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser \
  --backbones gcn,sage \
  --seeds 1,2,3 \
  --epochs 300 \
  --unlearn_epochs 60 \
  --eraser_epochs 200 \
  --hidden_dim 128 \
  --device cuda \
  --quiet \
  --continue_on_error
```

For Yelp:

```bash
python scripts/run_matched_controls_grid.py \
  --summary_csv workspace/matched_controls/yelp/matched_controls_summary.csv \
  --out_dir workspace/runs/matched_yelp \
  --methods retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser \
  --backbones gcn,sage \
  --seeds 1,2,3 \
  --epochs 300 \
  --unlearn_epochs 60 \
  --eraser_epochs 150 \
  --hidden_dim 128 \
  --device cuda \
  --quiet \
  --continue_on_error
```

---

## 9. Summarize results and generate figures

```bash
python scripts/summarize_results.py \
  --run_dir workspace/runs/main_lastfm \
  --out_dir workspace/analysis/results_main_lastfm

python scripts/summarize_results.py \
  --run_dir workspace/runs/main_yelp \
  --out_dir workspace/analysis/results_main_yelp

python scripts/summarize_results.py \
  --run_dir workspace/runs/matched_lastfm \
  --out_dir workspace/analysis/results_matched_lastfm

python scripts/summarize_results.py \
  --run_dir workspace/runs/matched_yelp \
  --out_dir workspace/analysis/results_matched_yelp
```

## 10. One-command main pipeline

This command builds the main corpora, runs validations, creates matched controls, runs main experiments, and writes summaries:

```bash
DEVICE=cuda SEEDS=1,2,3 EPOCHS=300 UNLEARN_EPOCHS=60 \
  bash scripts/run_from_scratch_main.sh workspace
```

To include matched-control runs in the same command:

```bash
DEVICE=cuda SEEDS=1,2,3 EPOCHS=300 UNLEARN_EPOCHS=60 RUN_MATCHED_GRID=1 \
  bash scripts/run_from_scratch_main.sh workspace
```

For a quick dry-scale test, reduce epochs and seeds:

```bash
DEVICE=cuda SEEDS=1 EPOCHS=20 UNLEARN_EPOCHS=5 RUN_MATCHED_GRID=0 \
  bash scripts/run_from_scratch_main.sh workspace_quick
```

---

## 11. Optional edge-unlearning appendix

The main GDPRBench claim is node-record deletion. Use this only as an appendix sanity check:

```bash
python scripts/run_experiment.py \
  --npz workspace/data/gdpr_lastfm_asia.npz \
  --task edge \
  --method utu_edge \
  --backbone gcn \
  --forget_name none \
  --forget_edge_count 5000 \
  --epochs 300 \
  --device cuda \
  --out_dir workspace/runs/edge_appendix
```


