# GDPR-GU Bench Implementation

This file gives the complete start-to-finish workflow: environment, corpus/NPZ generation, validation, matched controls, main unlearning runs, membership-inference/locality/efficiency summaries, optional extended datasets, and optional official-repository comparison setup.

### Backbones

- `gcn`: full-batch GCN implemented from scratch.
- `sage`: GraphSAGE-style mean aggregation implemented from scratch.
- `gat`: full-batch GAT implemented from scratch with native PyTorch `scatter_reduce_`.

### Node-unlearning methods in the core runner

- `retrain`: full retrain on the retained graph; gold reference.
- `no_unlearn`: remove the records/edges from the graph view but do not update weights.
- `finetune_retain`: fine-tune all model weights on retained training nodes.
- `last_layer`: freeze earlier layers and fine-tune the last layer on retained training nodes.
- `gradient_ascent`: gradient-ascent forgetting on the forget nodes, followed by retained fine-tuning.
- `gif_diag`: diagonal-Fisher influence-function update inspired by GIF, followed by optional retained fine-tuning.
- `agu_lite`: AGU-inspired affected-neighbor weighted retained fine-tuning. This is a transparent reproduction baseline, not the official AGU code.
- `graph_eraser`: GraphEraser-style shard training and mean-logit aggregation, implemented independently.

### Optional edge-unlearning method

- `utu_edge`: unlink-only edge unlearning. This is included for optional appendix experiments. GDPRBench's main benchmark contract is node-record deletion, so edge unlearning should not be part of the main claim unless you define edge-level error types.


## Complete from-scratch workflow

For the complete start-to-finish commands, including generating NPZ/CSV corpora from public graph topologies, running validity diagnostics, matched controls, unlearning grids, summaries, and optional official-repository exports, see [`END_TO_END_COMMANDS.md`](END_TO_END_COMMANDS.md).

The main corpus builder is:

```bash
python scripts/build_gdpr_corpus.py --dataset lastfm_asia --name gdpr_lastfm_asia --out_root workspace --seed 1
python scripts/build_gdpr_corpus.py --dataset yelp --name gdpr_yelp --out_root workspace --seed 1 --no_edges_csv
```

Or build the paper-scope corpora together:

```bash
bash scripts/build_all_corpora.sh workspace main
```

## 1. Environment setup

Create a clean environment:

```bash
conda create -n gdprbench-fresh python=3.10 -y
conda activate gdprbench-fresh
```

Install CUDA PyTorch for your driver/CUDA version. Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For CPU-only smoke tests:

```bash
pip install torch
pip install -r requirements.txt
```

Optional public-dataset builders require PyG/OGB:

```bash
pip install torch-geometric ogb
```

## 2. Put the GDPRBench NPZ files in `data/`

```bash
mkdir -p data
cp /path/to/gdpr_lastfm.npz data/
cp /path/to/gdpr_yelp.npz data/
```

Expected NPZ keys:

```text
x, y, edge_index, train_mask, val_mask, test_mask,
forget_idx, forget_idx_all, is_corrupted, error_code
```

Error-code convention:

```text
0 clean
1 wrong_fact
2 misattributed
3 outdated
4 invalid_combination
5 incorrect_derived_label  # legacy: noisy_label
```

## 3. Quick smoke test

Generate a tiny graph if you do not want to start with LastFM/Yelp:

```bash
python scripts/make_tiny_npz.py --out data/tiny_gdpr.npz --nodes 1000 --avg_degree 8 --seed 0
```

Run one training/unlearning job:

```bash
python scripts/run_experiment.py \
  --npz data/tiny_gdpr.npz \
  --method finetune_retain \
  --backbone gcn \
  --forget_name wrong_fact \
  --epochs 20 \
  --unlearn_epochs 5 \
  --device auto \
  --out_dir runs_tiny
```

`metrics.json` contains utility, agreement with retrain, MI raw AUC and AUC*, locality KL/agreement, update time, peak CUDA memory, and speedup vs retrain.

## 4. Construct-validity diagnostics

Run these before model experiments. They answer whether typed forget sets differ in label, degree, feature distribution, homophily, and affected-neighborhood size.

```bash
python scripts/validate_benchmark.py \
  --npz data/gdpr_lastfm.npz \
  --out_dir analysis_validity \
  --groups all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label

python scripts/validate_benchmark.py \
  --npz data/gdpr_yelp.npz \
  --out_dir analysis_validity \
  --groups all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label
```

## 5. Matched-control forget sets

```bash
python scripts/create_matched_controls.py \
  --npz data/gdpr_lastfm.npz \
  --out_dir matched_controls \
  --targets all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label \
  --modes uniform,label,degree,label_degree,label_degree_profile \
  --seed 42
```

```bash
python scripts/run_experiment.py \
  --npz matched_controls/lastfm/lastfm__control_wrong_fact__label_degree_profile__seed42.npz \
  --forget_name npz:forget_idx \
  --method finetune_retain \
  --backbone gcn \
  --seed 1 \
  --epochs 300 \
  --unlearn_epochs 60 \
  --device cuda \
  --out_dir runs_matched_controls
```

```bash
python scripts/run_matched_controls_grid.py \
  --summary_csv matched_controls/lastfm/matched_controls_summary.csv \
  --out_dir runs_matched_controls_lastfm \
  --methods retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser \
  --backbones gcn,sage \
  --seeds 1,2,3 \
  --epochs 300 \
  --unlearn_epochs 60 \
  --eraser_epochs 200 \
  --device cuda \
  --quiet
```

## 6. Full node-unlearning grid

Start with LastFM only:

```bash
python scripts/run_grid.py \
  --npzs data/gdpr_lastfm.npz \
  --out_dir runs_fresh_lastfm \
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
```

Then run Yelp. For large Yelp, begin with GCN and GraphSAGE; add GAT only if GPU memory allows:

```bash
python scripts/run_grid.py \
  --npzs data/gdpr_yelp.npz \
  --out_dir runs_fresh_yelp \
  --methods retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser \
  --backbones gcn,sage \
  --forget_names all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label \
  --seeds 1,2,3 \
  --epochs 300 \
  --unlearn_epochs 60 \
  --eraser_epochs 150 \
  --hidden_dim 128 \
  --device cuda \
  --quiet
```

Optional Yelp GAT memory-conservative run:

```bash
python scripts/run_grid.py \
  --npzs data/gdpr_yelp.npz \
  --out_dir runs_fresh_yelp_gat \
  --methods retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite \
  --backbones gat \
  --forget_names all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label \
  --seeds 1,2,3 \
  --epochs 200 \
  --unlearn_epochs 40 \
  --hidden_dim 64 \
  --heads 1 \
  --device cuda \
  --quiet
```

## 7. Summarize and plot results

```bash
python scripts/summarize_results.py --run_dir runs_fresh_lastfm --out_dir analysis_results_lastfm
python scripts/summarize_results.py --run_dir runs_fresh_yelp --out_dir analysis_results_yelp
```


## 8. Add more graph families

Use public graph topology and generate GDPRBench-style node records/errors:

```bash
# Citation graph
python scripts/build_public_corpus.py \
  --source pyg_planetoid \
  --name PubMed \
  --root public_raw \
  --out data/gdpr_pubmed.npz \
  --seed 1

# OGB graph, if resources allow
python scripts/build_public_corpus.py \
  --source ogb \
  --name ogbn-arxiv \
  --root public_raw \
  --out data/gdpr_ogbn_arxiv.npz \
  --seed 1
```

Then run the same grid commands on the new NPZ files.


