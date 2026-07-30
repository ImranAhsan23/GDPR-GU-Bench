#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _PathForSys
ROOT = _PathForSys(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from gdprbench_unlearn.data import (
    get_forget_idx,
    graph_summary,
    load_npz,
    mask_from_idx,
    remove_nodes_from_edges,
    retained_eval_mask,
    retained_train_mask,
    sample_random_edges,
)
from gdprbench_unlearn.metrics import classification_metrics, membership_inference_auc, prediction_agreement, symmetric_kl, locality_metrics
from gdprbench_unlearn.train import TrainConfig, save_checkpoint
from gdprbench_unlearn.unlearners import (
    ModelConfig,
    UnlearnConfig,
    SUPPORTED_METHODS,
    new_model,
    train_original,
    retrain_reference,
    run_unlearner,
)
from gdprbench_unlearn.utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one GDPRBench graph-unlearning experiment from scratch.")
    p.add_argument("--npz", required=True, help="Path to GDPRBench .npz")
    p.add_argument("--dataset_name", default=None)
    p.add_argument("--out_dir", default="runs_fresh")
    p.add_argument("--method", required=True, choices=SUPPORTED_METHODS)
    p.add_argument("--backbone", default="gcn", choices=["gcn", "gat", "sage"])
    p.add_argument("--forget_name", default="all_corrupted")
    p.add_argument("--file_forget_idx", default=None)
    p.add_argument("--forget_npz_key", default="forget_idx")
    p.add_argument("--train_only_forget", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")

    # Model/training.
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--quiet", action="store_true")

    # Unlearning.
    p.add_argument("--unlearn_epochs", type=int, default=60)
    p.add_argument("--unlearn_lr", type=float, default=0.005)
    p.add_argument("--unlearn_weight_decay", type=float, default=5e-4)
    p.add_argument("--ga_epochs", type=int, default=10)
    p.add_argument("--ga_lr", type=float, default=0.001)
    p.add_argument("--gif_damping", type=float, default=1e-3)
    p.add_argument("--gif_scale", type=float, default=1.0)
    p.add_argument("--gif_retain_ft_epochs", type=int, default=10)
    p.add_argument("--agu_khop", type=int, default=2)
    p.add_argument("--agu_alpha", type=float, default=2.0)
    p.add_argument("--eraser_shards", type=int, default=10)
    p.add_argument("--eraser_epochs", type=int, default=200)
    p.add_argument("--eraser_partition", default="degree_label", choices=["random", "degree_label"])

    # Evaluation.
    p.add_argument("--no_retrain_ref", action="store_true", help="Skip retrain reference to save time. Agreement metrics become NaN.")
    p.add_argument("--mi_max_points", type=int, default=20000)
    p.add_argument("--locality_k", type=int, default=2)

    # Optional edge unlearning.
    p.add_argument("--task", default="node", choices=["node", "edge"])
    p.add_argument("--forget_edge_count", type=int, default=1000)
    p.add_argument("--forget_edge_idx_file", default=None)
    return p.parse_args()


def make_run_dir(args: argparse.Namespace, data_name: str) -> Path:
    f = args.forget_name.replace(":", "_")
    return Path(args.out_dir) / data_name / f / args.backbone / args.method / f"seed_{args.seed}"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    data = load_npz(args.npz, name=args.dataset_name)
    run_dir = make_run_dir(args, data.name)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and args.resume and not args.force:
        print(f"[skip] metrics already exist: {metrics_path}")
        return
    ensure_dir(run_dir)

    model_cfg = ModelConfig(args.backbone, args.hidden_dim, args.num_layers, args.dropout, args.heads)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        eval_every=args.eval_every,
        verbose=not args.quiet,
    )
    unlearn_cfg = UnlearnConfig(
        method=args.method,
        unlearn_epochs=args.unlearn_epochs,
        unlearn_lr=args.unlearn_lr,
        unlearn_weight_decay=args.unlearn_weight_decay,
        ga_epochs=args.ga_epochs,
        ga_lr=args.ga_lr,
        gif_damping=args.gif_damping,
        gif_scale=args.gif_scale,
        gif_retain_ft_epochs=args.gif_retain_ft_epochs,
        agu_khop=args.agu_khop,
        agu_alpha=args.agu_alpha,
        eraser_shards=args.eraser_shards,
        eraser_epochs=args.eraser_epochs,
        eraser_partition=args.eraser_partition,
        verbose=not args.quiet,
    )

    forget_idx = get_forget_idx(
        data,
        args.forget_name,
        train_only=args.train_only_forget,
        file_forget_idx=args.file_forget_idx,
        npz_key=args.forget_npz_key,
    )
    if args.task == "edge":
        if args.forget_edge_idx_file:
            forget_edge_idx = torch.as_tensor(np.load(args.forget_edge_idx_file), dtype=torch.long)
        else:
            forget_edge_idx = sample_random_edges(data.edge_index, args.forget_edge_count, seed=args.seed)
    else:
        forget_edge_idx = None

    cache_root = Path(args.out_dir) / "_cache" / data.name / args.backbone / f"seed_{args.seed}"
    original_model = None
    original_logits = None
    original_meta: Dict = {}
    if args.method != "retrain":
        print(f"[original] cache={cache_root / 'original'}")
        original_model, original_logits, original_meta = train_original(data, model_cfg, train_cfg, device, cache_root / "original", resume=args.resume and not args.force)

    retrain_logits = None
    retrain_meta: Dict = {}
    retrain_model = None
    if not args.no_retrain_ref:
        print(f"[retrain] forget={args.forget_name}")
        retrain_model, retrain_logits, retrain_meta = retrain_reference(data, forget_idx, model_cfg, train_cfg, device, cache_root / f"retrain_{args.forget_name.replace(':','_')}", resume=args.resume and not args.force)

    if args.method == "original":
        method_model, method_logits, method_meta = original_model, original_logits, {"update_seconds": 0.0, "peak_cuda_mb": 0.0, "note": "original model, no deletion applied"}
    elif args.method == "retrain":
        method_model, method_logits, method_meta = retrain_model, retrain_logits, {"update_seconds": retrain_meta.get("train_result", {}).get("train_seconds", float("nan")), "peak_cuda_mb": retrain_meta.get("train_result", {}).get("peak_cuda_mb", 0.0), "note": "full retrain reference"}
    else:
        print(f"[unlearn] method={args.method}")
        method_model, method_logits, method_meta = run_unlearner(
            args.method,
            original_model,
            data,
            forget_idx,
            model_cfg,
            train_cfg,
            unlearn_cfg,
            device,
            run_dir,
            seed=args.seed,
            resume=args.resume and not args.force,
            forget_edge_idx=forget_edge_idx,
        )

    # Metrics on CPU.
    y = data.y.cpu()
    method_logits = method_logits.cpu()
    test_retained = retained_eval_mask(data.test_mask.cpu(), forget_idx.cpu())
    val_retained = retained_eval_mask(data.val_mask.cpu(), forget_idx.cpu())
    train_retained = retained_train_mask(data, forget_idx.cpu())
    forget_mask = mask_from_idx(data.num_nodes, forget_idx.cpu(), device="cpu")

    metrics: Dict = {}
    metrics.update({f"test_retained_{k}": v for k, v in classification_metrics(method_logits, y, test_retained).items()})
    metrics.update({f"val_retained_{k}": v for k, v in classification_metrics(method_logits, y, val_retained).items()})
    metrics.update({f"forget_{k}": v for k, v in classification_metrics(method_logits, y, forget_mask).items()})

    if retrain_logits is not None:
        retrain_logits = retrain_logits.cpu()
        metrics["test_retained_pred_agreement_with_retrain"] = prediction_agreement(method_logits, retrain_logits, test_retained)
        metrics["test_retained_symmetric_kl_with_retrain"] = symmetric_kl(method_logits, retrain_logits, test_retained)
        retrain_f1 = classification_metrics(retrain_logits, y, test_retained)["macro_f1"]
        metrics["test_retained_macro_f1_abs_delta_from_retrain"] = abs(metrics["test_retained_macro_f1"] - retrain_f1)
    else:
        metrics["test_retained_pred_agreement_with_retrain"] = float("nan")
        metrics["test_retained_symmetric_kl_with_retrain"] = float("nan")
        metrics["test_retained_macro_f1_abs_delta_from_retrain"] = float("nan")

    mi_forget = membership_inference_auc(method_logits, y, forget_mask, test_retained, seed=args.seed, max_points_per_class=args.mi_max_points)
    mi_retain = membership_inference_auc(method_logits, y, train_retained, test_retained, seed=args.seed + 17, max_points_per_class=args.mi_max_points)
    metrics.update({f"mi_forget_vs_test_{k}": v for k, v in mi_forget.items()})
    metrics.update({f"mi_retaintrain_vs_test_{k}": v for k, v in mi_retain.items()})

    loc = locality_metrics(
        before_logits=original_logits.cpu() if original_logits is not None else None,
        after_logits=method_logits,
        retrain_logits=retrain_logits if retrain_logits is not None else None,
        edge_index=data.edge_index.cpu(),
        forget_idx=forget_idx.cpu(),
        num_nodes=data.num_nodes,
        k=args.locality_k,
    )
    metrics.update({f"locality_{k}": v for k, v in loc.items()})

    retrain_seconds = retrain_meta.get("train_result", {}).get("train_seconds", float("nan")) if retrain_meta else float("nan")
    update_seconds = float(method_meta.get("update_seconds", float("nan")))
    speedup = float(retrain_seconds / update_seconds) if update_seconds and update_seconds > 0 and not np.isnan(retrain_seconds) else float("nan")

    result = {
        "dataset": graph_summary(data),
        "args": vars(args),
        "model_config": model_cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "unlearn_config": unlearn_cfg.__dict__,
        "forget": {
            "name": args.forget_name,
            "size": int(forget_idx.numel()),
            "train_only": bool(args.train_only_forget),
        },
        "sizes": {
            "num_edges_directed_original": int(data.edge_index.shape[1]),
            "num_edges_directed_after_node_delete": int(remove_nodes_from_edges(data.edge_index, forget_idx, data.num_nodes).shape[1]),
            "train_retained_nodes": int(train_retained.sum().item()),
            "test_retained_nodes": int(test_retained.sum().item()),
        },
        "timing": {
            "original_train_seconds": original_meta.get("train_result", {}).get("train_seconds", float("nan")) if original_meta else float("nan"),
            "retrain_seconds": retrain_seconds,
            "update_seconds": update_seconds,
            "speedup_vs_retrain": speedup,
            "peak_cuda_mb": float(method_meta.get("peak_cuda_mb", 0.0)),
        },
        "method_meta": method_meta,
        "metrics": metrics,
    }
    torch.save(method_logits, run_dir / "logits.pt")
    if method_model is not None:
        save_checkpoint(method_model, run_dir / "model.pt", extra={"method_meta": method_meta})
    save_json(result, metrics_path)
    print(f"[done] wrote {metrics_path}")


if __name__ == "__main__":
    main()
