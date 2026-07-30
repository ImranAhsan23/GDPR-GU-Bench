#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _PathForSys
ROOT = _PathForSys(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, list):
            continue
        else:
            out[key] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Flatten JSON metrics and generate summary CSV/figures.")
    p.add_argument("--run_dir", default="runs_fresh")
    p.add_argument("--out_dir", default="analysis_results")
    return p.parse_args()


def safe_group_summary(df: pd.DataFrame, metric: str, out_path: Path):
    cols = ["dataset.name", "args.backbone", "args.forget_name", "args.method"]
    if metric not in df.columns:
        return
    g = df.groupby(cols, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
    g.to_csv(out_path, index=False)


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path, title: str):
    if metric not in df.columns or df.empty:
        return
    # Plot one figure per dataset/backbone to keep figures readable.
    for (dataset, backbone), sub in df.groupby(["dataset.name", "args.backbone"], dropna=False):
        pivot = sub.pivot_table(index="args.forget_name", columns="args.method", values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        ax = pivot.plot(kind="bar", figsize=(12, 5))
        ax.set_title(f"{title}: {dataset}-{backbone}")
        ax.set_xlabel("Forget set")
        ax.set_ylabel(metric)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        name = f"{metric.replace('.', '_')}__{dataset}__{backbone}.png".replace("/", "_")
        plt.savefig(out_path / name, dpi=200)
        plt.close()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    figs = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in run_dir.rglob("metrics.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                rows.append(flatten(json.load(f)) | {"metrics_path": str(p)})
        except Exception as e:
            print(f"[warn] failed {p}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("No metrics.json files found.")
        return
    df.to_csv(out_dir / "all_metrics_flat.csv", index=False)
    summary_metrics = {
        "metrics.test_retained_macro_f1": "summary_utility_macro_f1.csv",
        "metrics.test_retained_macro_f1_abs_delta_from_retrain": "summary_delta_from_retrain.csv",
        "metrics.test_retained_pred_agreement_with_retrain": "summary_retrain_agreement.csv",
        "metrics.mi_forget_vs_test_raw_auc": "summary_mi_raw_auc.csv",
        "metrics.mi_forget_vs_test_auc_star": "summary_mi_auc_star.csv",
        "timing.update_seconds": "summary_update_seconds.csv",
        "timing.speedup_vs_retrain": "summary_speedup_vs_retrain.csv",
        "timing.peak_cuda_mb": "summary_peak_cuda_mb.csv",
    }
    for metric, fname in summary_metrics.items():
        safe_group_summary(df, metric, out_dir / fname)
    plot_metric(df, "metrics.test_retained_macro_f1", figs, "Retained-node macro-F1")
    plot_metric(df, "metrics.test_retained_macro_f1_abs_delta_from_retrain", figs, "Absolute macro-F1 delta from retrain")
    plot_metric(df, "metrics.mi_forget_vs_test_auc_star", figs, "MI AUC* forget-vs-test")
    plot_metric(df, "timing.speedup_vs_retrain", figs, "Speedup vs full retrain")
    print(f"[done] wrote {out_dir / 'all_metrics_flat.csv'} and figures in {figs}")


if __name__ == "__main__":
    main()
