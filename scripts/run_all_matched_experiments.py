#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline_common import METHODS, METHODS_NO_ERASER, dataset_specs, python_cmd, run_command


def parse_args():
    p = argparse.ArgumentParser(description="Run matched-control unlearning experiments for all selected datasets.")
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--suite", default="main", choices=["smoke", "main", "extended", "large", "all", "all_public"])
    p.add_argument("--preset", default="quick", choices=["quick", "paper"])
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--unlearn_epochs", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_error", action="store_true", default=True)
    return p.parse_args()


def preset_values(preset: str):
    if preset == "quick":
        return {"seeds": "1", "epochs": 5, "unlearn_epochs": 2, "hidden_dim": 16}
    return {"seeds": "1,2,3", "epochs": 300, "unlearn_epochs": 60, "hidden_dim": 128}


def main() -> int:
    args = parse_args()
    vals = preset_values(args.preset)
    seeds = args.seeds or vals["seeds"]
    epochs = args.epochs or vals["epochs"]
    unlearn_epochs = args.unlearn_epochs or vals["unlearn_epochs"]
    hidden_dim = args.hidden_dim or vals["hidden_dim"]
    workspace = Path(args.workspace)
    logs = workspace / "logs"
    failed = []

    for spec in dataset_specs(args.suite):
        data_name = spec.name.replace("gdpr_", "")
        summary_csv = workspace / "matched_controls" / data_name / "matched_controls_summary.csv"
        if not summary_csv.exists():
            print(f"[skip] Missing {summary_csv}. Run create_all_matched_controls.py first.")
            failed.append((str(summary_csv), 1))
            if not args.continue_on_error:
                return 1
            continue
        is_large = any(x in data_name for x in ["yelp", "ogbn", "products", "flickr"])
        methods = METHODS_NO_ERASER if is_large and args.suite in {"extended", "large", "all"} else METHODS
        out_dir = workspace / "runs" / f"matched_{data_name}"
        cmd = python_cmd(
            "run_matched_controls_grid.py",
            "--summary_csv", str(summary_csv),
            "--out_dir", str(out_dir),
            "--methods", methods,
            "--backbones", "gcn,sage",
            "--seeds", seeds,
            "--epochs", str(epochs),
            "--unlearn_epochs", str(unlearn_epochs),
            "--eraser_epochs", "150" if "yelp" in data_name else "200",
            "--hidden_dim", str(hidden_dim),
            "--device", args.device,
            "--quiet",
            "--continue_on_error",
        )
        rc = run_command(cmd, log_file=logs / f"runs_matched_{data_name}.log", dry_run=args.dry_run)
        if rc != 0:
            failed.append((str(summary_csv), rc))
            if not args.continue_on_error:
                return rc
    print(f"\n[done] Matched-control experiments are under {workspace / 'runs'}")
    if failed:
        print("Failures/skips:", failed)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
