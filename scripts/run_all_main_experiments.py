#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline_common import FORGETS, METHODS, METHODS_NO_ERASER, npz_paths, python_cmd, require_files, run_command


def parse_args():
    p = argparse.ArgumentParser(description="Run all main node-unlearning experiments using preset commands.")
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--suite", default="main", choices=["smoke", "main", "extended", "large", "all", "all_public"])
    p.add_argument("--preset", default="quick", choices=["quick", "paper"], help="quick is for testing; paper is full run.")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    p.add_argument("--seeds", default=None, help="Override preset seeds, e.g. 1,2,3")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--unlearn_epochs", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--methods", default=METHODS)
    p.add_argument("--forget_names", default=FORGETS)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_error", action="store_true", default=True)
    return p.parse_args()


def preset_values(preset: str):
    if preset == "quick":
        return {"seeds": "1", "epochs": 5, "unlearn_epochs": 2, "hidden_dim": 16, "eval_every": 1, "patience": 5}
    return {"seeds": "1,2,3", "epochs": 300, "unlearn_epochs": 60, "hidden_dim": 128, "eval_every": 10, "patience": 80}


def main() -> int:
    args = parse_args()
    vals = preset_values(args.preset)
    seeds = args.seeds or vals["seeds"]
    epochs = args.epochs or vals["epochs"]
    unlearn_epochs = args.unlearn_epochs or vals["unlearn_epochs"]
    hidden_dim = args.hidden_dim or vals["hidden_dim"]
    eval_every = vals["eval_every"]
    patience = vals["patience"]

    paths = npz_paths(args.workspace, args.suite)
    require_files(paths)
    logs = Path(args.workspace) / "logs"
    failed = []

    for p in paths:
        stem = p.stem.replace("gdpr_", "")
        is_yelp = "yelp" in stem
        is_large = any(x in stem for x in ["ogbn", "products", "flickr"])
        if args.preset == "quick":
            backbones = "gcn"
            methods = "retrain,no_unlearn,finetune_retain"
            forgets = "all_corrupted,wrong_fact"
        elif is_yelp or is_large:
            backbones = "gcn,sage"
            methods = args.methods if not is_large else METHODS_NO_ERASER
            forgets = args.forget_names
        else:
            backbones = "gcn,sage,gat"
            methods = args.methods
            forgets = args.forget_names
        out_dir = Path(args.workspace) / "runs" / f"main_{stem}"
        cmd = python_cmd(
            "run_grid.py",
            "--npzs", str(p),
            "--out_dir", str(out_dir),
            "--methods", methods,
            "--backbones", backbones,
            "--forget_names", forgets,
            "--seeds", seeds,
            "--epochs", str(epochs),
            "--unlearn_epochs", str(unlearn_epochs),
            "--eraser_epochs", "150" if is_yelp else "200",
            "--hidden_dim", str(hidden_dim),
            "--heads", "1" if is_yelp else "2",
            "--device", args.device,
            "--patience", str(patience),
            "--eval_every", str(eval_every),
            "--quiet",
            "--continue_on_error",
        )
        rc = run_command(cmd, log_file=logs / f"runs_main_{stem}.log", dry_run=args.dry_run)
        if rc != 0:
            failed.append((str(p), rc))
            if not args.continue_on_error:
                return rc
    print(f"\n[done] Main experiments are under {Path(args.workspace) / 'runs'}")
    if failed:
        print("Failures:", failed)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
