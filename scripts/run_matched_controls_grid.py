#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _PathForSys
ROOT = _PathForSys(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path
import pandas as pd


def split_csv(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Run experiments on matched-control NPZ files produced by create_matched_controls.py.")
    p.add_argument("--summary_csv", required=True, help="matched_controls_summary.csv")
    p.add_argument("--out_dir", default="runs_matched_controls")
    p.add_argument("--methods", default="retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser")
    p.add_argument("--backbones", default="gcn,sage")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--unlearn_epochs", type=int, default=60)
    p.add_argument("--eraser_epochs", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    paths = df["path"].dropna().unique().tolist()
    methods = split_csv(args.methods)
    backbones = split_csv(args.backbones)
    seeds = [int(x) for x in split_csv(args.seeds)]
    runner = Path(__file__).resolve().parent / "run_experiment.py"
    total = len(paths) * len(methods) * len(backbones) * len(seeds)
    failed = []
    count = 0
    for npz, method, backbone, seed in itertools.product(paths, methods, backbones, seeds):
        count += 1
        cmd = [
            sys.executable, str(runner),
            "--npz", npz,
            "--out_dir", args.out_dir,
            "--method", method,
            "--backbone", backbone,
            "--forget_name", "npz:forget_idx",
            "--seed", str(seed),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--unlearn_epochs", str(args.unlearn_epochs),
            "--eraser_epochs", str(args.eraser_epochs),
            "--hidden_dim", str(args.hidden_dim),
            "--heads", str(args.heads),
        ]
        if args.quiet:
            cmd.append("--quiet")
        print("\n" + "=" * 100)
        print(f"[{count}/{total}] " + " ".join(shlex.quote(c) for c in cmd))
        if args.dry_run:
            continue
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            failed.append((cmd, proc.returncode))
            print(f"FAILED returncode={proc.returncode}")
            if not args.continue_on_error:
                sys.exit(proc.returncode)
    if failed:
        print(f"Completed with {len(failed)} failures")
        sys.exit(1)
    print("All matched-control runs completed or skipped by resume.")


if __name__ == "__main__":
    main()
