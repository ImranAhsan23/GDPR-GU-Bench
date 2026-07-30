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


def split_csv(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Run a GDPRBench experiment grid with checkpoint/resume.")
    p.add_argument("--npzs", required=True, help="Comma-separated NPZ paths")
    p.add_argument("--out_dir", default="runs_fresh")
    p.add_argument("--methods", default="retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser")
    p.add_argument("--backbones", default="gcn,sage,gat")
    p.add_argument("--forget_names", default="all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--unlearn_epochs", type=int, default=60)
    p.add_argument("--eraser_epochs", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--extra", default="", help="Extra string appended to each run_experiment.py command")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    runner = root / "run_experiment.py"
    npzs = split_csv(args.npzs)
    methods = split_csv(args.methods)
    backbones = split_csv(args.backbones)
    forgets = split_csv(args.forget_names)
    seeds = [int(x) for x in split_csv(args.seeds)]
    total = len(npzs) * len(methods) * len(backbones) * len(forgets) * len(seeds)
    print(f"Planned runs: {total}")
    failed = []
    count = 0
    for npz, method, backbone, forget, seed in itertools.product(npzs, methods, backbones, forgets, seeds):
        count += 1
        cmd = [
            sys.executable, str(runner),
            "--npz", npz,
            "--out_dir", args.out_dir,
            "--method", method,
            "--backbone", backbone,
            "--forget_name", forget,
            "--seed", str(seed),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--unlearn_epochs", str(args.unlearn_epochs),
            "--eraser_epochs", str(args.eraser_epochs),
            "--hidden_dim", str(args.hidden_dim),
            "--heads", str(args.heads),
            "--num_layers", str(args.num_layers),
            "--patience", str(args.patience),
            "--eval_every", str(args.eval_every),
        ]
        if args.quiet:
            cmd.append("--quiet")
        if args.extra:
            cmd += shlex.split(args.extra)
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
        print(f"\nCompleted with {len(failed)} failures:")
        for cmd, code in failed:
            print(code, " ".join(shlex.quote(c) for c in cmd))
        sys.exit(1)
    print("\nAll requested runs completed or skipped by resume.")


if __name__ == "__main__":
    main()
