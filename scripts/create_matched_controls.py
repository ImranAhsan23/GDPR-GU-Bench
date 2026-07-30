#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _PathForSys
ROOT = _PathForSys(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import pandas as pd

from gdprbench_unlearn.analysis import create_matched_control, save_control_npz, validity_summary_for_group
from gdprbench_unlearn.data import get_forget_idx, load_npz
from gdprbench_unlearn.utils import ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Create matched-control forget sets for novelty/semantic-lift experiments.")
    p.add_argument("--npz", required=True)
    p.add_argument("--out_dir", default="matched_controls")
    p.add_argument("--targets", default="all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label")
    p.add_argument("--modes", default="uniform,label,degree,label_degree,label_degree_profile")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean_only", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    data = load_npz(args.npz)
    out = ensure_dir(Path(args.out_dir) / data.name)
    rows = []
    for target in [x.strip() for x in args.targets.split(",") if x.strip()]:
        target_idx = get_forget_idx(data, target, train_only=True)
        for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
            ctrl = create_matched_control(data, target_idx, mode=mode, seed=args.seed, clean_only=args.clean_only)
            out_path = out / f"{data.name}__control_{target}__{mode}__seed{args.seed}.npz"
            save_control_npz(args.npz, out_path, ctrl, extra_name="forget_idx_control")
            row = {
                "target": target,
                "mode": mode,
                "seed": args.seed,
                "control_size": int(ctrl.numel()),
                "path": str(out_path),
            }
            row.update(validity_summary_for_group(data, f"control_{target}_{mode}", ctrl, seed=args.seed))
            rows.append(row)
            print(f"[wrote] {out_path} size={ctrl.numel()}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "matched_controls_summary.csv", index=False)
    print(f"[done] summary={out / 'matched_controls_summary.csv'}")


if __name__ == "__main__":
    main()
