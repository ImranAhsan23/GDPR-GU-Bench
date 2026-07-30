#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _PathForSys
ROOT = _PathForSys(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from gdprbench_unlearn.analysis import validate_benchmark
from gdprbench_unlearn.data import load_npz
from gdprbench_unlearn.utils import ensure_dir, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Compute construct-validity diagnostics for GDPRBench forget sets.")
    p.add_argument("--npz", required=True)
    p.add_argument("--out_dir", default="analysis_validity")
    p.add_argument("--groups", default="all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    data = load_npz(args.npz)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    out = ensure_dir(Path(args.out_dir) / data.name)
    df = validate_benchmark(data, groups, seed=args.seed)
    df.to_csv(out / "validity_summary.csv", index=False)
    save_json({"npz": args.npz, "groups": groups, "rows": df.to_dict(orient="records")}, out / "validity_summary.json")
    print(df.to_string(index=False))
    print(f"[done] wrote {out / 'validity_summary.csv'}")


if __name__ == "__main__":
    main()
