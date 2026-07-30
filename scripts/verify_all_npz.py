#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_common import dataset_specs, npz_paths

REQUIRED = ["x", "y", "edge_index", "train_mask", "val_mask", "test_mask", "forget_idx", "forget_idx_all", "is_corrupted", "error_code"]


def parse_args():
    p = argparse.ArgumentParser(description="Verify all GDPRBench NPZ files before experiments.")
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--suite", default="main", choices=["smoke", "main", "extended", "large", "all", "all_public"])
    p.add_argument("--out_csv", default=None, help="Default: <workspace>/analysis/npz_verification.csv")
    return p.parse_args()


def verify(path: Path) -> dict:
    row = {"path": str(path), "exists": path.exists(), "ok": False}
    if not path.exists():
        row["error"] = "missing file"
        return row
    try:
        d = np.load(path, allow_pickle=True)
        missing = [k for k in REQUIRED if k not in d.files]
        row["keys"] = ",".join(d.files)
        if missing:
            row["error"] = "missing keys: " + ",".join(missing)
            return row
        x, y, ei = d["x"], d["y"], d["edge_index"]
        n = int(x.shape[0])
        row.update({
            "num_nodes": n,
            "feature_dim": int(x.shape[1]) if x.ndim == 2 else None,
            "num_edges_directed": int(ei.shape[1]) if ei.ndim == 2 else None,
            "num_classes": int(np.max(y) + 1) if y.size else 0,
            "train": int(d["train_mask"].sum()),
            "val": int(d["val_mask"].sum()),
            "test": int(d["test_mask"].sum()),
            "forget_idx": int(d["forget_idx"].shape[0]),
            "forget_idx_all": int(d["forget_idx_all"].shape[0]),
            "corrupted": int(d["is_corrupted"].sum()),
        })
        if ei.ndim != 2 or ei.shape[0] != 2:
            row["error"] = f"edge_index shape must be [2,E], got {ei.shape}"
            return row
        for k in ["y", "train_mask", "val_mask", "test_mask", "is_corrupted", "error_code"]:
            if d[k].shape[0] != n:
                row["error"] = f"{k} length {d[k].shape[0]} != x nodes {n}"
                return row
        row["ok"] = True
        row["error"] = ""
        return row
    except Exception as e:
        row["error"] = repr(e)
        return row


def main() -> int:
    args = parse_args()
    out_csv = Path(args.out_csv) if args.out_csv else Path(args.workspace) / "analysis" / "npz_verification.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in npz_paths(args.workspace, args.suite):
        rows.append(verify(p))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\n[done] wrote {out_csv}")
    return 0 if bool(df["ok"].all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
