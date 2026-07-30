#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _PathForSys
ROOT = _PathForSys(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
import json
import numpy as np

from gdprbench_unlearn.data import load_npz, get_forget_idx


def parse_args():
    p = argparse.ArgumentParser(description="Export GDPRBench NPZ arrays in a simple folder format for adapting official repos.")
    p.add_argument("--npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--forget_name", default="all_corrupted")
    p.add_argument("--train_only_forget", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    data = load_npz(args.npz)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    forget_idx = get_forget_idx(data, args.forget_name, train_only=args.train_only_forget)
    np.save(out / "x.npy", data.x.numpy())
    np.save(out / "y.npy", data.y.numpy())
    np.save(out / "edge_index.npy", data.edge_index.numpy())
    np.save(out / "train_mask.npy", data.train_mask.numpy())
    np.save(out / "val_mask.npy", data.val_mask.numpy())
    np.save(out / "test_mask.npy", data.test_mask.numpy())
    np.save(out / "forget_idx.npy", forget_idx.numpy())
    np.savetxt(out / "edges.csv", data.edge_index.t().numpy(), delimiter=",", fmt="%d", header="src,dst", comments="")
    np.savetxt(out / "labels.csv", np.stack([np.arange(data.num_nodes), data.y.numpy()], axis=1), delimiter=",", fmt="%d", header="node,label", comments="")
    meta = {
        "source_npz": args.npz,
        "dataset": data.name,
        "forget_name": args.forget_name,
        "train_only_forget": args.train_only_forget,
        "num_nodes": data.num_nodes,
        "num_edges_directed": data.num_edges_directed,
        "feature_dim": data.feature_dim,
        "num_classes": data.num_classes,
        "forget_size": int(forget_idx.numel()),
        "notes": "Official graph-unlearning repositories use heterogeneous loaders.",
    }
    with (out / "README_OFFICIAL_ADAPTER.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] wrote official-adapter arrays to {out}")


if __name__ == "__main__":
    main()
