#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _PathForSys
ROOT = _PathForSys(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import numpy as np

from gdprbench_unlearn.corpus import save_gdpr_npz_from_edge_index


def parse_args():
    p = argparse.ArgumentParser(description="Create a tiny synthetic GDPRBench-like NPZ for smoke tests.")
    p.add_argument("--out", default="data/tiny_gdpr.npz")
    p.add_argument("--nodes", type=int, default=1000)
    p.add_argument("--avg_degree", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.nodes
    e = n * args.avg_degree
    src = rng.integers(0, n, size=e)
    dst = rng.integers(0, n, size=e)
    edge = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_gdpr_npz_from_edge_index(edge, n, args.out, seed=args.seed)
    print(f"[done] wrote {args.out} nodes={n} directed_edges={edge.shape[1]}")


if __name__ == "__main__":
    main()
