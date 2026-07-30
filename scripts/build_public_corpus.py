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
    p = argparse.ArgumentParser(description="Build GDPRBench-style NPZs from public PyG/OGB graph topologies.")
    p.add_argument("--source", choices=["pyg_planetoid", "pyg_amazon", "pyg_coauthor", "pyg_flickr", "ogb"], required=True)
    p.add_argument("--name", required=True, help="Dataset name, e.g., Cora, CiteSeer, PubMed, Computers, Photo, CS, Physics, Flickr, ogbn-arxiv")
    p.add_argument("--root", default="public_raw")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corruption_rate", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.source.startswith("pyg"):
        try:
            import torch
            from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Flickr
            from torch_geometric.utils import to_undirected
        except Exception as e:
            raise RuntimeError("PyG is required for --source pyg_*. Install torch_geometric or use existing NPZ files.") from e
        if args.source == "pyg_planetoid":
            ds = Planetoid(root=args.root, name=args.name)
        elif args.source == "pyg_amazon":
            ds = Amazon(root=args.root, name=args.name)
        elif args.source == "pyg_coauthor":
            ds = Coauthor(root=args.root, name=args.name)
        elif args.source == "pyg_flickr":
            ds = Flickr(root=args.root)
        data = ds[0]
        edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes).cpu().numpy()
        n = int(data.num_nodes)
    else:
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            from torch_geometric.utils import to_undirected
        except Exception as e:
            raise RuntimeError("ogb and PyG are required for --source ogb.") from e
        ds = PygNodePropPredDataset(name=args.name, root=args.root)
        data = ds[0]
        edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes).cpu().numpy()
        n = int(data.num_nodes)
    save_gdpr_npz_from_edge_index(edge_index, n, args.out, seed=args.seed, corruption_rate=args.corruption_rate)
    print(f"[done] wrote {args.out} from {args.source}:{args.name} nodes={n} directed_edges={edge_index.shape[1]}")


if __name__ == "__main__":
    main()
