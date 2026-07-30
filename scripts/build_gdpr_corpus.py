#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from gdprbench_unlearn.corpus_artifacts import save_corpus_from_edge_index, synthetic_edge_index


def parse_args():
    p = argparse.ArgumentParser(description="Build GDPRBench corpus artifacts and NPZ from public graph topologies.")
    p.add_argument("--dataset", required=True, choices=[
        "synthetic", "lastfm_asia", "yelp", "cora", "citeseer", "pubmed",
        "amazon_computers", "amazon_photo", "coauthor_cs", "coauthor_physics",
        "flickr", "ogbn-arxiv", "ogbn-products",
    ])
    p.add_argument("--name", default=None, help="Output name. Default: gdpr_<dataset-name>.")
    p.add_argument("--out_root", default="workspace", help="Creates <out_root>/data and <out_root>/corpora.")
    p.add_argument("--raw_root", default="workspace/raw_public", help="Where PyG/OGB raw downloads are cached.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--corruption_rate", type=float, default=0.25)
    p.add_argument("--mixture_profile", default="default", choices=["default", "balanced", "noise_heavy", "outdated_heavy", "wrongfact_heavy"])
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.20)
    p.add_argument("--synthetic_nodes", type=int, default=10000)
    p.add_argument("--synthetic_avg_degree", type=float, default=8.0)
    p.add_argument("--no_csv", action="store_true", help="Write NPZ only. Use when CSV artifacts are not needed.")
    p.add_argument("--no_edges_csv", action="store_true", help="Skip edges.csv. Useful for very large graphs.")
    p.add_argument("--no_symmetrize", action="store_true")
    p.add_argument("--force_redownload", action="store_true", help="Delete this dataset's raw/processed cache before loading.")
    return p.parse_args()


def _as_edge_index(edges: np.ndarray) -> np.ndarray:
    """Return a clean edge_index ndarray with shape [2, E]."""
    arr = np.asarray(edges)
    if arr.ndim != 2:
        raise ValueError(f"Expected edge array with 2 dimensions, got shape={arr.shape}")
    if arr.shape[0] == 2:
        out = arr
    elif arr.shape[1] == 2:
        out = arr.T
    else:
        raise ValueError(f"Cannot interpret edge array shape={arr.shape} as edge list")
    return out.astype(np.int64, copy=False)


def _maybe_delete(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _load_lastfm_npz_file(path: Path) -> Tuple[np.ndarray, int]:
    data = np.load(path, "r", allow_pickle=True)
    keys = set(data.files)
    if "edges" not in keys:
        raise ValueError(f"{path} does not contain 'edges'. Keys={sorted(keys)}")
    edge_index = _as_edge_index(data["edges"])
    if "features" in keys:
        num_nodes = int(np.asarray(data["features"]).shape[0])
    elif "target" in keys:
        num_nodes = int(np.asarray(data["target"]).shape[0])
    else:
        num_nodes = int(edge_index.max()) + 1
    return edge_index, num_nodes


def _try_load_lastfm_from_cache(raw: Path) -> Optional[Tuple[np.ndarray, int]]:
    candidates = [
        raw / "LastFMAsia" / "raw" / "lastfm_asia.npz",
        raw / "lastfm_asia.npz",
        raw / "LastFMAsia" / "lastfm_asia.npz",
    ]
    for c in candidates:
        if c.exists():
            print(f"[lastfm_asia] loading cached npz: {c}")
            return _load_lastfm_npz_file(c)
    return None


def _try_load_lastfm_from_snap_zip(raw: Path) -> Optional[Tuple[np.ndarray, int]]:
    import urllib.request
    import pandas as pd

    snap_dir = raw / "LastFMAsia_SNAP"
    zip_path = snap_dir / "lastfm_asia.zip"
    extract_dir = snap_dir / "extracted"
    snap_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        url = "https://snap.stanford.edu/data/lastfm_asia.zip"
        print(f"[lastfm_asia] PyG source unavailable; downloading SNAP fallback: {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"[lastfm_asia] SNAP fallback download failed: {e}")
            return None

    if not extract_dir.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            print(f"[lastfm_asia] Could not unzip SNAP fallback {zip_path}: {e}")
            return None

    # Prefer any npz with the PyG keys.
    for npz in extract_dir.rglob("*.npz"):
        try:
            return _load_lastfm_npz_file(npz)
        except Exception:
            pass

    # Otherwise parse CSV edge lists flexibly.
    edge_files = [p for p in extract_dir.rglob("*.csv") if "edge" in p.name.lower()]
    if not edge_files:
        print(f"[lastfm_asia] SNAP zip extracted but no edge CSV/NPZ was recognized under {extract_dir}")
        return None
    edge_file = edge_files[0]
    print(f"[lastfm_asia] loading SNAP edge CSV: {edge_file}")
    df = pd.read_csv(edge_file)
    lower_cols = {str(c).lower(): c for c in df.columns}
    if {"source", "target"}.issubset(lower_cols):
        edges = df[[lower_cols["source"], lower_cols["target"]]].to_numpy()
    elif {"src", "dst"}.issubset(lower_cols):
        edges = df[[lower_cols["src"], lower_cols["dst"]]].to_numpy()
    elif {"node_1", "node_2"}.issubset(lower_cols):
        edges = df[[lower_cols["node_1"], lower_cols["node_2"]]].to_numpy()
    else:
        # Use first two numeric-looking columns.
        edges = df.iloc[:, :2].to_numpy()
    edge_index = _as_edge_index(edges)

    # Use target CSV row count if available; otherwise max edge id + 1.
    target_files = [p for p in extract_dir.rglob("*.csv") if ("target" in p.name.lower() or "label" in p.name.lower())]
    if target_files:
        try:
            num_nodes = len(pd.read_csv(target_files[0]))
        except Exception:
            num_nodes = int(edge_index.max()) + 1
    else:
        num_nodes = int(edge_index.max()) + 1
    return edge_index, int(num_nodes)


def _load_graphsaint_adj(raw_dir: Path) -> Tuple[np.ndarray, int]:
    import scipy.sparse as sp

    adj_path = raw_dir / "adj_full.npz"
    if not adj_path.exists():
        raise FileNotFoundError(f"Missing GraphSAINT adjacency file: {adj_path}")
    f = np.load(adj_path, allow_pickle=True)
    if isinstance(f, np.lib.npyio.NpzFile):
        keys = set(f.files)
        if {"data", "indices", "indptr", "shape"}.issubset(keys):
            adj = sp.csr_matrix((f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"]))
        else:
            # Try scipy native loader as a fallback.
            adj = sp.load_npz(adj_path)
    else:
        adj = sp.load_npz(adj_path)
    coo = adj.tocoo()
    edge_index = np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
    return edge_index, int(adj.shape[0])


def _ensure_yelp_raw(raw: Path) -> Path:
    raw_dir = raw / "Yelp" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (raw_dir / "adj_full.npz").exists():
        return raw_dir
    try:
        from torch_geometric.data import download_google_url
    except Exception as e:
        raise RuntimeError("PyG is required to download Yelp. Install: pip install torch-geometric") from e
    files = {
        "adj_full.npz": "1Juwx8HtDwSzmVIJ31ooVa1WljI4U5JnA",
        "feats.npy": "1Zy6BZH_zLEjKlEFSduKE5tV9qqA_8VtM",
        "class_map.json": "1VUcBGr0T0-klqerjAjxRmAqFuld_SMWU",
        "role.json": "1NI5pa5Chpd-52eSmLW60OnB3WS5ikxq_",
    }
    for filename, file_id in files.items():
        out = raw_dir / filename
        if not out.exists():
            print(f"[yelp] downloading raw {filename}")
            download_google_url(file_id, str(raw_dir), filename)
    return raw_dir


def _try_load_yelp_manual(raw: Path) -> Optional[Tuple[np.ndarray, int]]:
    try:
        raw_dir = _ensure_yelp_raw(raw)
        print(f"[yelp] loading topology manually from {raw_dir / 'adj_full.npz'}")
        return _load_graphsaint_adj(raw_dir)
    except Exception as e:
        print(f"[yelp] manual topology loader failed: {e}")
        return None


def _try_load_flickr_manual(raw: Path) -> Optional[Tuple[np.ndarray, int]]:
    raw_dir = raw / "Flickr" / "raw"
    if not (raw_dir / "adj_full.npz").exists():
        return None
    try:
        print(f"[flickr] loading topology manually from {raw_dir / 'adj_full.npz'}")
        return _load_graphsaint_adj(raw_dir)
    except Exception as e:
        print(f"[flickr] manual topology loader failed: {e}")
        return None


def load_topology(dataset: str, raw_root: str, seed: int, synthetic_nodes: int, synthetic_avg_degree: float, force_redownload: bool = False):
    if dataset == "synthetic":
        edge_index = synthetic_edge_index(synthetic_nodes, synthetic_avg_degree, seed)
        return edge_index, synthetic_nodes

    raw = Path(raw_root)

    if force_redownload:
        cache_map = {
            "lastfm_asia": raw / "LastFMAsia",
            "yelp": raw / "Yelp",
            "flickr": raw / "Flickr",
            "cora": raw / "Cora",
            "citeseer": raw / "CiteSeer",
            "pubmed": raw / "PubMed",
            "amazon_computers": raw / "Amazon_Computers",
            "amazon_photo": raw / "Amazon_Photo",
            "coauthor_cs": raw / "Coauthor_CS",
            "coauthor_physics": raw / "Coauthor_Physics",
            "ogbn-arxiv": raw / "OGB" / "ogbn_arxiv",
            "ogbn-products": raw / "OGB" / "ogbn_products",
        }
        if dataset in cache_map:
            print(f"[force_redownload] deleting {cache_map[dataset]}")
            _maybe_delete(cache_map[dataset])

    # Dataset-specific manual loaders first for known fragile cases.
    if dataset == "yelp":
        manual = _try_load_yelp_manual(raw)
        if manual is not None:
            return manual
    if dataset == "lastfm_asia":
        cached = _try_load_lastfm_from_cache(raw)
        if cached is not None:
            return cached
        # Try PyG first below; if graphmining.ai DNS fails, SNAP fallback is used.

    try:
        import torch
        from torch_geometric.datasets import LastFMAsia, Yelp, Planetoid, Amazon, Coauthor, Flickr
        from torch_geometric.utils import to_undirected
    except Exception as e:
        raise RuntimeError(
            "PyTorch Geometric is required for this dataset. Install it with: pip install torch-geometric"
        ) from e

    try:
        if dataset == "lastfm_asia":
            ds = LastFMAsia(root=str(raw / "LastFMAsia"))
            data = ds[0]
        elif dataset == "yelp":
            ds = Yelp(root=str(raw / "Yelp"))
            data = ds[0]
        elif dataset in {"cora", "citeseer", "pubmed"}:
            name_map = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
            ds = Planetoid(root=str(raw / name_map[dataset]), name=name_map[dataset])
            data = ds[0]
        elif dataset in {"amazon_computers", "amazon_photo"}:
            name_map = {"amazon_computers": "Computers", "amazon_photo": "Photo"}
            ds = Amazon(root=str(raw / f"Amazon_{name_map[dataset]}"), name=name_map[dataset])
            data = ds[0]
        elif dataset in {"coauthor_cs", "coauthor_physics"}:
            name_map = {"coauthor_cs": "CS", "coauthor_physics": "Physics"}
            ds = Coauthor(root=str(raw / f"Coauthor_{name_map[dataset]}"), name=name_map[dataset])
            data = ds[0]
        elif dataset == "flickr":
            ds = Flickr(root=str(raw / "Flickr"))
            data = ds[0]
        elif dataset in {"ogbn-arxiv", "ogbn-products"}:
            try:
                from ogb.nodeproppred import PygNodePropPredDataset
            except Exception as e:
                raise RuntimeError("OGB is required. Install it with: pip install ogb") from e
            ds = PygNodePropPredDataset(name=dataset, root=str(raw / "OGB"))
            data = ds[0]
        else:  # pragma: no cover
            raise ValueError(dataset)
    except Exception as e:
        if dataset == "lastfm_asia":
            print(f"[lastfm_asia] PyG loader failed: {e}")
            cached = _try_load_lastfm_from_cache(raw)
            if cached is not None:
                return cached
            snap = _try_load_lastfm_from_snap_zip(raw)
            if snap is not None:
                return snap
        if dataset == "yelp":
            manual = _try_load_yelp_manual(raw)
            if manual is not None:
                return manual
        if dataset == "flickr":
            manual = _try_load_flickr_manual(raw)
            if manual is not None:
                return manual
        raise

    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.cpu()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes).cpu().numpy()
    return edge_index, num_nodes


def main():
    args = parse_args()
    name = args.name or f"gdpr_{args.dataset.replace('-', '_')}"
    edge_index, num_nodes = load_topology(
        args.dataset,
        args.raw_root,
        args.seed,
        args.synthetic_nodes,
        args.synthetic_avg_degree,
        args.force_redownload,
    )
    stats = save_corpus_from_edge_index(
        edge_index=edge_index,
        num_nodes=num_nodes,
        name=name,
        out_dir=args.out_root,
        seed=args.seed,
        corruption_rate=args.corruption_rate,
        mixture_profile=args.mixture_profile,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        symmetrize_edges=not args.no_symmetrize,
        write_csv=not args.no_csv,
        write_edges_csv=not args.no_edges_csv,
    )
    print("[done] GDPRBench corpus created")
    for k in ["name", "num_nodes", "num_edges_directed", "feature_dim", "corruption_rate", "forget_idx_train_only_size", "npz_path", "corpus_dir"]:
        print(f"  {k}: {stats[k]}")
    print(f"  error_counts: {stats['error_counts']}")


if __name__ == "__main__":
    main()
